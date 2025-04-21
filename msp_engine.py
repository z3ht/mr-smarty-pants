import argparse
import base64
import io
import os
import threading
import asyncio
from collections import deque
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import AsyncIterator, Optional, List

import flet as ft
import numpy as np
import pywinctl as pwc
import sounddevice as sd
import soundfile as sf
from PIL import Image
from dotenv import load_dotenv
from mss import mss
from openai import AsyncOpenAI
from skimage.metrics import structural_similarity as ssim
import tiktoken

# --- Parse Arguments ---
parser = argparse.ArgumentParser(description="Run Mr. Smarty Pants")
parser.add_argument("--window", type=str, help="Fuzzy match window title to capture")
args = parser.parse_args()

WINDOW_NAME = args.window if args.window else None


# --- Setup environment ---
load_dotenv()
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY; set it in .env file")

AUDIO_CHUNK_S = int(os.getenv("AUDIO_CHUNK_S", "3"))
VOICE_NAME = os.getenv("VOICE_NAME", "onyx")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
SCREENSHOT_INTERVAL_S = float(os.getenv("SCREENSHOT_INTERVAL_S", "0.1"))
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "500"))
SCREENSHOT_SIMILARITY_THRESHOLD_PCT = float(os.getenv("SCREENSHOT_SIMILARITY_THRESHOLD_PCT", "0.95"))
MAX_SCREENSHOT_AGE_S = int(os.getenv("MAX_SCREENSHOT_AGE_S", "30"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
MAX_HISTORY_AGE_S = int(os.getenv("MAX_HISTORY_AGE_S", "300"))
NUM_LATEST_SCREENSHOTS = int(os.getenv("NUM_LATEST_SCREENSHOTS", "1"))
TOKEN_LIMIT_PER_M = int(os.getenv("TOKEN_LIMIT_PER_M", "200_000"))

client = AsyncOpenAI(api_key=API_KEY)

# --- Global shutdown event ---
shutdown_event = threading.Event()


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model(CHAT_MODEL)
    tokens = encoding.encode(text)
    return len(tokens)


def estimate_message_tokens(message: dict) -> int:
    content = message.get("content")
    total_tokens = 0

    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                total_tokens += count_tokens(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:image/"):
                    base64_data = url.split(",", 1)[-1]
                    byte_length = (len(base64_data) * 3) // 4
                    estimated_tokens = max(1, byte_length // 4)
                    total_tokens += estimated_tokens
    elif isinstance(content, str):
        total_tokens += count_tokens(content)

    total_tokens += 3  # small constant overhead per message

    return total_tokens


class TokenUsageTracker:
    def __init__(self):
        self.usage = deque()  # stores (timestamp: float, tokens: int)

    def _now(self) -> float:
        return datetime.now().timestamp()

    def _prune_old_usage(self):
        cutoff_time = datetime.now() - timedelta(minutes=1)
        cutoff_timestamp = cutoff_time.timestamp()
        while self.usage and self.usage[0][0] < cutoff_timestamp:
            self.usage.popleft()

    def add_tokens(self, tokens: int):
        now = self._now() + 1.0  # Add 1 second to account for send delay
        self.usage.append((now, tokens))
        self._prune_old_usage()

    def tokens_used_last_minute(self) -> int:
        self._prune_old_usage()
        return sum(tokens for _, tokens in self.usage)

    def tokens_available(self) -> int:
        return TOKEN_LIMIT_PER_M - self.tokens_used_last_minute()

    def seconds_until_tokens_available(self, needed_tokens: int, safety_margin_s: float = 0.10) -> float:
        """
        Returns how many seconds until needed_tokens are available within the rolling 60s window.
        Returns 0.0 if enough tokens are available right now.
        """
        self._prune_old_usage()

        available = self.tokens_available()
        if available >= needed_tokens or not self.usage:
            return 0.0

        now = self._now()
        # Simulate future pruning
        usage_list = list(self.usage)
        idx = 0

        while idx < len(usage_list):
            oldest_timestamp, oldest_tokens = usage_list[idx]
            idx += 1
            # Advance time to when this usage entry will be pruned
            simulated_now = oldest_timestamp + 60
            # Recompute available tokens after removing all earlier entries
            remaining_usage = usage_list[idx:]
            used_tokens = sum(tokens for _, tokens in remaining_usage)
            available_tokens = TOKEN_LIMIT_PER_M - used_tokens

            if available_tokens >= needed_tokens:
                wait_time = simulated_now - now + safety_margin_s
                return max(0.0, wait_time)

        # If even after 60s everything there isn't enough (unlikely), return wait after last entry
        last_timestamp, _ = usage_list[-1]
        wait_time = (last_timestamp + 60) - now + safety_margin_s
        return max(0.0, wait_time)

    def __repr__(self):
        used = self.tokens_used_last_minute()
        available = self.tokens_available()
        return f"<TokenUsageTracker used={used} available={available} limit={TOKEN_LIMIT_PER_M} tokens/min>"


token_tracker = TokenUsageTracker()


class ContextManager:
    def __init__(self, system_prompt: dict):
        self.system_prompt = system_prompt
        self.history: List[dict] = []
        self.latest_screenshots: List[dict] = []

    def _now(self) -> datetime:
        return datetime.now()

    def set_latest_screenshots(self, screenshots: List[bytes]) -> None:
        parts = []
        for img_bytes in screenshots:
            b64 = base64.b64encode(img_bytes).decode()
            parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        self.latest_screenshots = parts

    def add_user_message(self, text: str) -> None:
        self.history.append({
            "role": "user",
            "content": text,
            "timestamp": self._now()
        })

    def add_assistant_message(self, text: str) -> None:
        self.history.append({
            "role": "assistant",
            "content": text,
            "timestamp": self._now()
        })

    def build_context(self, now: Optional[datetime] = None) -> List[dict]:
        now = now or self._now()
        cutoff = now - timedelta(seconds=MAX_HISTORY_AGE_S)
        recent = [m for m in self.history if m.get("timestamp", now) >= cutoff]

        context = [self.system_prompt]
        assembled = []

        for i, message in enumerate(reversed(recent[-MAX_HISTORY_MESSAGES:])):
            role = message["role"]
            text_content = message["content"]

            is_most_recent = (i == 0)
            screenshots_to_include = (
                self.latest_screenshots[-NUM_LATEST_SCREENSHOTS:]
                if is_most_recent and role == "user" and self.latest_screenshots
                else []
            )

            content = [{"type": "text", "text": text_content}] + screenshots_to_include

            assembled.append({
                "role": role,
                "content": content
            })

        full_context = context + list(reversed(assembled))

        self.latest_screenshots = []  # Clear after building
        return full_context

    def estimate_total_tokens(self, now: Optional[datetime] = None) -> int:
        return sum(estimate_message_tokens(m) for m in self.build_context(now))


_last_selected = None
_last_bbox = None
async def take_screenshot() -> bytes:
    global _last_selected, _last_bbox

    def get_bbox(win):
        return {
            "left": win.left,
            "top": win.top,
            "width": win.width,
            "height": win.height,
        }

    if _last_selected is None:
        # First time: find window
        all_windows = pwc.getAllWindows()
        matches = [w for w in all_windows if WINDOW_NAME.lower() in w.title.lower()]
        if not matches:
            print(f"[screenshot] No matching window found for '{WINDOW_NAME}'")
            return b""
        _last_selected = matches[0]
        _last_bbox = get_bbox(_last_selected)
    else:
        # Window exists: check if its bbox changed
        current_bbox = get_bbox(_last_selected)
        if current_bbox != _last_bbox:
            print("[screenshot] Window bbox changed, refreshing...")
            all_windows = pwc.getAllWindows()
            matches = [w for w in all_windows if WINDOW_NAME.lower() in w.title.lower()]
            if not matches:
                print(f"[screenshot] No matching window found for '{WINDOW_NAME}'")
                return b""
            _last_selected = matches[0]
            _last_bbox = get_bbox(_last_selected)

    with mss() as sct:
        raw = sct.grab(_last_bbox)
        img = Image.frombytes("RGB", raw.size, raw.rgb)

    max_dim = 1600
    if img.width > max_dim or img.height > max_dim:
        img.thumbnail((max_dim, max_dim))

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # save_dir = "screenshots"
    # os.makedirs(save_dir, exist_ok=True)
    # filename = os.path.join(save_dir, f"screenshot_{timestamp}.jpg")
    # img.save(filename, format="JPEG", quality=92, optimize=True)
    # print(f"[screenshot] Saved to {filename}")

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92, optimize=True)
    return buf.getvalue()


class MicRecorder:
    def __init__(self, queue: asyncio.Queue[np.ndarray]):
        self.queue = queue
        self.loop = asyncio.get_running_loop()
        self.thread = None
        self.stream = None
        self.running = threading.Event()  # 👈 Local shutdown control

    def start(self):
        if self.thread and self.thread.is_alive():
            print("[MicRecorder] Recorder already running.")
            return

        self.running.set()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print("[MicRecorder] Recorder started.")

    def stop(self):
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.stream:
            try:
                self.stream.close()
                print("[MicRecorder] Stream closed.")
            except Exception as e:
                print(f"[MicRecorder] Error closing stream: {e}")
        print("[MicRecorder] Recorder stopped.")

    def _record_loop(self):
        samplerate = 16000
        frame_duration_ms = 100
        frame_samples = int(samplerate * frame_duration_ms / 1000)
        recording_buffer = []
        speaking = False
        silence_frames = 0
        silence_threshold = 500
        max_silence_frames = AUDIO_CHUNK_S * 10

        self.stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
        )

        with self.stream:
            while self.running.is_set():
                frame, _ = self.stream.read(frame_samples)
                frame = frame.flatten()
                rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2))
                if rms > silence_threshold:
                    if not speaking:
                        speaking = True
                    recording_buffer.append(frame)
                    silence_frames = 0
                elif speaking:
                    recording_buffer.append(frame)
                    silence_frames += 1
                    if silence_frames > max_silence_frames:
                        audio_data = np.concatenate(recording_buffer)
                        asyncio.run_coroutine_threadsafe(self.queue.put(audio_data), self.loop)
                        speaking = False
                        recording_buffer = []
                        silence_frames = 0


async def mic_stream(page: ft.Page) -> AsyncIterator[str]:
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    recorder = None

    try:
        while not shutdown_event.is_set():
            if page.mic_enabled:
                if recorder is None:
                    recorder = MicRecorder(queue)
                    recorder.start()
            else:
                if recorder:
                    recorder.stop()
                    recorder = None
                await asyncio.sleep(0.01)
                continue

            try:
                recording = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            buf = io.BytesIO()
            sf.write(buf, recording, 16000, format="WAV", subtype="PCM_16")
            buf.seek(0)
            buf.name = "speech.wav"

            needed_tokens = 1000  # Rough guess for transcription input
            wait_s = token_tracker.seconds_until_tokens_available(needed_tokens=needed_tokens)
            if wait_s > 0.0:
                print(
                    f"[tokens] Waiting {wait_s:.2f}s before transcribing (used={token_tracker.tokens_used_last_minute()} / limit={TOKEN_LIMIT_PER_M})"
                )
                await asyncio.sleep(wait_s)

            print(f"[tokens] Using {needed_tokens} tokens for transcription (available={token_tracker.tokens_available()})")
            token_tracker.add_tokens(needed_tokens)

            # --- Send transcription request ---
            text_obj = await client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=buf,
                response_format="text",
                language="en"
            )

            yield text_obj.strip()
    finally:
        if recorder:
            recorder.stop()


class Speaker:
    def __init__(self):
        self.current_stream = None
        self.speaking = False
        self.stopping = False
        self._speak_task = None

    async def speak(self, text: str, status_button: ft.ElevatedButton, page: ft.Page):
        if not page.speech_enabled:
            print("[speak] Speech disabled; skipping speaking")
            return
        await self.stop()
        self._speak_task = asyncio.create_task(
            self._internal_speak(text, status_button, page)
        )

    async def _internal_speak(self, text: str, status_button: ft.ElevatedButton, page: ft.Page):
        self.stopping = False
        self.speaking = True
        status_button.text = "Stop"
        status_button.tooltip = "Stop AI speech"
        status_button.bgcolor = ft.Colors.BLUE_GREY_900
        page.update()

        try:
            # --- Estimate tokens needed ---
            needed_tokens = max(50, len(text) // 4)

            # --- Wait if necessary ---
            wait_s = token_tracker.seconds_until_tokens_available(needed_tokens=needed_tokens)
            if wait_s > 0.0:
                print(
                    f"[tokens] Waiting {wait_s:.2f}s before speaking (used={token_tracker.tokens_used_last_minute()} / limit={TOKEN_LIMIT_PER_M})"
                )
                await asyncio.sleep(wait_s)

            print(f"[tokens] Using {needed_tokens} tokens for speech (available={token_tracker.tokens_available()})")
            token_tracker.add_tokens(needed_tokens)

            # --- Send TTS request ---
            resp = await client.audio.speech.create(
                model=TTS_MODEL,
                voice=VOICE_NAME,
                input=text,
                response_format="pcm"
            )

            audio_bytes = resp.content
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)

            audio_queue = Queue()
            chunk_size = 480

            for i in range(0, len(pcm), chunk_size):
                audio_queue.put(pcm[i:i + chunk_size])
            audio_queue.put(None)

            def callback(outdata, frames, time, status):
                try:
                    if self.stopping:
                        raise sd.CallbackStop()
                    chunk = audio_queue.get_nowait()
                    if chunk is None:
                        raise sd.CallbackStop()
                    outdata[:len(chunk)] = chunk.reshape(-1, 1)
                    if len(chunk) < len(outdata):
                        outdata[len(chunk):] = 0
                except Empty:
                    outdata.fill(0)
                except sd.CallbackStop:
                    outdata.fill(0)
                    raise

            self.current_stream = sd.OutputStream(
                samplerate=28000,  # ~1.4x speed
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
                callback=callback
            )
            self.current_stream.start()

            while self.current_stream.active:
                await asyncio.sleep(0.05)

        except Exception as e:
            print(f"[error] TTS failed: {e}")
        finally:
            if self.current_stream:
                try:
                    self.current_stream.close()
                except Exception as e:
                    print(f"[error] closing output stream: {e}")
            self.current_stream = None
            self.speaking = False

            status_button.text = "Send"
            status_button.tooltip = "Send a message"
            status_button.bgcolor = None
            page.update()

    async def stop(self):
        if self._speak_task:
            self.stopping = True
            try:
                self._speak_task.cancel()
                await asyncio.sleep(0.05)
            except Exception:
                pass
            self._speak_task = None


def main(page: ft.Page):
    page.title = "Mr. Smarty Pants Assistant"
    page.vertical_alignment = "start"

    chat = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    input_field = ft.TextField(
        label="Type your message…",
        expand=True,
        multiline=True,
        min_lines=1,
        max_lines=15,
        shift_enter=True,
    )
    send_button = ft.ElevatedButton("Send")

    speech_button = ft.IconButton(
        content=ft.Text("🔇"),
        tooltip="Toggle speech (TTS)"
    )
    page.speech_enabled = False

    mic_button = ft.IconButton(
        content=ft.Text("🎙️"),
        tooltip="Toggle microphone (STT)"
    )
    page.mic_enabled = False

    screenshot_button = ft.IconButton(
        content=ft.Text("📷"),
        tooltip="Toggle screenshots",
    )
    page.include_screenshots = False
    page.screenshot_buffer = []

    end_conversation_button = ft.IconButton(
            content=ft.Image(
            src=os.path.join(ASSETS_DIR, "logo.png"),
            width=50,
            height=50,
            fit=ft.ImageFit.CONTAIN,
        ),
        tooltip="New conversation",
        padding=0
    )

    speaker = Speaker()
    page.send_task = None
    page.thinking_task = None

    system_prompt = {
        "role": "system",
        "content": (
            "You are Mr. Smarty Pants, an AI assistant. Speak clearly, stay concise, "
            "and format code examples inside triple backticks."
        )
    }
    context_manager = ContextManager(system_prompt)

    # --- Handle user send ---
    async def send_message(user_text: str):
        if page.thinking_task:
            try:
                page.thinking_task.cancel()
            except Exception as e:
                print(f"[error] cancelling previous thinking task: {e}")
            page.thinking_task = None

        chat.controls = [
            c for c in chat.controls[-10:]
            if not (isinstance(c, ft.Container) and isinstance(c.data, dict) and c.data.get("temporary"))
        ]
        page.update()

        chat.controls.append(
            ft.Container(
                content=ft.Text(value=f'{user_text}', selectable=True, color="#888888"),
                padding=2,
                alignment=ft.alignment.center_left,
                expand=True,
            )
        )
        page.update()

        context_manager.set_latest_screenshots(page.screenshot_buffer)
        page.screenshot_buffer.clear()

        context_manager.add_user_message(user_text)
        full_context = context_manager.build_context()
        needed_tokens = sum(estimate_message_tokens(m) for m in full_context)

        thinking_text = ft.Text("Thinking...", italic=True, selectable=True, color="#888888")
        thinking_bubble = ft.Container(
            content=thinking_text,
            padding=2,
            alignment=ft.alignment.center_left,
            data={"temporary": True}
        )
        chat.controls.append(thinking_bubble)
        page.update()

        async def show_thinking_status(waiting_for_tokens: bool, needed_tokens: int):
            if waiting_for_tokens:
                while True:
                    waiting_s = int(token_tracker.seconds_until_tokens_available(needed_tokens))
                    if waiting_s <= 0:
                        break
                    thinking_text.value = f"Waiting for tokens... ({waiting_s}s)"
                    page.update()
                    await asyncio.sleep(0.5)
                thinking_text.value = "Generating response..."
                page.update()
            else:
                await asyncio.sleep(0.5)
                thinking_text.value = "Generating response..."
                page.update()

        wait_s = token_tracker.seconds_until_tokens_available(needed_tokens)
        waiting_for_tokens = wait_s > 0.0
        page.thinking_task = asyncio.create_task(show_thinking_status(waiting_for_tokens, needed_tokens))

        if waiting_for_tokens:
            print(f"[tokens] Waiting {wait_s:.2f}s (needed={needed_tokens})")
            await asyncio.sleep(wait_s)
        else:
            print(f"[tokens] Using {needed_tokens} tokens")

        token_tracker.add_tokens(needed_tokens)

        try:
            stream = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=full_context,
                stream=True,
            )

            if page.thinking_task:
                try:
                    page.thinking_task.cancel()
                except Exception as e:
                    print(f"[error] cancelling thinking task after wait: {e}")
                page.thinking_task = None

        except Exception as e:
            thinking_text.value = f"Error: {e}"
            thinking_text.color = ft.colors.RED
            page.update()
            return

        full_reply = ""
        first_chunk = True
        ai_message = None

        async for delta in stream:
            part = delta.choices[0].delta.content or ""
            if not part:
                continue
            full_reply += part

            if first_chunk:
                ai_message = thinking_text
                ai_message.value = part.strip()
                ai_message.italic = False
                ai_message.color = "#FFFFFF"

                if isinstance(ai_message.parent, ft.Container):
                    if isinstance(ai_message.parent.data, dict):
                        ai_message.parent.data["temporary"] = False

                page.update()
                first_chunk = False
            else:
                ai_message.value = full_reply
                page.update()

        full_reply = full_reply.strip()

        if full_reply:
            if len(full_reply) <= 100:
                await speaker.speak(full_reply, send_button, page)
            else:
                wait_s = token_tracker.seconds_until_tokens_available(needed_tokens=500)
                if wait_s > 0.0:
                    print(f"[tokens] Waiting {wait_s:.2f}s before summarizing")
                    await asyncio.sleep(wait_s)

                print(f"[tokens] Using 500 tokens for summarizing")
                token_tracker.add_tokens(500)

                try:
                    summary_resp = await client.chat.completions.create(
                        model=CHAT_MODEL,
                        messages=[
                            {"role": "system",
                             "content": "Summarize the following text into a single concise sentence for speaking aloud."},
                            {"role": "user", "content": full_reply}
                        ]
                    )
                    summary_text = summary_resp.choices[0].message.content.strip()
                    if summary_text:
                        print(f"[summary] {summary_text}")
                        await speaker.speak(summary_text, send_button, page)
                    else:
                        print("[summary] No summary generated")
                except Exception as e:
                    print(f"[summary error] {e}")

            context_manager.add_assistant_message(full_reply)

    async def handle_send(e=None):
        print("[handle_send] send triggered")
        user_text = input_field.value.strip()
        if not user_text:
            return
        input_field.value = ""
        page.update()

        if page.send_task and not page.send_task.done():
            try:
                print("[handle_send] Cancelling previous send task")
                page.send_task.cancel()
            except Exception as e:
                print(f"[error] problem cancelling previous send task: {e}")

        if page.thinking_task:
            try:
                page.thinking_task.cancel()
                print("[cancel] Previous thinking_task canceled.")
            except Exception as e:
                print(f"[error] cancelling previous thinking task: {e}")
            page.thinking_task = None

        page.send_task = asyncio.create_task(send_message(user_text))

    async def handle_send_button(e=None):
        if speaker.speaking:
            print("[handle_send_button] Stopping speech instead of sending message")
            await speaker.stop()
        else:
            await handle_send(e)

    def toggle_speech(e=None):
        page.speech_enabled = not page.speech_enabled
        if page.speech_enabled:
            speech_button.content = ft.Text("🔈")
            speech_button.tooltip = "Speech Enabled (click to mute speech)"
            print("[toggle_speech] Speech enabled.")
        else:
            speech_button.content = ft.Text("🔇")
            speech_button.tooltip = "Speech Muted (click to unmute speech)"
            print("[toggle_speech] Speech disabled.")
        speech_button.update()

    def toggle_screenshots(e=None):
        page.include_screenshots = not page.include_screenshots
        screenshot_button.content.value = "📸" if page.include_screenshots else "📷"
        screenshot_button.content.tooltip = "Taking screenshots" if page.include_screenshots else "Screenshots disabled"
        screenshot_button.update()
        print(f"[toggle_screenshots] Screenshots {'enabled' if page.include_screenshots else 'disabled'}.")

    def toggle_mic(e=None):
        page.mic_enabled = not page.mic_enabled
        if page.mic_enabled:
            mic_button.content = ft.Text("🔴")
            mic_button.tooltip = "Listening (click to mute)"
            print("[toggle_mic] Mic enabled.")
        else:
            mic_button.content = ft.Text("🎙️")
            mic_button.tooltip = "Muted (click to unmute)"
            print("[toggle_mic] Mic disabled.")
        mic_button.update()
        page.update()

    def end_conversation(e=None):
        print("[end_conversation] Ending conversation...")

        chat.controls.clear()
        context_manager.history.clear()
        page.screenshot_buffer.clear()

        page.update()

    send_button.on_click = handle_send_button
    input_field.on_submit = handle_send
    mic_button.on_click = toggle_mic
    speech_button.on_click = toggle_speech
    end_conversation_button.on_click = end_conversation

    screenshot_button.on_click = toggle_screenshots

    page.add(
        ft.Container(
            ft.Row(
                [end_conversation_button],
                alignment=ft.MainAxisAlignment.START,
                spacing=0,
            ),
            padding=0,
            margin=0,
        ),
        chat,
        ft.Row([input_field, send_button, speech_button, mic_button, screenshot_button]),
    )

    async def mic_listener():
        async for speech in mic_stream(page):
            await send_message(speech)

    async def screenshot_collector():
        last_image = None
        while not shutdown_event.is_set():
            await asyncio.sleep(SCREENSHOT_INTERVAL_S)
            if page.include_screenshots and WINDOW_NAME:
                try:
                    current_bytes = await take_screenshot()
                    if not current_bytes:
                        continue

                    img = Image.open(io.BytesIO(current_bytes)).convert("L")
                    img = img.resize((320, 200))
                    img_arr = np.asarray(img, dtype=np.float32)

                    if last_image is not None:
                        similarity, _ = ssim(img_arr, last_image, data_range=255.0, full=True)
                        if similarity > SCREENSHOT_SIMILARITY_THRESHOLD_PCT:
                            continue

                    page.screenshot_buffer.append(current_bytes)
                    last_image = img_arr

                except Exception as e:
                    print(f"[error] screenshot collector: {e}")

    page.mic_task = page.run_task(mic_listener)
    page.screenshot_task = page.run_task(screenshot_collector)

    async def on_close(e=None):
        print("[shutdown] Cleaning up...")

        shutdown_event.set()

        await speaker.stop()

        if page.mic_task:
            try:
                page.mic_task.cancel()
            except Exception as e:
                print(f"[error] cancelling mic task: {e}")

        if page.screenshot_task:
            try:
                page.screenshot_task.cancel()
            except Exception as e:
                print(f"[error] cancelling screenshot task: {e}")

        if speaker.current_stream:
            try:
                speaker.current_stream.close()
            except Exception as e:
                print(f"[error] closing stream on shutdown: {e}")

        await asyncio.sleep(0.1)
        print("[shutdown] Done.")

    page.on_close = on_close


ft.app(target=main, assets_dir="assets")
