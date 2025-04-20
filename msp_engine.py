import argparse
import base64
import io
import os
import threading
import asyncio
from collections import deque
from datetime import datetime, timedelta
from queue import Queue, Empty
from typing import AsyncIterator

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

SELECTED_WINDOW = None

if args.window:
    all_windows = pwc.getAllWindows()
    matches = [w for w in all_windows if args.window.lower() in w.title.lower()]
    if not matches:
        available_titles = [w.title for w in all_windows]
        raise SystemExit(f"No matching window found for '{args.window}'. Available windows: {available_titles}")
    SELECTED_WINDOW = matches[0]
    print(f"[window capture] Selected window: {SELECTED_WINDOW.title}")
else:
    print("[window capture] No window selected; screenshots will be disabled.")

# --- Setup environment ---
load_dotenv()

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
                image_base64 = part.get("image_url", {}).get("url", "")
                estimated_tokens = int(0.35 * len(image_base64))
                total_tokens += estimated_tokens
    elif isinstance(content, str):
        total_tokens += count_tokens(content)

    total_tokens += 3

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


async def take_screenshot() -> bytes:
    if not SELECTED_WINDOW:
        return b""
    with mss() as sct:
        bbox = {
            "left": SELECTED_WINDOW.left,
            "top": SELECTED_WINDOW.top,
            "width": SELECTED_WINDOW.width,
            "height": SELECTED_WINDOW.height,
        }
        raw = sct.grab(bbox)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img.thumbnail((800, 600))

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_dir = "screenshots"
        # os.makedirs(save_dir, exist_ok=True)
        # filename = os.path.join(save_dir, f"screenshot_{timestamp}.jpg")
        #
        # # Save the image as a JPEG file
        # img.save(filename, format="JPEG", quality=80)
        # print(f"[screenshot] Saved to {filename}")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=65)
        return buf.getvalue()


class MicRecorder:
    def __init__(self, queue: asyncio.Queue[np.ndarray]):
        self.queue = queue
        self.loop = asyncio.get_running_loop()
        self.thread = None
        self.stream = None

    def start(self):
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()

    def stop(self):
        shutdown_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.stream:
            try:
                self.stream.close()
            except Exception as e:
                print(f"[error] closing input stream: {e}")

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
            while not shutdown_event.is_set():
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


async def mic_stream() -> AsyncIterator[str]:
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    recorder = MicRecorder(queue)
    recorder.start()

    try:
        while not shutdown_event.is_set():
            try:
                recording = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            buf = io.BytesIO()
            sf.write(buf, recording, 16000, format="WAV", subtype="PCM_16")
            buf.seek(0)
            buf.name = "speech.wav"

            # --- Estimate tokens needed for transcription ---
            needed_tokens = 1000  # Rough guess for transcription input

            # --- Wait if necessary ---
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
        recorder.stop()

class Speaker:
    def __init__(self):
        self.current_stream = None
        self.speaking = False
        self.stopping = False
        self._speak_task = None

    async def speak(self, text: str, status_button: ft.IconButton, page: ft.Page):
        await self.stop()
        self._speak_task = asyncio.create_task(
            self._internal_speak(text, status_button, page)
        )

    async def _internal_speak(self, text: str, status_button: ft.IconButton, page: ft.Page):
        self.stopping = False
        self.speaking = True
        status_button.content.value = "🛑"
        status_button.tooltip = "Stop AI speech"
        status_button.disabled = False
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
            chunk_size = 240

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
                blocksize=240,
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

            status_button.content.value = "🎤"
            status_button.tooltip = "Listening"
            status_button.disabled = False
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
    status_button = ft.IconButton(content=ft.Text("🎤"), tooltip="Listening", disabled=False)
    screenshot_button = ft.IconButton(content=ft.Text("📸"), tooltip="Toggle Screenshots")

    page.send_task = None
    page.thinking_task = None

    page.include_screenshots = True
    page.screenshot_buffer = []

    speaker = Speaker()
    page.mic_task = None
    page.screenshot_task = None

    history = [
        {"role": "system", "content": (
            "You are Mr. Smarty Pants, an AI assistant. Speak clearly, stay concise, "
            "and format code examples inside triple backticks."
        )}
    ]

    async def send_message(user_text: str):
        now = datetime.now()

        # --- Cancel any old thinking task ---
        if page.thinking_task:
            try:
                page.thinking_task.cancel()
            except Exception as e:
                print(f"[error] cancelling previous thinking task: {e}")
            page.thinking_task = None

        # --- Clear old temporary thinking/waiting bubbles ---
        chat.controls = [
            c for c in chat.controls[-10:]
            if not (isinstance(c, ft.Container) and isinstance(c.data, dict) and c.data.get("temporary"))
        ]
        page.update()

        # --- Add user's new message ---
        chat.controls.append(
            ft.Container(
                content=ft.Text(f"You: {user_text}", selectable=True, color="#BBBBBB"),
                padding=8,
                alignment=ft.alignment.center_left,
                expand=True
            )
        )
        page.update()

        # --- Prepare attachments (screenshots) ---
        attachments = []
        if page.screenshot_buffer:
            shot_time, shot_bytes = page.screenshot_buffer[-1]
            if (now - shot_time).total_seconds() <= MAX_SCREENSHOT_AGE_S:
                b64 = base64.b64encode(shot_bytes).decode()
                attachments.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                })
            page.screenshot_buffer = [(shot_time, shot_bytes)]
        else:
            page.screenshot_buffer = []

        # --- Update history ---
        history.append({
            "role": "user",
            "content": [{"type": "text", "text": user_text}] + attachments
        })

        # --- Build full context ---
        system_prompt = history[0]
        user_assistant_messages = history[1:]
        full_context = [system_prompt] + [
            {k: v for k, v in m.items() if k in {"role", "content"}} for m in user_assistant_messages
        ]

        # --- Estimate needed tokens ---
        needed_tokens = sum(estimate_message_tokens(m) for m in full_context)

        # --- Setup thinking bubble ---
        thinking_text = ft.Text("Thinking...", italic=True, selectable=True, color="#888888")
        thinking_bubble = ft.Container(
            content=thinking_text,
            padding=8,
            alignment=ft.alignment.center_left,
            data={"temporary": True}
        )
        chat.controls.append(thinking_bubble)
        page.update()

        # --- Start merged thinking / waiting status ---
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
            print(
                f"[tokens] Waiting {wait_s:.2f}s (needed={needed_tokens}, used={token_tracker.tokens_used_last_minute()} / limit={TOKEN_LIMIT_PER_M})")
            await asyncio.sleep(wait_s)
        else:
            print(f"[tokens] Using {needed_tokens} tokens (available={token_tracker.tokens_available()})")

        # --- After waiting: officially consume tokens ---
        token_tracker.add_tokens(needed_tokens)

        # --- Send the chat completion request ---
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

        # --- Stream the AI's response ---
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

        # --- Handle speaking or summarizing ---
        if full_reply:
            if len(full_reply) <= 100:
                await speaker.speak(full_reply, status_button, page)
            else:
                wait_s = token_tracker.seconds_until_tokens_available(needed_tokens=500)
                if wait_s > 0.0:
                    print(
                        f"[tokens] Waiting {wait_s:.2f}s before summarizing (used={token_tracker.tokens_used_last_minute()} / limit={TOKEN_LIMIT_PER_M})")
                    await asyncio.sleep(wait_s)

                print(f"[tokens] Using 500 tokens for summarization (available={token_tracker.tokens_available()})")
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
                        await speaker.speak(summary_text, status_button, page)
                    else:
                        print("[summary] No summary generated")
                except Exception as e:
                    print(f"[summary error] {e}")

            history.append({
                "role": "assistant",
                "content": full_reply
            })

    async def handle_send(e=None):
        user_text = input_field.value.strip()
        if not user_text:
            return
        input_field.value = ""
        page.update()

        # Cancel any previous send_task if running
        if page.send_task and not page.send_task.done():
            try:
                print(f"[handle_send] cancelling previous send task")
                page.send_task.cancel()
            except Exception as ex:
                print(f"[error] problem cancelling previous send task: {ex}")

        # Cancel previous thinking task too
        if page.thinking_task:
            try:
                page.thinking_task.cancel()
                print("[cancel] Previous thinking_task canceled.")
            except Exception as ex:
                print(f"[error] cancelling previous thinking task: {ex}")
            page.thinking_task = None

        # Start new send_message
        page.send_task = asyncio.create_task(send_message(user_text))

    async def stop_speaking(e=None):
        if speaker.speaking:
            print("[stop] Stopping AI speech…")
            await speaker.stop()

    def toggle_screenshots(e):
        page.include_screenshots = not page.include_screenshots
        screenshot_button.content.value = "📸" if page.include_screenshots else "📷"
        screenshot_button.update()

    send_button.on_click = handle_send
    input_field.on_submit = handle_send
    status_button.on_click = stop_speaking
    screenshot_button.on_click = toggle_screenshots

    page.add(chat, ft.Row([input_field, send_button, status_button, screenshot_button]))

    async def mic_listener():
        async for speech in mic_stream():
            await send_message(speech)

    async def screenshot_collector():
        last_image = None
        while not shutdown_event.is_set():
            await asyncio.sleep(SCREENSHOT_INTERVAL_S)
            if page.include_screenshots and SELECTED_WINDOW:
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
                            print("[motion] No significant change, skipping screenshot.")
                            continue

                    now = datetime.now()
                    page.screenshot_buffer.append((now, current_bytes))
                    print(f"[motion] Saved screenshot at {now.strftime('%Y%m%d_%H%M%S')}.")
                    last_image = img_arr

                except Exception as e:
                    print(f"[error] screenshot collector: {e}")

    page.mic_task = page.run_task(mic_listener)
    page.screenshot_task = page.run_task(screenshot_collector)

    async def on_close(e):
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


ft.app(target=main)
