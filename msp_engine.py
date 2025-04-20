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
        now = self._now()
        self.usage.append((now, tokens))
        self._prune_old_usage()

    def tokens_used_last_minute(self) -> int:
        self._prune_old_usage()
        return sum(tokens for _, tokens in self.usage)

    def tokens_available(self) -> int:
        return TOKEN_LIMIT_PER_M - self.tokens_used_last_minute()

    async def wait_for_tokens(self, needed_tokens: int, safety_margin_s: float = 0.10):
        """
        Wait until there is room for needed_tokens within the rolling 60s window, w/ small safety margin.
        """
        while True:
            available = self.tokens_available()
            if available >= needed_tokens:
                break

            # Estimate when the oldest token will fall out of the window
            if self.usage:
                oldest_timestamp, _ = self.usage[0]
                now = self._now()
                wait_time = (oldest_timestamp + 60) - now + safety_margin_s
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                else:
                    await asyncio.sleep(safety_margin_s)
            else:
                await asyncio.sleep(safety_margin_s)

    def __repr__(self):
        used = self.tokens_used_last_minute()
        available = self.tokens_available()
        return (f"<TokenUsageTracker used={used} available={available} "
                f"limit={TOKEN_LIMIT_PER_M} tokens/min>")


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
                samplerate=28000,    # ~1.4x speed
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
    token_tracker = TokenUsageTracker()

    async def handle_send(e=None):
        user_text = input_field.value.strip()
        if not user_text:
            return
        input_field.value = ""
        page.update()
        await send_message(user_text)

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
        on_submit=handle_send
    )
    send_button = ft.ElevatedButton("Send", on_click=handle_send)
    status_button = ft.IconButton(content=ft.Text("🎤"), tooltip="Listening", disabled=False)
    screenshot_button = ft.IconButton(content=ft.Text("📸"), tooltip="Toggle Screenshots")

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

        chat.controls.append(
            ft.Container(
                content=ft.Text(f"You: {user_text}", selectable=True, color="#BBBBBB"),
                padding=8,
                alignment=ft.alignment.center_left,
                expand=True
            )
        )
        page.update()

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

        history.append({
            "role": "user",
            "content": [{"type": "text", "text": user_text}] + attachments
        })

        system_prompt = history[0]
        user_assistant_messages = history[1:]

        full_context = [system_prompt] + [
            {k: v for k, v in m.items() if k in {"role", "content"}}
            for m in user_assistant_messages
        ]

        needed_tokens = sum(estimate_message_tokens(m) for m in full_context)
        await token_tracker.wait_for_tokens(needed_tokens)
        token_tracker.add_tokens(needed_tokens)
        print(f"[tokens] {token_tracker.tokens_used_last_minute()} tokens used in last 60s")

        try:
            stream = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=full_context,
                stream=True,
            )
        except Exception as e:
            chat.controls.append(
                ft.Container(
                    content=ft.Text(
                        f"Error: {e}",
                        color=ft.colors.RED
                    ),
                    padding=10,
                    alignment=ft.alignment.center_left
                )
            )
            page.update()
            return

        ai_message = ft.Text("", selectable=True)
        ai_container = ft.Container(content=ai_message, padding=3, alignment=ft.alignment.center_left)
        chat.controls.append(ai_container)
        page.update()

        full_reply = ""
        async for delta in stream:
            part = delta.choices[0].delta.content or ""
            full_reply += part
            ai_message.value = full_reply
            page.update()

        full_reply = full_reply.strip()
        final_text = full_reply

        if full_reply:
            if len(full_reply) <= 100:
                await speaker.speak(full_reply, status_button, page)
            else:
                await token_tracker.wait_for_tokens(needed_tokens=500)

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
                        final_text = summary_text
                    else:
                        print("[summary] No summary generated")
                except Exception as e:
                    print(f"[summary error] {e}")

            history.append({
                "role": "assistant",
                "content": final_text
            })

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
