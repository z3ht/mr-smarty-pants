import argparse
import asyncio
import base64
import threading
from queue import Queue, Empty
from typing import AsyncIterator

import os
import io
from datetime import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from mss import mss
from PIL import Image
import pywinctl as pwc
import flet as ft
from openai import AsyncOpenAI

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
VOICE_NAME = os.getenv("VOICE_NAME", "alloy")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
SCREENSHOT_INTERVAL_S = float(os.getenv("SCREENSHOT_INTERVAL_S", "1"))
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "500"))

client = AsyncOpenAI(api_key=API_KEY)

# --- Global shutdown event ---
shutdown_event = threading.Event()


async def take_screenshot() -> bytes:
    if not SELECTED_WINDOW:
        return b""

    with mss() as sct:
        # Define the bounding box for the screenshot
        bbox = {
            "left": SELECTED_WINDOW.left,
            "top": SELECTED_WINDOW.top,
            "width": SELECTED_WINDOW.width,
            "height": SELECTED_WINDOW.height,
        }

        raw = sct.grab(bbox)

        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img.thumbnail((1280, 800))

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_dir = "screenshots"
        # os.makedirs(save_dir, exist_ok=True)
        # filename = os.path.join(save_dir, f"screenshot_{timestamp}.jpg")
        #
        # # Save the image as a JPEG file
        # img.save(filename, format="JPEG", quality=80)
        # print(f"[screenshot] Saved to {filename}")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


# --- Mic Recorder ---
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


# --- Mic Stream ---
async def mic_stream(mic_status_label: ft.Text) -> AsyncIterator[str]:
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    recorder = MicRecorder(queue)
    recorder.start()

    try:
        while not shutdown_event.is_set():
            try:
                mic_status_label.value = "🎤 Listening..."
                mic_status_label.update()
                recording = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            mic_status_label.value = "🛑 Idle"
            mic_status_label.update()

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
        mic_status_label.value = "🛑 Idle"
        mic_status_label.update()


# --- TTS Speaker ---
class Speaker:
    def __init__(self):
        self.current_stream = None
        self.speaking = False
        self.stopping = False
        self._speak_task = None

    async def speak(self, text: str, stop_button: ft.ElevatedButton, page: ft.Page):
        await self.stop()
        self._speak_task = asyncio.create_task(
            self._internal_speak(text, stop_button, page)
        )

    async def _internal_speak(self, text: str, stop_button: ft.ElevatedButton, page: ft.Page):
        self.stopping = False
        self.speaking = True
        stop_button.disabled = False
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
                samplerate=24000,
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
            stop_button.disabled = True
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

    input_field = ft.TextField(label="Type your message...", expand=True)
    send_button = ft.ElevatedButton("Send")
    stop_button = ft.ElevatedButton("Stop Speaking", disabled=True)
    mic_status_label = ft.Text("🛑 Idle", size=12, color="green")
    screenshot_switch = ft.Switch(label="Include Screenshots", value=True)

    speaker = Speaker()
    page.mic_task = None
    page.screenshot_task = None
    page.include_screenshots = screenshot_switch.value
    page.screenshot_buffer = []  # List of (timestamp, bytes)

    history = [
        {"role": "system", "content": (
            "You are Mr. Smarty Pants, an AI assistant. Speak clearly, stay concise, and format code examples inside triple backticks."
        )}
    ]

    async def send_message(user_text: str):
        if not user_text.strip():
            return

        chat.controls.append(
            ft.Container(
                content=ft.Text(f"You: {user_text}", selectable=True),
                padding=8,
                alignment=ft.alignment.center_left,
                expand=True
            )
        )
        page.update()

        # Gather all screenshots taken so far
        attachments = []
        for ts, shot_bytes in page.screenshot_buffer:
            b64 = base64.b64encode(shot_bytes).decode()
            attachments.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })
        page.screenshot_buffer.clear()

        user_content = [{"type": "text", "text": user_text}] + attachments

        history.append({"role": "user", "content": user_content})

        try:
            stream = await client.chat.completions.create(
                model=CHAT_MODEL,
                messages=history,
                stream=True,
            )
        except Exception as e:
            chat.controls.append(
                ft.Container(
                    content=ft.Text(f"[red]Error: {e}[/red]"),
                    padding=10,
                    alignment=ft.alignment.center_left
                )
            )
            page.update()
            return

        ai_message = ft.Text("", selectable=True)
        ai_container = ft.Container(
            content=ai_message,
            padding=3,
            alignment=ft.alignment.center_left
        )
        chat.controls.append(ai_container)
        page.update()

        full_reply = ""

        async for delta in stream:
            part = delta.choices[0].delta.content or ""
            full_reply += part
            ai_message.value = full_reply
            page.update()

        if full_reply.strip():
            await speaker.speak(full_reply.strip(), stop_button, page)
            history.append({"role": "assistant", "content": full_reply.strip()})

    async def handle_send(e=None):
        user_text = input_field.value
        input_field.value = ""
        page.update()
        await send_message(user_text)

    async def stop_speaking(e=None):
        if speaker.speaking:
            print("[stop] Stopping AI speech...")
            await speaker.stop()
            stop_button.disabled = True
            page.update()

    def toggle_screenshot_switch(e):
        page.include_screenshots = screenshot_switch.value
        print(f"[screenshot] Include screenshots: {page.include_screenshots}")

    screenshot_switch.on_change = toggle_screenshot_switch

    send_button.on_click = stop_speaking
    stop_button.on_click = stop_speaking
    input_field.on_submit = handle_send

    page.add(
        chat,
        ft.Row([input_field, send_button, stop_button, mic_status_label, screenshot_switch])
    )

    async def mic_listener():
        async for speech in mic_stream(mic_status_label):
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

                    img = Image.open(io.BytesIO(current_bytes)).convert("L")  # grayscale
                    img = img.resize((320, 200))  # small for speed
                    img_arr = np.asarray(img, dtype=np.float32)

                    if last_image is not None:
                        mse = np.mean((img_arr - last_image) ** 2)
                        if mse < MOTION_THRESHOLD:
                            print("[motion] No significant change, skipping screenshot.")
                            continue

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    page.screenshot_buffer.append((timestamp, current_bytes))
                    print(f"[motion] Saved screenshot at {timestamp}.")
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

# --- Run App ---
ft.app(target=main)
