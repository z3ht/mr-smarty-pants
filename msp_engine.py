import asyncio
import base64
import io
import os
import threading
from queue import Queue, Empty
from typing import AsyncIterator

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from mss import mss
from PIL import Image
import flet as ft
from openai import AsyncOpenAI

from region_selector import select_region

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

client = AsyncOpenAI(api_key=API_KEY)

# --- Global shutdown event ---
shutdown_event = threading.Event()

# --- Select screen region ---
print("Select screen region...")
CROP_BOX = select_region()
print(f"Selected region: {CROP_BOX}")

# --- Screenshot ---
async def take_screenshot() -> bytes:
    with mss() as sct:
        raw = sct.grab({
            "left": CROP_BOX[0],
            "top": CROP_BOX[1],
            "width": CROP_BOX[2],
            "height": CROP_BOX[3],
        })
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img.thumbnail((1280, 800))
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
async def mic_stream() -> AsyncIterator[str]:
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    recorder = MicRecorder(queue)
    recorder.start()

    try:
        while True:
            if shutdown_event.is_set():
                break
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

# --- TTS Speaker ---
class Speaker:
    def __init__(self):
        self.current_stream = None
        self.stopping = False
        self.speaking = False

    async def speak(self, text: str, stop_button: ft.ElevatedButton, page: ft.Page):
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
                audio_queue.put(pcm[i:i+chunk_size])
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
                dtype='int16',
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

    def stop(self):
        if not self.speaking:
            return
        self.stopping = True

# --- Main App ---
def main(page: ft.Page):
    page.title = "Mr. Smarty Pants Assistant"
    page.vertical_alignment = "start"

    chat = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    input_field = ft.TextField(label="Type your message...", expand=True)
    send_button = ft.ElevatedButton("Send")
    stop_button = ft.ElevatedButton("Stop Speaking", disabled=True)

    speaker = Speaker()

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

        shot = await take_screenshot()
        b64 = base64.b64encode(shot).decode()

        user_content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]

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

    def stop_speaking(e=None):
        if speaker.speaking:
            print("[stop] Stopping AI speech...")
            speaker.stop()
            stop_button.disabled = True
            page.update()

    send_button.on_click = handle_send
    stop_button.on_click = stop_speaking
    input_field.on_submit = handle_send

    page.add(
        chat,
        ft.Row([input_field, send_button, stop_button])
    )

    async def mic_listener():
        async for speech in mic_stream():
            await send_message(speech)

    page.run_task(mic_listener)

    async def on_close(e):
        print("[shutdown] Cleaning up...")
        shutdown_event.set()
        speaker.stop()
        if speaker.current_stream:
            try:
                speaker.current_stream.close()
            except Exception as e:
                print(f"[error] closing stream on shutdown: {e}")
        print("[shutdown] Done.")

    page.on_close = on_close

# --- Run App ---
ft.app(target=main)
