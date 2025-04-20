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

# --- Global control ---
current_stream = None
stopping = False
speaking_now = False

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

# --- Mic capture ---

async def mic_stream() -> AsyncIterator[str]:
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def record_loop():
        samplerate = 16000
        frame_duration_ms = 100
        frame_samples = int(samplerate * frame_duration_ms / 1000)
        recording_buffer = []
        speaking = False
        silence_frames = 0
        silence_threshold = 500
        max_silence_frames = AUDIO_CHUNK_S * 10

        stream = sd.InputStream(
            samplerate=samplerate,
            channels=1,
            dtype="int16",
            blocksize=frame_samples,
        )

        with stream:
            while True:
                frame, _ = stream.read(frame_samples)
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
                        asyncio.run_coroutine_threadsafe(queue.put(audio_data), loop)
                        speaking = False
                        recording_buffer = []
                        silence_frames = 0

    threading.Thread(target=record_loop, daemon=True).start()

    while True:
        recording = await queue.get()
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

# --- TTS (chunked and stoppable version) ---

async def speak(text: str, stop_button: ft.ElevatedButton, page: ft.Page):
    global current_stream, stopping, speaking_now
    try:
        stopping = False
        speaking_now = True
        stop_button.disabled = False
        page.update()

        # Get AI speech audio
        resp = await client.audio.speech.create(
            model=TTS_MODEL,
            voice=VOICE_NAME,
            input=text,
            response_format="pcm"
        )
        audio_bytes = resp.content
        pcm = np.frombuffer(audio_bytes, dtype=np.int16)

        # Setup queue for streaming
        audio_queue = Queue()

        # Feed audio into the queue in small pieces
        chunk_size = 240  # very small chunks
        for i in range(0, len(pcm), chunk_size):
            audio_queue.put(pcm[i:i+chunk_size])
        audio_queue.put(None)  # sentinel to mark end

        def callback(outdata, frames, time, status):
            try:
                if stopping:
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

        current_stream = sd.OutputStream(
            samplerate=24000,
            channels=1,
            dtype='int16',
            blocksize=240,  # match chunk size
            callback=callback
        )
        current_stream.start()

        # Wait until playback finishes
        while current_stream.active:
            if stopping:
                current_stream.abort()
                break
            await asyncio.sleep(0.05)

        current_stream.close()
        current_stream = None

    except Exception as e:
        print(f"[error] TTS failed: {e}")
        current_stream = None
    finally:
        speaking_now = False
        stop_button.disabled = True
        page.update()

# --- Main Flet App ---

def main(page: ft.Page):
    page.title = "Smart Assistant"
    page.vertical_alignment = "start"

    chat = ft.ListView(expand=True, spacing=10, auto_scroll=True)

    input_field = ft.TextField(label="Type here...", expand=True)
    send_button = ft.ElevatedButton("Send")
    stop_button = ft.ElevatedButton("Stop Speaking", disabled=True)

    history = [
        {"role": "system", "content": (
            "You are an AI assistant. Speak clearly, stay concise, and format code examples inside ``` blocks."
        )}
    ]

    async def send_message(user_text: str):
        if not user_text.strip():
            return

        chat.controls.append(
            ft.Container(
                content=ft.Markdown(f"**You:** {user_text}"),
                padding=10,
                alignment=ft.alignment.center_left
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
                    content=ft.Markdown(f"[red]Error: {e}[/red]"),
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
            ai_message.value = f"{full_reply}"
            page.update()

        if full_reply.strip():
            await speak(full_reply.strip(), stop_button, page)
            history.append({"role": "assistant", "content": full_reply.strip()})

    async def handle_send(e=None):
        user_text = input_field.value
        input_field.value = ""
        page.update()
        await send_message(user_text)

    def stop_speaking(e=None):
        global current_stream, stopping, speaking_now
        if speaking_now:
            print("[stop] Stopping AI speech...")
            stopping = True
            if current_stream is not None:
                try:
                    current_stream.abort()
                    current_stream.close()
                except Exception as e:
                    print(f"[stop] Error stopping: {e}")
                finally:
                    current_stream = None
            speaking_now = False
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

ft.app(target=main)
