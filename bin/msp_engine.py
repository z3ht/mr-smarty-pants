#!/usr/bin/env python3
from __future__ import annotations
import base64
import io
import os
from datetime import datetime
from typing import List

import numpy as np
import sounddevice as sd
from mss import mss
from PIL import Image
from openai import AsyncOpenAI
from dotenv import load_dotenv
import soundfile as sf
from textual.scroll_view import ScrollView

from region_selector import select_region
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Input, Static
import asyncio
from typing import AsyncIterator


print(sd.query_devices())
print("Selected sound input:", sd.default.device[0])

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


os.makedirs("screenshots", exist_ok=True)
CROP_BOX = select_region()

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
        data = buf.getvalue()

        # Optionally still save it for debug purposes
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = f"screenshots/{ts}.jpg"
        with open(path, "wb") as f:
            f.write(data)

        return data


async def mic_stream() -> AsyncIterator[str]:
    samplerate = 16000
    frame_duration_ms = 100  # 0.1 sec per frame
    frame_samples = int(samplerate * frame_duration_ms / 1000)

    recording_buffer = []
    speaking = False
    silence_frames = 0
    silence_threshold = 500  # tune this: lower = more sensitive
    max_silence_frames = AUDIO_CHUNK_S * 10     # <- number of frames per second

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
                    print("[mic] Detected start of speech")
                    speaking = True
                recording_buffer.append(frame)
                silence_frames = 0
            elif speaking:
                recording_buffer.append(frame)
                silence_frames += 1
                if silence_frames > max_silence_frames:
                    print("[mic] Detected end of speech")

                    # Save full recording
                    audio_data = np.concatenate(recording_buffer)
                    buf = io.BytesIO()
                    sf.write(buf, audio_data, samplerate, format="WAV", subtype="PCM_16")
                    buf.seek(0)
                    buf.name = "speech.wav"

                    print("Sending to Whisper...")
                    text = await client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=buf,
                        response_format="text",
                        language="en"
                    )

                    text = text.strip()
                    print(f"[mic] {text}")
                    yield text

                    # Reset for next utterance
                    speaking = False
                    recording_buffer = []
                    silence_frames = 0
            else:
                # not speaking, just keep waiting
                await asyncio.sleep(frame_duration_ms / 1000)


async def speak(text: str):
    resp = await client.audio.speech.create(
        model=TTS_MODEL,
        voice=VOICE_NAME,
        input=text,
        response_format="pcm"
    )
    audio_bytes = resp.content
    pcm = np.frombuffer(audio_bytes, dtype=np.int16)

    sd.play(pcm, samplerate=24000)
    sd.wait()


class TextualKeyboardApp(App):
    CSS = """
    Screen {
        align: center middle;
    }
    """

    def __init__(self):
        super().__init__()
        self.queue = asyncio.Queue()

    def compose(self) -> ComposeResult:
        with Vertical():
            self.chat_history = ScrollView()
            self.chat_history.update(Static("Welcome to AI Assistant!\n"))
            yield self.chat_history

            self.input = Input(placeholder="Type here and press Enter...")
            yield self.input

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if text:
            self.input.value = ""
            await self.queue.put(text)
            await self.chat_history.mount(Static(f"[You] {text}"))
            await self.chat_history.scroll_end(animate=False)

    async def keyboard_stream(self) -> AsyncIterator[str]:
        while True:
            text = await self.queue.get()
            yield text
textual_keyboard_app = TextualKeyboardApp()


async def combined_input_stream() -> AsyncIterator[str]:
    mic = mic_stream()
    keyboard = textual_keyboard_app.keyboard_stream()

    mic_task = asyncio.create_task(mic.__anext__())
    keyboard_task = asyncio.create_task(keyboard.__anext__())

    mic_alive = True
    keyboard_alive = True

    while mic_alive or keyboard_alive:
        tasks = []
        if mic_alive:
            tasks.append(mic_task)
        if keyboard_alive:
            tasks.append(keyboard_task)

        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            try:
                text = task.result()
            except StopAsyncIteration:
                if task is mic_task:
                    print("[combined_input_stream] mic_stream ended.")
                    mic_alive = False
                elif task is keyboard_task:
                    print("[combined_input_stream] keyboard_stream ended.")
                    keyboard_alive = False
                continue

            yield text

            if task is mic_task and mic_alive:
                mic_task = asyncio.create_task(mic.__anext__())
            elif task is keyboard_task and keyboard_alive:
                keyboard_task = asyncio.create_task(keyboard.__anext__())

    print("[combined_input_stream] All sources exhausted. Shutting down.")


async def chat_session():
    inputs = combined_input_stream()
    history: List[dict] = [
        {"role": "system", "content": (
            "You are an AI pair‑programmer. Your programmer is working hard so please don't distract him with "
            "unimportant details. If he asks you a question, feel free to respond as your chat gpt agent would."
            "Please use knowledge of your history to know if you're repeating redundant information"
        )}
    ]

    async for speech in inputs:
        speech = speech.strip()
        if not speech:
            continue  # Ignore empty messages

        user_updates = [{"type": "text", "text": speech}]

        # NEW: take screenshot *now*, right before sending
        shot = await take_screenshot()
        b64 = base64.b64encode(shot).decode()
        user_updates.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

        history.append({"role": "user", "content": user_updates})

        stream = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=history,
            stream=True,
        )

        sentence_buf = ""
        async for delta in stream:
            part = delta.choices[0].delta.content or ""
            print(part, end="", flush=True)
            sentence_buf += part
            if sentence_buf.endswith((".", "?", "!")):
                if sentence_buf.lstrip().startswith("TALK:"):
                    await speak(sentence_buf.lstrip()[5:].strip())
                sentence_buf = ""


def main():
    try:
        asyncio.run(chat_session())
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
