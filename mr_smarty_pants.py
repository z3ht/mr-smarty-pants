#!/usr/bin/env python3
from __future__ import annotations
import asyncio
import base64
import io
import os
from typing import AsyncIterator, List, Tuple

import numpy as np
import sounddevice as sd
from mss import mss
from PIL import Image
import openai
from dotenv import load_dotenv

from msp.region_selector import select_region


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY; set it in .env file")

FRAME_INTERVAL_S = int(os.getenv("FRAME_INTERVAL_S", "5"))
AUDIO_CHUNK_S = int(os.getenv("AUDIO_CHUNK_S", "3"))
VOICE_NAME = os.getenv("VOICE_NAME", "alloy")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
client = openai.OpenAI(api_key=API_KEY)

CROP_BOX: Tuple[int, int, int, int] = select_region()
print(f"Using crop region: {CROP_BOX}")


async def screenshot_stream() -> AsyncIterator[bytes]:
    with mss() as sct:
        while True:
            raw = sct.grab({
                "left": CROP_BOX[0],
                "top": CROP_BOX[1],
                "width": CROP_BOX[2],
                "height": CROP_BOX[3]
            })
            img = Image.frombytes("RGB", raw.size, raw.rgb)
            img.thumbnail((1280, 800))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            yield buf.getvalue()
            await asyncio.sleep(FRAME_INTERVAL_S)


async def mic_stream() -> AsyncIterator[str]:
    samplerate = 16000
    frames_per_chunk = int(samplerate * AUDIO_CHUNK_S)
    q: asyncio.Queue[np.ndarray] = asyncio.Queue()

    def _callback(indata, frames, time, status):
        q.put_nowait(indata.copy())

    with sd.InputStream(channels=1, samplerate=samplerate, dtype="int16", callback=_callback):
        buf = np.empty((0,), dtype=np.int16)
        while True:
            chunk = await q.get()
            buf = np.concatenate((buf, chunk.reshape(-1)))
            if buf.shape[0] >= frames_per_chunk:
                wav_bytes = buf.tobytes()
                buf = np.empty((0,), dtype=np.int16)
                # Send to Whisper
                audio_file = ("speech.wav", wav_bytes, "audio/wav")
                transcription = await asyncio.to_thread(
                    client.audio.transcriptions.create,
                    model=WHISPER_MODEL,
                    file=audio_file,
                    format="text",
                )
                yield transcription.text.strip()


async def speak(text: str):
    def _synth_play():
        audio_bytes = client.audio.speech.create(
            model=TTS_MODEL,
            voice=VOICE_NAME,
            input=text,
            response_format="pcm",
        )
        pcm_data = np.frombuffer(audio_bytes, dtype=np.int16)
        sd.play(pcm_data, samplerate=48000)
        sd.wait()
    await asyncio.to_thread(_synth_play)

# ────────────────────────── CHAT DRIVER ────────────────────────
async def chat_session():
    screenshots = screenshot_stream()
    mic = mic_stream()
    history: List[dict] = [
        {"role": "system", "content": (
            "You are an AI pair‑programmer. When you detect a compiler error, runtime exception, "
            "or failed test in the screenshot, respond with TALK: followed by a brief fix."
        )}
    ]

    async for shot in screenshots:
        try:
            user_updates = []
            try:
                speech = await asyncio.wait_for(mic.__anext__(), timeout=0.0)
                if speech:
                    user_updates.append({"type": "text", "text": speech})
            except asyncio.TimeoutError:
                pass

            b64 = base64.b64encode(shot).decode()
            user_updates.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
            })

            history.append({"role": "user", "content": user_updates})
            stream = client.chat.completions.create(
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
        except Exception as e:
            print("[error]", e)
            await asyncio.sleep(2)


def main():
    try:
        asyncio.run(chat_session())
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
