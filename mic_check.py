import asyncio
import io
import os
import sounddevice as sd
from dotenv import load_dotenv
from openai import AsyncOpenAI
import soundfile as sf


load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY; set it in .env file")

client = AsyncOpenAI(api_key=API_KEY)

async def record_and_transcribe():
    samplerate = 16000
    duration_s = 3

    print("Recording...")
    recording = sd.rec(
        int(duration_s * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    print("Recording complete!")

    buf = io.BytesIO()
    sf.write(buf, recording, samplerate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    buf.name = "speech.wav"

    print("Sending to Whisper...")
    text = await client.audio.transcriptions.create(
        model="whisper-1",
        file=buf,
        response_format="text",
        language="en"
    )

    print(f"Transcription: {text.strip()}")


if __name__ == "__main__":
    asyncio.run(record_and_transcribe())
