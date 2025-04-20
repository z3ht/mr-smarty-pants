### mr-smarty-pants

minimal multimodal cli assistant

Features:
- Async event‑loop (asyncio)
- Periodic screenshot capture (mss + Pillow) → JPEG bytes
- Microphone capture → Whisper STT (OpenAI)
- GPT‑4o chat with text + image payloads
- Streaming text‑to‑speech playback (OpenAI TTS + sounddevice)

Cost: $0.50 per hour

Requires:
- pyhton-tk >= 3.11; *not* vanilla python (required for screen captures)
- OS level permissions for screen capture
- All python packages defined in requirements.txt (e.g. `pip install -r requirements.txt`)
- All keys populated + copied from `example.env` to `.env`
