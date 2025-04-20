### mr-smarty-pants

minimal multimodal cli assistant

Features:
- flet terminal ui
- Async event‑loop (asyncio)
- Periodic screenshot capture (mss + Pillow) → JPEG bytes
- Microphone capture → Whisper STT (OpenAI)
- GPT‑4o chat with text + image payloads
- Streaming text‑to‑speech playback (OpenAI TTS + sounddevice)

Todo:
- integrate with IDE text

Ideas:
- day-in-review feedback session
- bug repository
- code base annotations
- flet conversation tabbing, markdown support

Cost: $0.50-$2.00 per hour
- significantly correlated with # of files being worked on

Requires:
- python >= 3.11
- OS level permissions for screen capture, accessibility, sound, and audio
- All python packages defined in requirements.txt (e.g. `pip install -r requirements.txt`)
- All keys from `example.env` specified in `.env`
