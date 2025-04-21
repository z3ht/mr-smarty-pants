### mr-smarty-pants

minimal multimodal cli assistant

Features:
- native ui (flet/flutter)
- Async event‑loop (asyncio)
- Periodic screenshot capture (mss + Pillow) → JPEG bytes
- Microphone capture → Whisper STT (OpenAI)
- gpt4o-mini chat with text + image payloads
- Streaming text‑to‑speech playback (OpenAI TTS + sounddevice)

Cost: $0.50-$2.00 per hour
- significantly correlated with # of files being worked on

Requires:
- python >= 3.11
- OS level permissions for screen capture, accessibility, sound, and audio
- All python packages defined in requirements.txt (e.g. `pip install -r requirements.txt`)
- All keys from `example.env` specified in `.env`

---

Todo (prioritized):
- Add support for pasting selection as context rather than screenshots. Add hotkey context upload
    - Support visualizing current context. Screenshot as part of text history.
    - Text blocks as part of history
- Investigate more optimal difference functions for screenshotting text. Also better compression algorithms. Also cheaper api endpoints.
- Add dropdowns for settings that are defined in code. Two kinds of settings: intelligence and objective
    - Allow for specifying system prompt
    - Support a screenshot upload equation considering age vs difference. Support number of screenshots to upload
    - Allow for updating tts prompt (or disabling). Also allow for specifying input
    - Easily allow for changing model. Maybe even specify model + sampling rates as a prebuilt in a dropdown. Like an ide run config
    - Add cost estimate per minute
    - Allow for uploading just screenshots. Probably not worth including much discussion context?
- Mark messages as important for including in conversation history. By default use configurable sliding window. maybe decay visual. Possibly summarize old context
    - Ask ai if question is sill relevant to conversation
    - Do not include screenshots from previous messages
    - Ask the ai what level of context is useful for answering the question

Ideas:
- day-in-review feedback session
- bug repository
- code base annotations
- flet conversation tabbing, markdown support
