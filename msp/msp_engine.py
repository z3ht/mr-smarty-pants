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
from mss import mss
from openai import AsyncOpenAI
from skimage.metrics import structural_similarity as ssim

from msp.chat_search import SearchBar
from msp.history_utils import unique_history_basename, get_previous_conversation_names, save_last_closed_conversation, get_last_closed_conversation
from msp.settings import OPENAI_API_KEY, TOKEN_LIMIT_PER_M, WINDOW_NAME, AUDIO_CHUNK_S, WHISPER_MODEL, VOICE_NAME, \
    TTS_MODEL, ASSETS_DIR, CHAT_MODEL, SCREENSHOT_INTERVAL_S, SCREENSHOT_SIMILARITY_THRESHOLD_PCT, PROJECT_DIR
from msp.context_manager import ContextManager
from msp.token_cost_estimate import estimate_message_tokens
from msp.chat_view import ChatView, ChatBubble


client = AsyncOpenAI(api_key=OPENAI_API_KEY)
shutdown_event = threading.Event()


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
        now = self._now() + 1.0  # Add 1 second to account for send delay
        self.usage.append((now, tokens))
        self._prune_old_usage()

    def tokens_used_last_minute(self) -> int:
        self._prune_old_usage()
        return sum(tokens for _, tokens in self.usage)

    def tokens_available(self) -> int:
        return TOKEN_LIMIT_PER_M - self.tokens_used_last_minute()

    def seconds_until_tokens_available(self, needed_tokens: int, safety_margin_s: float = 0.10) -> float:
        self._prune_old_usage()

        available = self.tokens_available()
        if available >= needed_tokens or not self.usage:
            return 0.0

        now = self._now()
        usage_list = list(self.usage)
        idx = 0

        while idx < len(usage_list):
            oldest_timestamp, oldest_tokens = usage_list[idx]
            idx += 1
            simulated_now = oldest_timestamp + 60
            remaining_usage = usage_list[idx:]
            used_tokens = sum(tokens for _, tokens in remaining_usage)
            available_tokens = TOKEN_LIMIT_PER_M - used_tokens

            if available_tokens >= needed_tokens:
                wait_time = simulated_now - now + safety_margin_s
                return max(0.0, wait_time)

        last_timestamp, _ = usage_list[-1]
        wait_time = (last_timestamp + 60) - now + safety_margin_s
        return max(0.0, wait_time)

    def __repr__(self):
        used = self.tokens_used_last_minute()
        available = self.tokens_available()
        return f"<TokenUsageTracker used={used} available={available} limit={TOKEN_LIMIT_PER_M} tokens/min>"


token_tracker = TokenUsageTracker()


_last_selected = None
_last_bbox = None
async def take_screenshot() -> bytes:
    global _last_selected, _last_bbox

    def get_bbox(win):
        return {
            "left": win.left,
            "top": win.top,
            "width": win.width,
            "height": win.height,
            "title": win.title
        }

    if _last_selected is None:
        all_windows = pwc.getAllWindows()
        matches = [w for w in all_windows if WINDOW_NAME.lower() in w.title.lower()]
        if not matches:
            print(f"[screenshot] No matching window found for '{WINDOW_NAME}'")
            return b""
        _last_selected = matches[0]
        _last_bbox = get_bbox(_last_selected)
    else:
        current_bbox = get_bbox(_last_selected)
        if current_bbox != _last_bbox:
            print("[screenshot] Window bbox changed, refreshing...")
            all_windows = pwc.getAllWindows()
            matches = [w for w in all_windows if WINDOW_NAME.lower() in w.title.lower()]
            if not matches:
                print(f"[screenshot] No matching window found for '{WINDOW_NAME}'")
                return b""
            _last_selected = matches[0]
            _last_bbox = get_bbox(_last_selected)

    if WINDOW_NAME.lower() not in _last_bbox["title"].lower():
        print("[screenshot] Lost bbox, refreshing on next screenshot...")
        _last_selected = None
        return b""

    with mss() as sct:
        raw = sct.grab(_last_bbox)
        img = Image.frombytes("RGB", raw.size, raw.rgb)

        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # save_dir = "screenshots"
        # os.makedirs(save_dir, exist_ok=True)
        # filename = os.path.join(save_dir, f"screenshot_{timestamp}.png")
        # img.save(filename, format="PNG")
        # print(f"[screenshot] Saved to {filename}")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class MicRecorder:
    def __init__(self, queue: asyncio.Queue[np.ndarray]):
        self.queue = queue
        self.loop = asyncio.get_running_loop()
        self.thread = None
        self.stream = None
        self.running = threading.Event()

    def start(self):
        if self.thread and self.thread.is_alive():
            print("[MicRecorder] Recorder already running.")
            return

        self.running.set()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print("[MicRecorder] Recorder started.")

    def stop(self):
        self.running.clear()
        if self.thread and self.thread.is_alive():
            self.thread.join()
        if self.stream:
            try:
                self.stream.close()
                print("[MicRecorder] Stream closed.")
            except Exception as e:
                print(f"[MicRecorder] Error closing stream: {e}")
        print("[MicRecorder] Recorder stopped.")

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
            while self.running.is_set():
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


async def mic_stream(page: ft.Page) -> AsyncIterator[str]:
    queue: asyncio.Queue[np.ndarray] = asyncio.Queue()
    recorder = None

    try:
        while not shutdown_event.is_set():
            if page.mic_enabled:
                if recorder is None:
                    recorder = MicRecorder(queue)
                    recorder.start()
            else:
                if recorder:
                    recorder.stop()
                    recorder = None
                await asyncio.sleep(0.01)
                continue

            try:
                recording = await asyncio.wait_for(queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            buf = io.BytesIO()
            sf.write(buf, recording, 16000, format="WAV", subtype="PCM_16")
            buf.seek(0)
            buf.name = "speech.wav"

            needed_tokens = 1000
            wait_s = token_tracker.seconds_until_tokens_available(needed_tokens=needed_tokens)
            if wait_s > 0.0:
                print(
                    f"[tokens] Waiting {wait_s:.2f}s before transcribing (used={token_tracker.tokens_used_last_minute()} / limit={TOKEN_LIMIT_PER_M})"
                )
                await asyncio.sleep(wait_s)

            print(f"[tokens] Using {needed_tokens} tokens for transcription (available={token_tracker.tokens_available()})")
            token_tracker.add_tokens(needed_tokens)

            text_obj = await client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=buf,
                response_format="text",
                language="en"
            )

            yield text_obj.strip()
    finally:
        if recorder:
            recorder.stop()


class Speaker:
    def __init__(self):
        self.current_stream = None
        self.speaking = False
        self.stopping = False
        self._speak_task = None

    async def speak(self, text: str, status_button: ft.ElevatedButton, page: ft.Page):
        if not page.speech_enabled:
            print("[speak] Speech disabled; skipping speaking")
            return
        await self.stop()
        self._speak_task = asyncio.create_task(
            self._internal_speak(text, status_button, page)
        )

    async def _internal_speak(self, text: str, status_button: ft.ElevatedButton, page: ft.Page):
        self.stopping = False
        self.speaking = True
        status_button.text = "Stop"
        status_button.tooltip = "Stop AI speech"
        status_button.bgcolor = ft.Colors.BLUE_GREY_900
        page.update()

        try:
            needed_tokens = max(50, len(text) // 4)

            wait_s = token_tracker.seconds_until_tokens_available(needed_tokens=needed_tokens)
            if wait_s > 0.0:
                print(
                    f"[tokens] Waiting {wait_s:.2f}s before speaking (used={token_tracker.tokens_used_last_minute()} / limit={TOKEN_LIMIT_PER_M})"
                )
                await asyncio.sleep(wait_s)

            print(f"[tokens] Using {needed_tokens} tokens for speech (available={token_tracker.tokens_available()})")
            token_tracker.add_tokens(needed_tokens)

            resp = await client.audio.speech.create(
                model=TTS_MODEL,
                voice=VOICE_NAME,
                input=text,
                response_format="pcm"
            )

            audio_bytes = resp.content
            pcm = np.frombuffer(audio_bytes, dtype=np.int16)

            audio_queue = Queue()
            chunk_size = 480

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
                samplerate=28000,  # ~1.4x speed
                channels=1,
                dtype="int16",
                blocksize=chunk_size,
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

            status_button.text = "Send"
            status_button.tooltip = "Send a message"
            status_button.bgcolor = None
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
    page.title = "Mr. Smarty Pants"
    page.vertical_alignment = "start"

    chat = ChatView(expand=True)

    input_field = ft.TextField(
        label="Type…",
        expand=True,
        multiline=True,
        min_lines=1,
        max_lines=15,
        shift_enter=True,
    )
    send_button = ft.ElevatedButton("Send")

    speech_button = ft.IconButton(content=ft.Text("🔇"), tooltip="Toggle speech (TTS)")
    page.speech_enabled = False

    mic_button = ft.IconButton(content=ft.Text("🎙️"), tooltip="Toggle microphone (STT)")
    page.mic_enabled = False

    screenshot_button = ft.IconButton(content=ft.Text("📷"), tooltip="Toggle screenshots")
    page.include_screenshots = False
    page.screenshot_buffer = []

    end_conversation_button = ft.IconButton(
        content=ft.Text("📝"),
        tooltip="Start a new conversation",
        padding=0
    )

    speaker = Speaker()
    page.send_task = None
    page.thinking_task = None

    system_prompt = {
        "role": "system",
        "content": (
            "You are Mr. Smarty Pants, an AI assistant. Speak clearly, stay concise, "
            "and format code examples inside triple backticks."
        ),
    }
    context_manager = ContextManager(system_prompt)

    search_bar = SearchBar(chat, page)

    current_stem: str | None = None
    pending_load_stem: str | None = None

    sidebar = ft.Column(spacing=4, width=190)
    sidebar.visible = False

    def toggle_sidebar():
        sidebar.visible = not sidebar.visible
        toggle_sidebar_btn.tooltip = "Hide history" if sidebar.visible else "Show history"
        page.update()

    toggle_sidebar_btn = ft.IconButton(
        content=ft.Image(
            src=os.path.join(ASSETS_DIR, "logo.png"),
            width=32,
            height=32,
            fit=ft.ImageFit.CONTAIN,
        ),
        tooltip="Show history",
        on_click=lambda e: toggle_sidebar(),
        style=ft.ButtonStyle(padding=ft.padding.all(0)),
    )

    end_conversation_button = ft.IconButton(
        content=ft.Text("📝"),
        tooltip="Start a new conversation",
        padding=0
    )

    def toggle_search():
        if search_bar.visible:
            search_bar.close()
        else:
            search_bar.open()

    search_toggle_btn = ft.IconButton(
        content=ft.Text("🔍"),
        tooltip="Search chat (Ctrl/⌘ + F)",
        on_click=lambda e: toggle_search(),
        style=ft.ButtonStyle(padding=ft.padding.all(0)),
    )

    def do_save_conversation(name: str, *, is_internal_stem_name: bool = False) -> str:
        nonlocal current_stem
        if not is_internal_stem_name:
            name = unique_history_basename(name)
        context_manager.save_conversation(name)
        chat.save_view(name)
        current_stem = name
        return name

    def do_load_conversation(stem_name: str):
        nonlocal current_stem
        context_manager.load_conversation(stem_name)
        chat.load_view(stem_name)
        current_stem = stem_name

    def refresh_sidebar():
        sidebar.controls.clear()

        top_buttons = ft.Row(
            [
                search_toggle_btn,
                end_conversation_button,
            ],
            alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
            spacing=8,
        )

        sidebar.controls.append(
            ft.Container(top_buttons, padding=0, margin=0, alignment=ft.alignment.center_right)
        )

        for stem in get_previous_conversation_names():
            sidebar.controls.append(
                ft.TextButton(
                    stem,
                    data=stem,
                    on_click=_load_chat,
                    style=ft.ButtonStyle(padding=ft.padding.symmetric(8, 4)),
                )
            )

        sidebar.update()

    def _load_chat(e: ft.ControlEvent):
        nonlocal pending_load_stem, current_stem

        new_stem = e.control.data
        if current_stem is None or current_stem.startswith("__"):
            pending_load_stem = new_stem
            end_conversation()
            return

        do_save_conversation(current_stem, is_internal_stem_name=True)
        do_load_conversation(new_stem)
        refresh_sidebar()

    async def send_message(user_text: str):
        if page.thinking_task:
            page.thinking_task.cancel()
            page.thinking_task = None

        chat.add_user(user_text)

        context_manager.add_screenshots(page.screenshot_buffer.copy())
        page.screenshot_buffer.clear()

        context_manager.add_user_message(user_text)
        full_context = context_manager.build_context()
        needed_tokens = sum(estimate_message_tokens(m) for m in full_context)

        thinking_bubble: ChatBubble = chat.start_ai()
        thinking_text = thinking_bubble.text

        async def show_thinking_status(waiting: bool):
            if waiting:
                while True:
                    wait_s = int(
                        token_tracker.seconds_until_tokens_available(needed_tokens)
                    )
                    if wait_s <= 0:
                        break
                    thinking_text.value = f"Waiting for tokens… ({wait_s}s)"
                    page.update()
                    await asyncio.sleep(0.5)
            await asyncio.sleep(0.5)
            thinking_text.value = "Generating response…"
            page.update()

        wait_s = token_tracker.seconds_until_tokens_available(needed_tokens)
        page.thinking_task = asyncio.create_task(show_thinking_status(wait_s > 0))

        if wait_s:
            await asyncio.sleep(wait_s)
        token_tracker.add_tokens(needed_tokens)

        try:
            stream = await client.chat.completions.create(
                model=CHAT_MODEL, messages=full_context, stream=True
            )
        finally:
            if page.thinking_task:
                page.thinking_task.cancel()
                page.thinking_task = None

        full_reply = ""
        first_chunk = True

        async for delta in stream:
            part = delta.choices[0].delta.content or ""
            if not part:
                continue
            full_reply += part

            if first_chunk:
                thinking_bubble.update_text(part.strip(), italic=False, color="#FFFFFF")
                thinking_bubble.data["temporary"] = False
                first_chunk = False
            else:
                thinking_bubble.update_text(full_reply)
            page.update()

        full_reply = full_reply.strip()
        chat.finish_ai(thinking_bubble)   # mark as final (no italics)

        if full_reply:
            if len(full_reply) <= 100:
                await speaker.speak(full_reply, send_button, page)
            else:
                await _speak_summary(full_reply)

            context_manager.add_assistant_message(full_reply)

    async def _speak_summary(text: str):
        wait_s = token_tracker.seconds_until_tokens_available(needed_tokens=500)
        if wait_s:
            await asyncio.sleep(wait_s)
        token_tracker.add_tokens(500)

        summary_resp = await client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the text into a single concise sentence for speaking aloud.",
                },
                {"role": "user", "content": text},
            ],
        )
        summary = summary_resp.choices[0].message.content.strip()
        if summary:
            await speaker.speak(summary, send_button, page)

    async def handle_send(_=None):
        user_text = input_field.value.strip()
        if not user_text:
            return
        input_field.value = ""
        page.update()

        if page.send_task and not page.send_task.done():
            page.send_task.cancel()
        page.send_task = asyncio.create_task(send_message(user_text))

    async def handle_send_button(_=None):
        if speaker.speaking:
            await speaker.stop()
        else:
            await handle_send()

    def toggle_speech(e=None):
        page.speech_enabled = not page.speech_enabled
        if page.speech_enabled:
            speech_button.content = ft.Text("🔈")
            speech_button.tooltip = "Speech Enabled (click to mute speech)"
            print("[toggle_speech] Speech enabled.")
        else:
            speech_button.content = ft.Text("🔇")
            speech_button.tooltip = "Speech Muted (click to unmute speech)"
            print("[toggle_speech] Speech disabled.")
        speech_button.update()

    def toggle_screenshots(e=None):
        page.include_screenshots = not page.include_screenshots
        screenshot_button.content.value = "📸" if page.include_screenshots else "📷"
        screenshot_button.tooltip = (
            "Taking screenshots" if page.include_screenshots else "Screenshots disabled"
        )
        screenshot_button.update()
        print(f"[toggle_screenshots] Screenshots {'enabled' if page.include_screenshots else 'disabled'}.")

    def toggle_mic(e=None):
        page.mic_enabled = not page.mic_enabled
        if page.mic_enabled:
            mic_button.content = ft.Text("🔴")
            mic_button.tooltip = "Listening (click to mute)"
            print("[toggle_mic] Mic enabled.")
        else:
            mic_button.content = ft.Text("🎙️")
            mic_button.tooltip = "Muted (click to unmute)"
            print("[toggle_mic] Mic disabled.")
        mic_button.update()
        page.update()

    conversation_name_field = ft.TextField(label="Name", autofocus=True, width=350)

    dlg_end_convo = ft.AlertDialog(
        modal=True,
        title=ft.Text("Save Current Chat?"),
        content=conversation_name_field,
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.overlay.append(dlg_end_convo)

    _prev_key_handler = page.on_keyboard_event

    def _close_cleanup_end_convo_dlg(e=None, *, clear_chat: bool = True):
        nonlocal _prev_key_handler, current_stem, pending_load_stem
        if _prev_key_handler:
            page.on_keyboard_event = _prev_key_handler
            _prev_key_handler = None

        if clear_chat:
            chat.clear()
            page.screenshot_buffer.clear()
            context_manager.clear()
            current_stem = "__on_close__"
        else:
            pending_load_stem = None

        refresh_sidebar()
        dlg_end_convo.open = False
        page.update()

        if clear_chat and pending_load_stem:
            do_load_conversation(pending_load_stem)
            pending_load_stem = None
            refresh_sidebar()

    def _save_conversation(e=None):
        name = conversation_name_field.value.strip()
        if not name:
            return
        do_save_conversation(name)
        _close_cleanup_end_convo_dlg()

    def _end_convo_dlg_key_handler(e: ft.KeyboardEvent):
        if not dlg_end_convo.open:
            return
        if e.key == "Escape":
            _close_cleanup_end_convo_dlg(clear_chat=False)
        elif e.key == "Enter" and not (e.alt or e.ctrl or e.shift):
            _save_conversation()

    dlg_end_convo.actions = [
        ft.TextButton("Cancel", on_click=lambda e: _close_cleanup_end_convo_dlg(e, clear_chat=False)),
        ft.TextButton("Wipe", on_click=_close_cleanup_end_convo_dlg),
        ft.ElevatedButton("Save", on_click=_save_conversation),
    ]

    def end_conversation(e=None):
        nonlocal current_stem, _prev_key_handler
        print("[end_conversation] Ending conversation…")

        if (current_stem is None or current_stem.startswith("__")) and not chat.is_empty():
            conversation_name_field.value = ""
            dlg_end_convo.open = True
            _prev_key_handler = page.on_keyboard_event
            page.on_keyboard_event = _end_convo_dlg_key_handler
            page.update()
            return

        do_save_conversation(current_stem, is_internal_stem_name=True)
        _close_cleanup_end_convo_dlg()

    def _page_key_handler(e: ft.KeyboardEvent):
        if e.key.lower() == "f" and (e.ctrl or e.meta):
            toggle_search()

    page.on_keyboard_event = _page_key_handler

    send_button.on_click = handle_send_button
    input_field.on_submit = handle_send
    mic_button.on_click = toggle_mic
    speech_button.on_click = toggle_speech
    end_conversation_button.on_click = end_conversation

    screenshot_button.on_click = toggle_screenshots

    main_col = ft.Column(
        [
            search_bar,
            chat,
            ft.Row(
                controls=[toggle_sidebar_btn, input_field, send_button, speech_button, mic_button, screenshot_button]
            )
        ],
        expand=True,
    )

    page.add(
        ft.Row(
            [sidebar, main_col],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        )
    )

    refresh_sidebar()
    do_load_conversation(get_last_closed_conversation())

    async def mic_listener():
        async for speech in mic_stream(page):
            await send_message(speech)

    async def screenshot_collector():
        last_image = None

        while not shutdown_event.is_set():
            await asyncio.sleep(SCREENSHOT_INTERVAL_S)
            if page.include_screenshots and WINDOW_NAME:
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
                            continue

                    last_image = img_arr
                    page.screenshot_buffer.append(current_bytes)
                except Exception as e:
                    print(f"[error] screenshot collector: {e}")

    page.mic_task = page.run_task(mic_listener)
    page.screenshot_task = page.run_task(screenshot_collector)

    async def on_close(e=None):
        print("[shutdown] Cleaning up...")

        stem_to_save = current_stem
        do_save_conversation(stem_to_save, is_internal_stem_name=True)
        save_last_closed_conversation(stem_to_save)

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
    page.on_disconnect = on_close


ft.app(target=main, assets_dir=os.path.join(PROJECT_DIR, "assets"))
