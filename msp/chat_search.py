import typing
import flet as ft
from msp.chat_view import ChatView


class SearchBar(ft.Row):
    def __init__(self, chat_view: ChatView, page: ft.Page):
        super().__init__(
            spacing=4,
            visible=False,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )

        self.chat = chat_view
        self.page = page

        self._matches: list[int] = []
        self._idx: int = -1

        self.search_field = ft.TextField(
            label="Search chat…",
            expand=True,
            autofocus=True,
            on_submit=lambda _: self._run_search(),
            on_change=lambda _: self._run_search(live=True),
        )
        self.result_label = ft.Text(value="", width=70, text_align=ft.TextAlign.CENTER)

        self.prev_btn = ft.IconButton(content=ft.Text("⬆"), tooltip="Previous match", on_click=self._prev)
        self.next_btn = ft.IconButton(content=ft.Text("⬇"), tooltip="Next match", on_click=self._next)
        self.close_btn = ft.IconButton(content=ft.Text("❌"), tooltip="Close search", on_click=self.close)

        self.controls.extend([
            self.search_field,
            self.result_label,
            self.prev_btn,
            self.next_btn,
            self.close_btn
        ])

        self._prev_key_handler: typing.Optional[typing.Callable[[ft.KeyboardEvent], None]] = None
        self._update_ui()

    def open(self):
        if not self.visible:
            self.visible = True
            self._grab_keys()
        self.search_field.focus()
        self.page.update()

    def close(self, *_):
        self.chat.highlight(-1)
        self._matches.clear()
        self._idx = -1
        self.visible = False
        self.search_field.value = ""
        self.result_label.value = ""
        self._release_keys()
        self.page.update()

    def _run_search(self, *, live: bool = False):
        query = self.search_field.value.strip()
        if not query:
            self._matches.clear()
            self._idx = -1
            self.chat.highlight(-1)
            self._update_ui()
            return

        self._matches = self.chat.search(query)
        if not self._matches:
            self._idx = -1
            self.chat.highlight(-1)
            self._update_ui(no_matches=True)
            return

        self._idx = 0 if live or self._idx < 0 else self._idx
        self._jump()

    def _jump(self):
        if not self._matches:
            return
        self._idx %= len(self._matches)
        self.chat.highlight(self._matches[self._idx], auto_scroll=True)
        self._update_ui()

    def _next(self, *_):
        if self._matches:
            self._idx = (self._idx + 1) % len(self._matches)
            self._jump()

    def _prev(self, *_):
        if self._matches:
            self._idx = (self._idx - 1) % len(self._matches)
            self._jump()

    def _update_ui(self, *, no_matches: bool = False):
        if no_matches:
            self.result_label.value = "0 results"
            self.prev_btn.disabled = self.next_btn.disabled = True
        elif self._matches:
            self.result_label.value = f"{self._idx + 1} of {len(self._matches)}"
            self.prev_btn.disabled = self.next_btn.disabled = len(self._matches) == 1
        else:
            self.result_label.value = ""
            self.prev_btn.disabled = self.next_btn.disabled = True

        self.page.update()

    def _grab_keys(self):
        if self._prev_key_handler is None:
            self._prev_key_handler = self.page.on_keyboard_event
        self.page.on_keyboard_event = self._key_handler

    def _release_keys(self):
        if self._prev_key_handler:
            self.page.on_keyboard_event = self._prev_key_handler
            self._prev_key_handler = None

    def _highlight_query_in_field(self):
        query = self.search_field.value.strip()
        if not query:
            return

        text = self.search_field.value
        haystack = text.lower()
        needle = query.lower()

        match_idx = haystack.find(needle)
        if match_idx >= 0:
            self.search_field.selection_base = match_idx
            self.search_field.cursor_position = match_idx + len(needle)

    def _key_handler(self, e: ft.KeyboardEvent):
        k = e.key
        ctrl = e.ctrl or e.meta

        if k == "Escape":
            self.close()
            return

        if ctrl and k.lower() == "f":
            self.search_field.focus()
            self._highlight_query_in_field()
            return

        if k == "Enter" and not (e.ctrl or e.meta or e.alt):
            (self._prev if e.shift else self._next)()
            return

        if self._prev_key_handler:
            self._prev_key_handler(e)
