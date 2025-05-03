import flet as ft

from msp.chat_view import ChatView


class SearchBar(ft.Row):
    def __init__(self, chat_view: ChatView, page: ft.Page):
        super().__init__(spacing=4, visible=False, vertical_alignment=ft.CrossAxisAlignment.CENTER)
        self.chat = chat_view
        self.page = page

        # --- internal state
        self._matches: list[int] = []
        self._idx: int = -1

        # --- controls
        self.search_field = ft.TextField(
            label="Search chat…",
            expand=True,
            autofocus=True,
            on_submit=lambda e: self._run_search(),
            on_change=lambda e: self._run_search(live=True),
        )
        self.prev_btn = ft.IconButton(content=ft.Text("⬆"), tooltip="Prev match", on_click=self._prev)
        self.next_btn = ft.IconButton(content=ft.Text("⬇"), tooltip="Next match", on_click=self._next)
        self.close_btn = ft.IconButton(content=ft.Text("❌"), tooltip="Close search", on_click=self.close)

        self.controls.extend([self.search_field, self.prev_btn, self.next_btn, self.close_btn])
        self._update_nav_buttons(disable=True)

    def open(self):
        self.visible = True
        self.search_field.focus()
        self.page.update()

    def close(self, *_):
        self.chat.highlight(-1)
        self.visible = False
        self.search_field.value = ""
        self._matches.clear()
        self._idx = -1
        self.page.update()

    def _update_nav_buttons(self, disable: bool = False):
        self.prev_btn.disabled = self.next_btn.disabled = disable
        self.page.update()

    def _run_search(self, live: bool = False):
        query = self.search_field.value.strip()
        if not query:
            self._update_nav_buttons(disable=True)
            self.chat.highlight(-1)
            return

        self._matches = self.chat.search(query)
        if not self._matches:
            self._update_nav_buttons(disable=True)
            self.chat.highlight(-1)
            return

        if live:
            self._idx = 0
        else:
            self._idx = self._idx if self._idx >= 0 else 0

        self._jump()

    def _jump(self):
        if not self._matches:
            return
        self._idx %= len(self._matches)
        self.chat.highlight(self._matches[self._idx], auto_scroll=True)
        self._update_nav_buttons(disable=False)

    def _next(self, *_):
        if self._matches:
            self._idx = (self._idx + 1) % len(self._matches)
            self._jump()

    def _prev(self, *_):
        if self._matches:
            self._idx = (self._idx - 1) % len(self._matches)
            self._jump()
