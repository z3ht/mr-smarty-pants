import os
import pickle

import flet as ft
from flet import colors
from typing import Optional

from msp.settings import PROJECT_DIR


class ChatBubble(ft.Container):
    PALETTE = {
        "user": ("#CCCCCC", ft.alignment.center_left),
        "assistant": ("#FFFFFF", ft.alignment.center_left),
        "system": ("#FF9800", ft.alignment.center_left),
    }

    def __init__(self, text: str, sender: str = "user", temporary: bool = False, identifier: str = None):
        color, align = self.PALETTE.get(sender, ("#FF0000", ft.alignment.center_left))
        super().__init__(
            key=identifier,
            content=ft.Text(value=text, selectable=True, color=color, italic=temporary),
            padding=2,
            alignment=align,
            data={"temporary": temporary, "sender": sender}
        )

    @property
    def text(self) -> ft.Text:
        return self.content  # type: ignore

    def update_text(self, new_val: str, italic: bool = False, color: Optional[str] = None):
        self.text.value = new_val
        self.text.italic = italic
        if color:
            self.text.color = color


class ChatView(ft.Column):
    _HIGHLIGHT_CLR = colors.with_opacity(0.4, colors.AMBER_200)

    def __init__(self, *, expand: bool = True):
        super().__init__(spacing=0, expand=expand)

        self._col = ft.Column(
            spacing=10,
            scroll=ft.ScrollMode.AUTO,
            expand=True
        )
        self._scrollable = ft.Container(
            content=self._col,
            expand=True,
            clip_behavior=ft.ClipBehavior.ANTI_ALIAS
        )
        self.controls.append(self._scrollable)

    def _next_key(self, sender: str) -> str:
        return f"{sender}-{len(self._col.controls)}"

    def add_user(self, text: str) -> ChatBubble:
        b = ChatBubble(text, sender="user", identifier=self._next_key("user"))
        self._col.controls.append(b)
        if self.page:
            self.page.update()
        return b

    def start_ai(self) -> ChatBubble:
        b = ChatBubble("Thinking…", sender="assistant", temporary=True, identifier=self._next_key("ai"))
        self._col.controls.append(b)
        if self.page:
            self.page.update()
        return b

    def finish_ai(self, bubble: ChatBubble):
        bubble.data["temporary"] = False
        bubble.text.italic = False
        bubble.text.color = ChatBubble.PALETTE["assistant"][0]
        if self.page:
            self.page.update()

    def clear(self):
        self._col.controls.clear()
        if self.page:
            self.page.update()

    def is_empty(self) -> bool:
        return not self._col.controls

    def scroll_to_bubble(self, idx: int):
        if 0 <= idx < len(self._col.controls):
            ctrl = self._col.controls[idx]
            if ctrl.key:
                self.page.scroll_to(key=ctrl.key)
                self.page.update()

    def highlight(self, idx: int, *, colour: Optional[str] = None, auto_scroll: bool = False):
        colour = colour or self._HIGHLIGHT_CLR

        for ctrl in self._col.controls:
            if isinstance(ctrl, ChatBubble):
                ctrl.bgcolor = None

        if 0 <= idx < len(self._col.controls):
            bubble = self._col.controls[idx]
            if isinstance(bubble, ChatBubble):
                bubble.bgcolor = colour
                if auto_scroll:
                    self.scroll_to_bubble(idx)

        if self.page:
            self.page.update()

    def search(self, needle: str, *, case_sensitive: bool = False) -> list[int]:
        needle = needle if case_sensitive else needle.lower()
        matches = []
        for i, ctrl in enumerate(self._col.controls):
            if isinstance(ctrl, ChatBubble):
                text = ctrl.text.value
                text = text if case_sensitive else text.lower()
                if needle in text:
                    matches.append(i)
        return matches

    def save_view(self, stem_name: str):
        file_path = os.path.join(PROJECT_DIR, "history", "view", f"{stem_name}.pkl")
        export = []
        for ctrl in self._col.controls:
            if isinstance(ctrl, ft.Container) and isinstance(ctrl.content, ft.Text):
                export.append({
                    "sender": ctrl.data.get("sender") if isinstance(ctrl.data, dict) else None,
                    "text": ctrl.content.value,
                })

        with open(file_path, "wb") as f:
            pickle.dump(export, f)

    def load_view(self, stem_name: str) -> bool:
        file_path = os.path.join(PROJECT_DIR, "history", "view", f"{stem_name}.pkl")
        if not os.path.exists(file_path):
            print(f"[ChatView.load_view] File not found: {file_path}")
            return False

        try:
            with open(file_path, "rb") as f:
                export = pickle.load(f)
        except Exception as e:
            print(f"[ChatView.load_view] Error reading {file_path}: {e}")
            return False

        self.clear()
        for i, entry in enumerate(export):
            sender = entry.get("sender") or "assistant"
            text = entry.get("text", "")
            key = f"{sender}-{i}"
            bubble = ChatBubble(text, sender=sender, temporary=False, identifier=key)
            self._col.controls.append(bubble)

        if self.page:
            self.page.update()

        return True
