import flet as ft
from typing import Optional


class ChatBubble(ft.Container):
    _PALETTE = {
        "user":    ("#888888", ft.alignment.center_left),
        "assistant": ("#FFFFFF", ft.alignment.center_left),
        "system":  ("#FF9800", ft.alignment.center_left),
    }

    def __init__(
        self,
        text: str,
        sender: str = "user",
        temporary: bool = False,
    ):
        color, align = self._PALETTE.get(sender, ("#FF0000", ft.alignment.center_left))
        super().__init__(
            content=ft.Text(value=text, selectable=True, color=color, italic=temporary),
            padding=2,
            alignment=align,
            data={"temporary": temporary, "sender": sender},
            expand=True,
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
    def __init__(self, *, expand: bool = True, **lv_kwargs):
        super().__init__(spacing=0, expand=expand)
        self._lv = ft.ListView(
            expand=True, auto_scroll=True, spacing=10, **lv_kwargs
        )
        self.controls.append(self._lv)  # Column holds the ListView

    def add_user(self, text: str) -> ChatBubble:
        b = ChatBubble(text, sender="user")
        self._lv.controls.append(b)
        if self.page:
            self.page.update()
        return b

    def start_thinking(self) -> ChatBubble:
        b = ChatBubble("Thinking…", sender="assistant", temporary=True)
        self._lv.controls.append(b)
        if self.page:
            self.page.update()
        return b

    def finish_ai(self, bubble: ChatBubble):
        bubble.data["temporary"] = False
        bubble.text.italic = False
        bubble.text.color = ChatBubble._PALETTE["assistant"][0]
        if self.page:
            self.page.update()

    def cleanup(self, keep_last: int = 10):
        self._lv.controls = [
            c
            for c in self._lv.controls[-keep_last:]
            if not (isinstance(c, ft.Container) and c.data.get("temporary"))
        ]
        if self.page:
            self.page.update()

    def clear(self):
        self._lv.controls.clear()
        if self.page:
            self.page.update()
