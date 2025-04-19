"""
Opens a temporary full‑screen window that shows the current desktop snapshot.
Click‑and‑drag to draw a rectangle; on mouse release the window closes and
returns `(x, y, width, height)` coordinates.

Example
-------
>>> from msp.region_selector import select_region
>>> bbox = select_region()  # user selects area
>>> print(bbox)             # (left, top, width, height)

"""

from __future__ import annotations

import asyncio
import sys
from typing import Tuple, Optional

import mss
from PIL import Image, ImageTk
import tkinter as tk

__all__ = ["select_region"]


class _Selector(tk.Tk):
    def __init__(self, screenshot: Image.Image):
        super().__init__()
        self.withdraw()
        self.attributes("-fullscreen", True)
        self.attributes("-topmost", True)
        self.resizable(False, False)
        self.overrideredirect(True)

        self._start: Optional[Tuple[int, int]] = None
        self._rect_id: Optional[int] = None
        self._bbox: Optional[Tuple[int, int, int, int]] = None

        self.canvas = tk.Canvas(self, cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.bg_img = ImageTk.PhotoImage(screenshot)
        self.canvas.create_image(0, 0, anchor="nw", image=self.bg_img)

        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.deiconify()

    def get_bbox(self) -> Tuple[int, int, int, int]:
        self.mainloop()
        if self._bbox is None:
            raise RuntimeError("Selection aborted")
        return self._bbox

    def _on_press(self, event):
        self._start = (event.x, event.y)
        if self._rect_id:
            self.canvas.delete(self._rect_id)
            self._rect_id = None

    def _on_drag(self, event):
        if not self._start:
            return
        x0, y0 = self._start
        x1, y1 = event.x, event.y
        if self._rect_id:
            self.canvas.coords(self._rect_id, x0, y0, x1, y1)
        else:
            self._rect_id = self.canvas.create_rectangle(
                x0, y0, x1, y1,
                outline="red", width=2, dash=(4, 2)
            )

    def _on_release(self, event):
        if not self._start:
            return
        x0, y0 = self._start
        x1, y1 = event.x, event.y
        left, right = sorted((x0, x1))
        top, bottom = sorted((y0, y1))
        self._bbox = (left, top, right - left, bottom - top)
        self.destroy()


def select_region() -> Tuple[int,int,int,int]:
    """Captures current desktop, user draws crop, selected bbox returned"""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
    selector = _Selector(img)
    return selector.get_bbox()


# manual test
if __name__ == "__main__":
    try:
        bbox = select_region()
        print("Selected region:", bbox)
    except Exception as e:
        print("Selection cancelled:", e, file=sys.stderr)
        sys.exit(1)
