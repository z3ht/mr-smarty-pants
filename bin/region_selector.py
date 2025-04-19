import mss
import numpy as np
import cv2

drawing = False
ix, iy = -1, -1
rect = None
selection_done = False

def mouse_callback(event, x, y, flags, param):
    global drawing, ix, iy, rect, selection_done

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)
        selection_done = True

def select_region() -> tuple[int, int, int, int]:
    """Capture screen, let user drag a rectangle, confirm automatically on release."""
    global rect, selection_done

    with mss.mss() as sct:
        monitor = sct.monitors[1]
        sct_img = sct.grab(monitor)
        device_w, device_h = sct_img.size
        img = np.array(sct_img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    clone = img.copy()

    window_name = "Select Region (drag mouse)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = clone.copy()
        if rect:
            x0, y0, x1, y1 = rect
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(1) & 0xFF

        if selection_done:
            break
        if key == 27:  # Escape key
            print("Selection cancelled.")
            cv2.destroyWindow(window_name)
            raise RuntimeError("User cancelled region selection.")

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)  # Flush events

    x0, y0, x1, y1 = rect
    left, right = sorted((x0, x1))
    top, bottom = sorted((y0, y1))
    w = right - left
    h = bottom - top

    print(f"Selected screen region: x={left}, y={top}, w={w}, h={h}")
    return left, top, w, h


if __name__ == "__main__":
    try:
        box = select_region()
    except RuntimeError as e:
        print(e)
