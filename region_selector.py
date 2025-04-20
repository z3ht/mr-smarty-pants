import mss
import numpy as np
import cv2
import time

drawing = False
start_point = None
end_point = None
selection_done = False

def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, end_point, selection_done

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        selection_done = True

def select_region() -> tuple[int, int, int, int]:
    global drawing, start_point, end_point, selection_done

    # Step 1: Screenshot the screen
    with mss.mss() as sct:
        monitor = sct.monitors[1]  # Primary monitor
        screenshot = np.array(sct.grab(monitor))
        screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

    # Step 2: Create window normally (not fullscreen)
    window_name = "Select Region (Press ESC to cancel)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, screenshot.shape[1], screenshot.shape[0])
    cv2.setMouseCallback(window_name, mouse_callback)

    img_display = screenshot.copy()

    while True:
        # Always start from clean screenshot
        display = img_display.copy()

        if start_point and end_point:
            cv2.rectangle(display, start_point, end_point, (0, 255, 0), 2)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == 27:  # ESC
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            raise RuntimeError("User cancelled region selection.")

        if selection_done:
            break

    cv2.destroyWindow(window_name)
    cv2.waitKey(1)
    time.sleep(0.1)  # Give GUI time to fully close

    x0, y0 = start_point
    x1, y1 = end_point
    left, right = sorted([x0, x1])
    top, bottom = sorted([y0, y1])
    w = right - left
    h = bottom - top

    if w == 0 or h == 0:
        raise RuntimeError("No region selected.")

    print(f"Selected region: x={left}, y={top}, w={w}, h={h}")
    return left, top, w, h
