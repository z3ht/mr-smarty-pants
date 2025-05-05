from msp.client_selection import get_client
from msp.settings import PROJECT_DIR, VISION_MODEL

import base64
import os


client = get_client(VISION_MODEL)

PROMPT = """This is a screenshot of an IDE. Extract all visible code blocks.

For each code block, return:
 - "text": The code content, excluding IDE overlays like type hints or annotations
 - "is_active_block": True if this appears to be the code the user is actively working in, otherwise False
 - "is_obscured": True if the code is incomplete or covered, False otherwise
"""

def extract_code_text(img_bytes: bytes) -> str:
    encoded = base64.b64encode(img_bytes).decode()
    base64_data_url = f"data:image/png;base64,{encoded}"

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": base64_data_url}},
                    {"type": "text", "text": PROMPT}
                ]
            }
        ]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    screenshot_path = os.path.join(PROJECT_DIR, "screenshots", "screenshot_20250503_201304.png")
    with open(screenshot_path, "rb") as f:
        img_bytes = f.read()
    print(extract_code_text(img_bytes))
