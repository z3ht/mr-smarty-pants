import base64
import os
from mistralai import Mistral
from dotenv import load_dotenv


load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
client = Mistral(api_key=api_key)

def extract_code_text(img_bytes: bytes) -> str:
    try:
        encoded = base64.b64encode(img_bytes).decode()
        base64_data_url = f"data:image/png;base64,{encoded}"

        # I tried a bunch of different models. Pixtral has the best mix of accuracy and price.
        # There is definitely a lot more optimizing that I can still do.
        # There are some cool hugging face models that I can't run locally, otherwise open source
        # OCR isnt there yet.
        chat_response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": base64_data_url
                        },
                        {
                            "type": "text",
                            "text": "Return only the code from this image"
                        }
                    ]
                }
            ]
        )

        return chat_response.choices[0].message.content

    except Exception as e:
        print(f"[error] extract_code_text: {e}")
        return ""