import tiktoken
from msp.settings import CHAT_MODEL


def count_tokens(text: str) -> int:
    encoding = tiktoken.encoding_for_model(CHAT_MODEL)
    tokens = encoding.encode(text)
    return len(tokens)


def estimate_message_tokens(message: dict) -> int:
    content = message.get("content")
    total_tokens = 0

    if isinstance(content, list):
        for part in content:
            if part.get("type") == "text":
                total_tokens += count_tokens(part.get("text", ""))
            elif part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if url.startswith("data:image/"):
                    base64_data = url.split(",", 1)[-1]
                    byte_length = (len(base64_data) * 3) // 4
                    estimated_tokens = max(1, byte_length // 4)
                    total_tokens += estimated_tokens
    elif isinstance(content, str):
        total_tokens += count_tokens(content)

    total_tokens += 3 # small constant overhead per message

    return total_tokens