import os
from datetime import datetime, timedelta
from typing import Optional, List
import pickle

from msp.ocr_screenshot import extract_code_text
from msp.settings import MAX_HISTORY_MESSAGES, MAX_HISTORY_AGE_S, NUM_LATEST_SCREENSHOTS, PROJECT_DIR
from msp.token_cost_estimate import estimate_message_tokens


class ContextManager:
    def __init__(self, system_prompt: dict):
        self.system_prompt = system_prompt
        self.history: List[dict] = []
        self.all_screenshots: List[List[bytes]] = []

    def _now(self) -> datetime:
        return datetime.now()

    def clear(self):
        self.history.clear()
        self.all_screenshots.clear()

    def save_conversation(self, conversation_name: str):
        if not conversation_name:
            conversation_name = "anonymous"
        attempt = 0
        while True:
            full_name = f"{self._now().strftime('%Y-%m-%d')}:{conversation_name}"
            if attempt > 0:
                full_name += f"({attempt})"
            full_name += ".pkl"
            if not os.path.exists(os.path.join(PROJECT_DIR, "chat_history", full_name)):
                break
            attempt += 1
        os.makedirs(os.path.join(PROJECT_DIR, "chat_history"), exist_ok=True)
        save_file = os.path.join(PROJECT_DIR, "chat_history", full_name)
        with open(save_file, "wb") as f:
            pickle.dump(self, f)

    def add_screenshots(self, screenshots: List[bytes]) -> None:
        self.all_screenshots.append(screenshots)

    def add_user_message(self, text: str) -> None:
        self.history.append({
            "role": "user",
            "content": text,
            "timestamp": self._now()
        })

    def add_assistant_message(self, text: str) -> None:
        self.history.append({
            "role": "assistant",
            "content": text,
            "timestamp": self._now()
        })

    def build_context(self, now: Optional[datetime] = None) -> List[dict]:
        now = now or self._now()
        cutoff = now - timedelta(seconds=MAX_HISTORY_AGE_S)
        recent = [m for m in self.history if m.get("timestamp", now) >= cutoff]

        context = [self.system_prompt]
        assembled = []

        for i, message in enumerate(reversed(recent[-MAX_HISTORY_MESSAGES:])):
            role = message["role"]
            text_content = message["content"]

            assembled.append({
                "role": role,
                "content": [{"type": "text", "text": text_content}]
            })

            if i == 0 and role == "user" and self.all_screenshots and self.all_screenshots[-1]:
                for png in self.all_screenshots[-1][-NUM_LATEST_SCREENSHOTS:]:
                    code_block = extract_code_text(png)
                    assembled.append({
                        "role": role,
                        "content": [{"type": "text", "text": code_block}]
                    })

        full_context = context + list(reversed(assembled))
        return full_context

    def estimate_total_tokens(self, now: Optional[datetime] = None) -> int:
        return sum(estimate_message_tokens(m) for m in self.build_context(now))