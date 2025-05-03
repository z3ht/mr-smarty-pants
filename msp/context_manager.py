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

    def save_conversation(self, stem_name: str):
        save_file = os.path.join(PROJECT_DIR, "history", "context", f"{stem_name}.pkl")
        with open(save_file, "wb") as f:
            pickle.dump(self, f)

    def load_conversation(self, stem_name: str) -> bool:
        file_path = os.path.join(PROJECT_DIR, "history", "context", f"{stem_name}.pkl")

        if not os.path.exists(file_path):
            print(f"[ContextManager.load_conversation] Not found: {file_path}")
            return False

        try:
            with open(file_path, "rb") as f:
                loaded = pickle.load(f)
        except Exception as e:
            print(f"[ContextManager.load_conversation] Error reading {file_path}: {e}")
            return False

        if not isinstance(loaded, ContextManager):
            print(f"[ContextManager.load_conversation] File does not contain a ContextManager object")
            return False

        self.__dict__.update(loaded.__dict__)
        return True

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