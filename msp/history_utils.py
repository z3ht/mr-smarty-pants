import os
from datetime import datetime

from msp.settings import PROJECT_DIR

_TIMESTAMP_FMT = "%Y-%m-%d_%H-%M-%S"


def _ensure_history_dirs() -> None:
    os.makedirs(os.path.join(PROJECT_DIR, "history", "context"), exist_ok=True)
    os.makedirs(os.path.join(PROJECT_DIR, "history", "view"),    exist_ok=True)


def unique_history_basename(conversation_name: str) -> str:
    if not conversation_name:
        conversation_name = "unnamed"

    _ensure_history_dirs()

    attempt = 0
    while True:
        stamp = datetime.now().strftime(_TIMESTAMP_FMT)
        stem = f"{stamp}:{conversation_name}"
        if attempt:
            stem += f"({attempt})"

        ctx_file  = os.path.join(PROJECT_DIR, "history", "context", f"{stem}.pkl")
        view_file = os.path.join(PROJECT_DIR, "history", "view",    f"{stem}.pkl")

        if not (os.path.exists(ctx_file) or os.path.exists(view_file)):
            return stem
        attempt += 1


def _parse_timestamp(stem: str) -> datetime:
    ts_part = stem.split(":", 1)[0]
    return datetime.strptime(ts_part, _TIMESTAMP_FMT)


def get_previous_conversation_names() -> list[str]:
    ctx_dir = os.path.join(PROJECT_DIR, "history", "context")
    view_dir = os.path.join(PROJECT_DIR, "history", "view")

    if not (os.path.isdir(ctx_dir) and os.path.isdir(view_dir)):
        return []

    ctx_stems = {os.path.splitext(f)[0] for f in os.listdir(ctx_dir) if f.endswith(".pkl") and "__on_close__" not in f}
    view_stems = {os.path.splitext(f)[0] for f in os.listdir(view_dir) if f.endswith(".pkl") and "__on_close__" not in f}

    common_stems = ctx_stems & view_stems
    if not common_stems:
        return []

    return sorted(
        common_stems,
        key=lambda stem: (-_parse_timestamp(stem).timestamp(), stem.split(":", 1)[1].lower())
    )


def _last_closed_path() -> str:
    return os.path.join(PROJECT_DIR, "history", "last_closed.txt")


def get_last_closed_conversation() -> str:
    path = _last_closed_path()
    try:
        with open(path, "r", encoding="utf-8") as f:
            stem = f.read().strip()
            return stem or "__on_close__"
    except FileNotFoundError:
        return "__on_close__"


def save_last_closed_conversation(stem: str | None) -> None:
    _ensure_history_dirs()
    if not stem:
        stem = "__on_close__"

    path = _last_closed_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(stem)
    except OSError:
        pass
