import argparse
import os

from dotenv import load_dotenv


load_dotenv()

parser = argparse.ArgumentParser(description="Run Mr. Smarty Pants")
parser.add_argument("--window", type=str, help="Fuzzy match window title to capture")
args = parser.parse_args()

WINDOW_NAME = args.window if args.window else None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise SystemExit("Missing OPENAI_API_KEY; set it in .env file")

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_API_KEY:
    raise SystemExit("Missing MISTRAL_API_KEY; set it in .env file")

LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY")
if not LAMBDA_API_KEY:
    raise SystemExit("Missing LAMBDA_API_KEY; set it in .env file")

PROJECT_DIR = os.getenv("MR_SMARTY_PANTS_PROJECT_DIR")
ASSETS_DIR = os.path.join(PROJECT_DIR, "assets")

AUDIO_CHUNK_S = int(os.getenv("AUDIO_CHUNK_S", "3"))
VOICE_NAME = os.getenv("VOICE_NAME", "onyx")
CHAT_MODEL = os.getenv("CHAT_MODEL", "deepseek-r1-671b")
VISION_MODEL = os.getenv("VISION_MODEL", "llama3.2-11b-vision-instruct")
TTS_MODEL = os.getenv("TTS_MODEL", "gpt-4o-mini-tts")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "whisper-1")
SCREENSHOT_INTERVAL_S = float(os.getenv("SCREENSHOT_INTERVAL_S", "0.1"))
MOTION_THRESHOLD = float(os.getenv("MOTION_THRESHOLD", "500"))
SCREENSHOT_SIMILARITY_THRESHOLD_PCT = float(os.getenv("SCREENSHOT_SIMILARITY_THRESHOLD_PCT", "0.95"))
TOKEN_LIMIT_PER_M = int(os.getenv("TOKEN_LIMIT_PER_M", "200_000"))
MAX_SCREENSHOT_AGE_S = int(os.getenv("MAX_SCREENSHOT_AGE_S", "30"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "10"))
MAX_HISTORY_AGE_S = int(os.getenv("MAX_HISTORY_AGE_S", "300"))
NUM_LATEST_SCREENSHOTS = int(os.getenv("NUM_LATEST_SCREENSHOTS", "1"))

