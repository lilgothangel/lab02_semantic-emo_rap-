# налаштування, ключ у .env
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "corpus"
OUT_DIR = ROOT / "output"
OUT_DIR.mkdir(exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# мінімум "думання": low / minimal для gemini-3, 0 для gemini-2.5, порожньо - як у моделі
GEMINI_THINKING = os.getenv("GEMINI_THINKING", "low")

GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# пауза між запитами через ліміт free tier
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "6.5"))

CHUNK_WORDS = 120
CHUNK_OVERLAP_SENT = 1
TOP_N = 5

GLOBAL_QUESTION = (
    "Хто є ключовими фігурами емо-репу, як вони пов'язані між собою "
    "і які спільні теми та проблеми повторюються в їхніх історіях?"
)
