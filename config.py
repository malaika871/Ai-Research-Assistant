# config.py
<<<<<<< HEAD
import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", ".env"), override=True)


def _get_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _get_list(name: str, default: list) -> list:
    val = os.getenv(name)
    if not val:
        return default
    return [item.strip() for item in val.split(",") if item.strip()]


# --- Environment ---
ENV = os.getenv("ENV", "development")  # development | staging | production
IS_PRODUCTION = ENV == "production"

# --- Embedding / LLM models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "5"))
# Used for whole-document requests (summaries, overviews) where narrow
# top-K similarity search misses most of the document's content.
SUMMARY_TOP_K = int(os.getenv("SUMMARY_TOP_K", "15"))

# --- Web search fallback ---
# Used when no documents are uploaded, or retrieval finds nothing relevant
# to the question (best match score below this threshold).
WEB_SEARCH_ENABLED = _get_bool("WEB_SEARCH_ENABLED", True)
WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "4"))
RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3"))

# --- Generation ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))

# --- Security ---
# Comma-separated list, e.g. "https://myapp.com,https://staging.myapp.com"
CORS_ORIGINS = _get_list("CORS_ORIGINS", ["http://localhost:3000"])
# Comma-separated list of valid API keys clients must send as: X-API-Key: <key>
API_KEYS = set(_get_list("API_KEYS", []))
REQUIRE_API_KEY = _get_bool("REQUIRE_API_KEY", False)  # flip on in production

# --- Uploads ---
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "10"))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# --- Rate limiting ---
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "10/minute")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "5/minute")

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
=======

# Embedding Model
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Chunking
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# Retrieval
TOP_K = 5
>>>>>>> 41cac17 (INitial COmmit)
