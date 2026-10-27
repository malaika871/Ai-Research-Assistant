# config.py
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
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "api").strip().lower()
if EMBEDDING_BACKEND not in {"api", "local"}:
    raise ValueError("EMBEDDING_BACKEND must be 'api' or 'local'")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "cpu")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_BACKEND = os.getenv("LLM_BACKEND", "hf").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Optional: force a specific Inference Provider (e.g. "novita", "together",
# "sambanova", "hf-inference") if HF's automatic "auto" provider selection
# can't find an active provider currently hosting LLM_MODEL. Blank = "auto".
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "").strip() or None
INTENT_MODE = os.getenv("INTENT_MODE", "auto").strip().lower()
if INTENT_MODE not in {"auto", "document"}:
    raise ValueError("INTENT_MODE must be 'auto' or 'document'")
REFUSAL_TEXT = os.getenv(
    "REFUSAL_TEXT",
    "I don't have enough information in the uploaded documents to answer this question.",
)

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "256"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
# --- Vector store ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_project_path(value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(PROJECT_ROOT, value))


CHROMA_PATH = _resolve_project_path(os.getenv("CHROMA_PATH", "chroma_db"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "research_documents_v2")

# Used for whole-document requests (summaries, overviews) where narrow
# top-K similarity search misses most of the document's content.
TOP_K = int(os.getenv("TOP_K", "5"))
SUMMARY_TOP_K = int(os.getenv("SUMMARY_TOP_K", "15"))
RETRIEVAL_SCORE_MARGIN = float(os.getenv("RETRIEVAL_SCORE_MARGIN", "0.15"))
CONTEXT_MAX_CHUNKS = int(os.getenv("CONTEXT_MAX_CHUNKS", "5"))
# --- Web search fallback ---
# Used when no documents are uploaded, or retrieval finds nothing relevant
# to the question (best match score below this threshold).
WEB_SEARCH_ENABLED = _get_bool("WEB_SEARCH_ENABLED", True)
USE_FALLBACK = _get_bool("USE_FALLBACK", True)
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
CORPUS_DIR = _resolve_project_path(os.getenv("CORPUS_DIR", "uploads"))
SOURCES_CSV = os.getenv("SOURCES_CSV", os.path.join(CORPUS_DIR, "sources.csv"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", "10"))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# --- Rate limiting ---
RATE_LIMIT_CHAT = os.getenv("RATE_LIMIT_CHAT", "10/minute")
RATE_LIMIT_UPLOAD = os.getenv("RATE_LIMIT_UPLOAD", "5/minute")

# --- Logging ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")