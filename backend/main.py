import json
import logging
import os
import shutil
import uuid
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from src/.env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, "src", ".env")
load_dotenv(dotenv_path, override=True)

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sse_starlette.sse import EventSourceResponse

import config
from src.rag_engine import RAGEngine

# --- Logging setup ---
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("ai_research_assistant")

# --- App + rate limiter setup ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="AI Research Assistant API")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

engine: RAGEngine | None = None
UPLOAD_DIR = config.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --- Auth dependency ---
def verify_api_key(x_api_key: str = Header(default=None)):
    """
    Simple shared-secret API key check.
    Enabled via REQUIRE_API_KEY=true and populated API_KEYS in the environment.
    Left off by default for local development.
    """
    if not config.REQUIRE_API_KEY:
        return
    if not x_api_key or x_api_key not in config.API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# --- Schemas ---
class ChatRequest(BaseModel):
    question: str


class DeleteResponse(BaseModel):
    message: str


# --- Startup / shutdown ---
@app.on_event("startup")
def startup_event():
    global engine
    logger.info("Starting up. Warming up models...")
    try:
        from src.services import ServiceHub

        ServiceHub.get_client()  # single warm-up call: shared HF Inference client
        engine = RAGEngine()
        logger.info("Models loaded and RAGEngine ready.")
    except Exception:
        logger.exception("Failed to initialize models on startup.")
        raise


# --- Health check ---
@app.get("/")
def home():
    return {
        "message": "AI Research Assistant API is running 🚀",
        "env": config.ENV,
        "docs": "/docs",
    }


@app.get("/health")
def health():
    """Basic liveness/readiness probe for load balancers/orchestrators."""
    return {
        "status": "ok" if engine is not None else "starting",
    }


# --- Upload ---
def _validate_upload(file: UploadFile) -> None:
    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}",
        )


@app.post("/upload", dependencies=[Depends(verify_api_key)])
@limiter.limit(config.RATE_LIMIT_UPLOAD)
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    if len(files) > config.MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Max {config.MAX_FILES_PER_UPLOAD} per request.",
        )

    max_bytes = config.MAX_UPLOAD_MB * 1024 * 1024
    saved_paths = []
    display_names = {}

    try:
        for file in files:
            _validate_upload(file)

            # Enforce size limit while streaming to disk (avoid loading whole file in memory)
            safe_name = f"{uuid.uuid4().hex}_{Path(file.filename).name}"
            file_path = os.path.join(UPLOAD_DIR, safe_name)

            size = 0
            with open(file_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        buffer.close()
                        os.remove(file_path)
                        raise HTTPException(
                            status_code=413,
                            detail=f"File '{file.filename}' exceeds {config.MAX_UPLOAD_MB}MB limit.",
                        )
                    buffer.write(chunk)

            saved_paths.append(file_path)
            display_names[file_path] = Path(file.filename).name

        result = engine.index_documents(saved_paths, display_names=display_names)
        logger.info("Indexed %s file(s), %s chunk(s).", result.get("indexed_files"), result.get("total_chunks"))
        return result

    except HTTPException:
        raise
    except Exception:
        logger.exception("Failed to index uploaded documents.")
        raise HTTPException(status_code=500, detail="Failed to process uploaded document(s).")
    finally:
        # Clean up files on disk after indexing (vector store already has the content).
        for path in saved_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except OSError:
                logger.warning("Could not remove temp upload file: %s", path)


# --- Chat (streaming) ---
@app.post("/chat/stream", dependencies=[Depends(verify_api_key)])
@limiter.limit(config.RATE_LIMIT_CHAT)
async def chat_stream(request: Request, req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question is too long (max 2000 characters).")

    try:
        token_generator, sources, context_type = engine.ask_stream(question)
    except Exception:
        logger.exception("Failed to start retrieval for question=%r", question)
        raise HTTPException(status_code=500, detail="Failed to process the question.")

    async def event_generator():
        try:
            yield {"data": json.dumps({"sources": sources, "context_type": context_type})}

            for token in token_generator:
                yield {"data": json.dumps({"token": token})}

        except Exception:
            logger.exception("Error while streaming response for question=%r", question)
            yield {"data": json.dumps({"error": "An error occurred while generating the response."})}

    return EventSourceResponse(event_generator())


# --- Documents ---
@app.get("/documents", dependencies=[Depends(verify_api_key)])
def list_docs():
    try:
        return engine.list_documents()
    except Exception:
        logger.exception("Failed to list documents.")
        raise HTTPException(status_code=500, detail="Failed to list documents.")


@app.delete("/documents/{doc_name}", response_model=DeleteResponse, dependencies=[Depends(verify_api_key)])
def delete_doc(doc_name: str):
    try:
        engine.delete_document(doc_name)
        logger.info("Deleted document: %s", doc_name)
        return {"message": f"{doc_name} deleted"}
    except Exception:
        logger.exception("Failed to delete document: %s", doc_name)
        raise HTTPException(status_code=500, detail=f"Failed to delete '{doc_name}'.")


# --- Generic fallback handler so clients always get clean JSON, never a raw traceback ---
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error."},
    )