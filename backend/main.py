import os
from dotenv import load_dotenv

# Load environment variables from src/.env
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(base_dir, "src", ".env")
load_dotenv(dotenv_path)

from fastapi import FastAPI, UploadFile, File
from typing import List
import shutil

from sse_starlette import EventSourceResponse

from src.rag_engine import RAGEngine

app = FastAPI()
engine = RAGEngine()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/")
def home():
    return {
        "message": "AI Research Assistant API is running 🚀",
        "docs": "/docs"
    }

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):

    saved_paths = []

    for file in files:

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        saved_paths.append(file_path)

    result = engine.index_documents(saved_paths)

    return result

from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
import json
class ChatRequest(BaseModel):
    question: str

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    token_generator, retrieved_chunks = engine.ask_stream(req.question)

    async def event_generator():
        # First send the sources to the client
        sources = [
            {
                "source": c.source,
                "page": c.page,
                "score": float(getattr(c, "score", 1.0))
            }
            for c in retrieved_chunks
        ]
        yield {
            "data": json.dumps({"sources": sources})
        }

        # Then stream the tokens
        for token in token_generator:
            yield {
                "data": json.dumps({"token": token})
            }

    return EventSourceResponse(event_generator())
@app.get("/documents")
def list_docs():

    return engine.list_documents()

@app.delete("/documents/{doc_name}")
def delete_doc(doc_name: str):

    engine.delete_document(doc_name)

    return {"message": f"{doc_name} deleted"}

@app.on_event("startup")
def startup_event():
    print("Warming up models...")

    from src.services import ServiceHub
    ServiceHub.get_embedding_model()
    ServiceHub.get_llm_client()