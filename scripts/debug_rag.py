from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import re
import sys
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Match backend.main's environment loading without displaying any values.
load_dotenv(PROJECT_ROOT / "src" / ".env", override=True)

import chromadb
import config
from src.embedder import Embedder
from src.generator import Generator
from src.models import RetrievalResult


SECRET_ENV_NAMES = ("HF_TOKEN", "LLM_API_KEY", "TAVILY_API_KEY", "API_KEYS")
SECRET_PATTERNS = (
    (re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"), r"\1<redacted>"),
    (re.compile(r"(?i)((?:api[_ -]?key|token|authorization)\s*[:=]\s*)[^\s,;]+"), r"\1<redacted>"),
)


def redact(text: Any) -> str:
    rendered = str(text)
    for name in SECRET_ENV_NAMES:
        secret = os.getenv(name, "")
        if not secret:
            continue
        values = [secret]
        if name == "API_KEYS":
            values.extend(part.strip() for part in secret.split(",") if part.strip())
        for value in values:
            rendered = rendered.replace(value, "<redacted>")
    for pattern, replacement in SECRET_PATTERNS:
        rendered = pattern.sub(replacement, rendered)
    return rendered


class CompletionCapture:
    def __init__(self, completions: Any, capture: dict[str, Any]):
        self._completions = completions
        self._capture = capture

    def create(self, **kwargs: Any) -> Any:
        response = self._completions.create(**kwargs)
        messages = kwargs.get("messages", [])
        if messages and str(messages[0].get("content", "")).startswith("Classify"):
            try:
                self._capture["raw"] = response.choices[0].message.content
            except (AttributeError, IndexError, TypeError):
                self._capture["raw"] = "<raw response unavailable>"
        return response


class ChatCapture:
    def __init__(self, chat: Any, capture: dict[str, Any]):
        self._chat = chat
        self.completions = CompletionCapture(chat.completions, capture)


class ClientCapture:
    def __init__(self, client: Any, capture: dict[str, Any]):
        self._client = client
        self.chat = ChatCapture(client.chat, capture)


def indexed_state(collection: Any) -> tuple[list[str], int]:
    total_chunks = collection.count()
    data = collection.get(include=["metadatas"])
    documents = sorted(
        {
            metadata["source"]
            for metadata in (data.get("metadatas") or [])
            if metadata and metadata.get("source")
        }
    )
    return documents, total_chunks


def retrieve(collection: Any, embedding: Any, top_k: int) -> list[RetrievalResult]:
    if collection.count() == 0:
        return []

    data = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    documents = data.get("documents", [[]])[0] or []
    metadatas = data.get("metadatas", [[]])[0] or []
    distances = data.get("distances", [[]])[0] or []

    return [
        RetrievalResult(
            text=text,
            source=metadata.get("source", ""),
            page=metadata.get("page", 0),
            chunk_number=metadata.get("chunk", 0),
            score=1 - distance,
        )
        for text, metadata, distance in zip(documents, metadatas, distances)
    ]


def retrieve_all_for_document(collection: Any, source_name: str) -> list[RetrievalResult]:
    data = collection.get(
        where={"source": source_name},
        include=["documents", "metadatas"],
    )
    chunks = list(zip(data.get("documents") or [], data.get("metadatas") or []))
    chunks.sort(key=lambda item: (item[1].get("page", 0), item[1].get("chunk", 0)))
    return [
        RetrievalResult(
            text=text,
            source=metadata.get("source", source_name),
            page=metadata.get("page", 0),
            chunk_number=metadata.get("chunk", 0),
            score=1.0,
        )
        for text, metadata in chunks
    ]


def safe_exception(exc: Exception) -> str:
    return f"{type(exc).__name__}: {redact(exc)}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only RAG pipeline diagnostic")
    parser.add_argument("question", help="Question to trace through the RAG pipeline")
    args = parser.parse_args()

    retrieval_threshold = getattr(config, "RETRIEVAL_SCORE_THRESHOLD", None)
    print(f"RETRIEVAL_SCORE_THRESHOLD={retrieval_threshold!r}")
    if hasattr(config, "TOP_K"):
        print(f"TOP_K={config.TOP_K!r}")
    else:
        print("TOP_K=<not configured; RAGEngine references config.TOP_K>")

    collection = None
    documents: list[str] = []
    total_chunks = 0
    try:
        client = chromadb.PersistentClient(path=config.CHROMA_PATH)
        collection = client.get_collection(name=config.COLLECTION_NAME)
        documents, total_chunks = indexed_state(collection)
        print(f"Indexed documents ({len(documents)}): {redact(documents)!r}")
        print(f"Total chunks: {total_chunks}")
    except Exception as exc:
        print(f"Collection read exception: {safe_exception(exc)}")

    logging.disable(logging.CRITICAL)
    try:
        generator = Generator()
    except Exception as exc:
        print(f"LLM client initialization exception: {safe_exception(exc)}")
        return 1

    raw_capture: dict[str, Any] = {}
    generator.client = ClientCapture(generator.client, raw_capture)
    classified_intent = "GENERAL"
    try:
        classified_intent = generator.classify_intent(args.question)
    except Exception as exc:
        print(f"Intent classification exception: {safe_exception(exc)}")
    raw_output = raw_capture.get("raw", "<no raw response captured>")
    if config.INTENT_MODE == "document":
        intent_used = "DOCUMENT"
    else:
        intent_used = classified_intent
    print(f"Raw classify_intent output: {redact(raw_output)!r}")
    print(f"Intent used (INTENT_MODE={config.INTENT_MODE!r}): {redact(intent_used)!r}")

    retrieved: list[RetrievalResult] = []
    embedding = None
    try:
        embedding = Embedder().embed_query(args.question)
        print(f"Query embedding length: {len(embedding)}")
    except Exception as exc:
        print(f"Query embedding exception: {safe_exception(exc)}")

    if embedding is not None:
        if collection is None:
            print("Retrieval skipped: collection could not be opened.")
        else:
            try:
                if intent_used == "DOCUMENT" and len(documents) == 1:
                    retrieved = retrieve_all_for_document(collection, documents[0])
                    retrieval_mode = "all chunks for the sole indexed document"
                else:
                    if intent_used == "DOCUMENT":
                        top_k = config.SUMMARY_TOP_K
                        retrieval_mode = "similarity search using SUMMARY_TOP_K"
                    else:
                        top_k = getattr(config, "TOP_K", 5)
                        retrieval_mode = "similarity search using TOP_K or Retriever default"
                    retrieved = retrieve(collection, embedding, top_k)
                print(f"Retrieval mode: {retrieval_mode}")
                print(f"Retrieved chunks ({len(retrieved)}):")
                for index, chunk in enumerate(retrieved, start=1):
                    preview = redact(chunk.text[:150])
                    print(
                        f"  {index}. source={redact(chunk.source)!r} "
                        f"page={chunk.page!r} score={chunk.score!r} "
                        f"text={preview!r}"
                    )
            except Exception as exc:
                print(f"Retrieval exception: {safe_exception(exc)}")

    try:
        answer = generator.generate(args.question, retrieved)["answer"]
        print(f"Raw generated answer: {redact(answer)!r}")
    except Exception as exc:
        print(f"Generation exception: {safe_exception(exc)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
