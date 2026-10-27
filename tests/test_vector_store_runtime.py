import importlib
import logging
from pathlib import Path
from unittest.mock import patch

import chromadb
import numpy as np
import pytest

with patch("dotenv.load_dotenv"):
    import config
    from src.models import DocumentChunk
    from src.vector_store import VectorStore


def test_relative_configured_paths_resolve_from_project_root():
    assert config._resolve_project_path("chroma_hec") == str(
        (Path(config.PROJECT_ROOT) / "chroma_hec").resolve()
    )
    absolute_path = str((Path(config.PROJECT_ROOT) / "absolute-store").resolve())
    assert config._resolve_project_path(absolute_path) == absolute_path


def test_missing_collection_is_not_created_when_disabled(tmp_path, caplog):
    collection_name = "missing_collection"
    path = tmp_path / "chroma"

    with caplog.at_level(logging.INFO, logger="src.vector_store"):
        store = VectorStore(
            collection_name=collection_name,
            persist_path=str(path),
            create_if_missing=False,
        )

    assert store.collection is None
    assert store.count_chunks() == 0
    assert store.count_documents() == 0
    assert collection_name in caplog.text
    assert str(path.resolve()) in caplog.text
    assert "chunks=0" in caplog.text
    assert chromadb.PersistentClient(path=str(path)).list_collections() == []


def test_upload_write_creates_missing_collection(tmp_path):
    store = VectorStore(
        collection_name="upload_collection",
        persist_path=str(tmp_path / "chroma"),
        create_if_missing=False,
    )
    chunk = DocumentChunk(
        id="upload::p1::c0",
        source="upload.pdf",
        page=1,
        chunk_number=0,
        text="uploaded text",
    )

    store.add_documents([chunk], np.array([[1.0, 0.0]], dtype=np.float32))

    assert store.count_chunks() == 1
    assert store.list_documents() == ["upload.pdf"]


def test_rag_engine_document_mode_warns_for_missing_collection(caplog):
    rag_module = importlib.import_module("src.rag_engine")
    vector_store = type(
        "EmptyStore",
        (),
        {
            "collection_name": "hec_collection",
            "persist_path": "/tmp/hec_chroma",
            "count_chunks": lambda self: 0,
        },
    )()

    with (
        patch.object(config, "INTENT_MODE", "document"),
        patch.object(rag_module, "VectorStore", return_value=vector_store) as store_factory,
        patch.object(rag_module, "DocumentProcessor") as processor_factory,
        patch.object(rag_module, "Embedder"),
        patch.object(rag_module, "Retriever") as retriever_factory,
        patch.object(rag_module, "Generator"),
        patch.object(rag_module, "WebSearcher"),
        caplog.at_level(logging.WARNING, logger="src.rag_engine"),
    ):
        engine = rag_module.RAGEngine()

    store_factory.assert_called_once_with(create_if_missing=False)
    processor_factory.assert_called_once_with(vector_store=vector_store)
    retriever_factory.assert_called_once_with(vector_store=vector_store)
    assert engine.vector_store is vector_store
    assert "missing or empty" in caplog.text
    assert "hec_collection" in caplog.text


def test_health_returns_collection_counts_without_secrets():
    with patch("dotenv.load_dotenv"):
        backend_module = importlib.import_module("backend.main")
    vector_store = type(
        "CountedStore",
        (),
        {
            "collection_name": "hec_collection",
            "count_chunks": lambda self: 17,
            "count_documents": lambda self: 3,
        },
    )()
    engine = type("Engine", (), {"vector_store": vector_store})()

    with patch.object(backend_module, "engine", engine):
        health = backend_module.health()

    assert health == {
        "status": "ok",
        "collection_name": "hec_collection",
        "chunk_count": 17,
        "document_count": 3,
    }