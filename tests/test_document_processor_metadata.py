from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import fitz
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Config loads environment files at import time. Suppress that loader here so
# this test uses only process defaults and never reads a local .env file.
with patch("dotenv.load_dotenv"):
    from src import document_processor as processor_module
    from src.vector_store import VectorStore


class FixedChunker:
    def chunk_text(self, text):
        return ["first test chunk", "second test chunk"]


class FixedEmbedder:
    def embed_documents(self, texts):
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)


class DocumentProcessorMetadataTest(unittest.TestCase):
    def test_reingesting_document_upserts_stable_chunk_ids(self):
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            pdf_path = tmp_path / "sample.pdf"
            pdf = fitz.open()
            page = pdf.new_page()
            page.insert_text((72, 72), "Small indexing regression test")
            pdf.save(pdf_path)
            pdf.close()

            store = VectorStore(
                collection_name="temporary_test_collection",
                persist_path=str(tmp_path / "chroma"),
            )

            with (
                patch.object(processor_module, "TokenChunker", return_value=FixedChunker()),
                patch.object(processor_module, "Embedder", return_value=FixedEmbedder()),
                patch.object(processor_module, "VectorStore", return_value=store),
            ):
                processor = processor_module.DocumentProcessor()
                metadata = {
                    "doc_id": "test-doc-001",
                    "title": "Test document",
                    "version": "1",
                    "status": "active",
                    "doc_type": "test",
                    "effective_date": "2026-01-01",
                    "source_url": "https://example.invalid/test.pdf",
                }

                processor.index([str(pdf_path)], metadata=metadata)
                count_after_first_index = store.collection.count()
                processor.index([str(pdf_path)], metadata=metadata)

            self.assertEqual(count_after_first_index, 2)
            self.assertEqual(store.collection.count(), count_after_first_index)
            stored = store.collection.get(include=["metadatas"])
            self.assertTrue(
                all(item["doc_id"] == "test-doc-001" for item in stored["metadatas"])
            )

    def test_same_named_uploads_receive_distinct_document_ids(self):
        with TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            paths = [tmp_path / "first.pdf", tmp_path / "second.pdf"]
            for path in paths:
                pdf = fitz.open()
                pdf.new_page().insert_text((72, 72), "Unique upload identity test")
                pdf.save(path)
                pdf.close()

            store = VectorStore(
                collection_name="temporary_duplicate_names",
                persist_path=str(tmp_path / "chroma"),
            )
            with (
                patch.object(processor_module, "TokenChunker", return_value=FixedChunker()),
                patch.object(processor_module, "Embedder", return_value=FixedEmbedder()),
            ):
                processor = processor_module.DocumentProcessor(vector_store=store)
                processor.index(
                    [str(path) for path in paths],
                    display_names={str(path): "same.pdf" for path in paths},
                )

            stored = store.collection.get(include=["metadatas"])
            doc_ids = {metadata["doc_id"] for metadata in stored["metadatas"]}
            self.assertEqual(len(doc_ids), 2)
            self.assertEqual(len(store.list_documents()), 2)

            for doc_id in doc_ids:
                store.delete_by_doc_id(doc_id)
            self.assertEqual(store.collection.count(), 0)


if __name__ == "__main__":
    unittest.main()