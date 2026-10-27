import logging
import os

import chromadb
import config

logger = logging.getLogger(__name__)


class VectorStore:

    def __init__(
        self,
        collection_name=config.COLLECTION_NAME,
        persist_path=config.CHROMA_PATH,
        create_if_missing=True,
    ):
        self.collection_name = collection_name
        self.persist_path = os.path.abspath(os.fspath(persist_path))
        self.create_if_missing = create_if_missing

        self.client = chromadb.PersistentClient(
            path=self.persist_path
        )

        if create_if_missing:
            self.collection = self._create_or_open_collection()
        elif self._collection_exists():
            self.collection = self.client.get_collection(name=collection_name)
        else:
            self.collection = None

        chunk_count = self.count_chunks()
        logger.info(
            "Chroma path=%s collection=%s chunks=%d",
            self.persist_path,
            self.collection_name,
            chunk_count,
        )
        self._validate_embedding_model()

    def _collection_exists(self):
        collections = self.client.list_collections()
        names = {item if isinstance(item, str) else item.name for item in collections}
        return self.collection_name in names

    def _create_or_open_collection(self):
        # Explicit cosine distance keeps score = 1 - distance meaningful.
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": config.EMBEDDING_MODEL,
            },
        )

    def _ensure_collection(self):
        if self.collection is None:
            self.collection = self._create_or_open_collection()
            logger.info(
                "Chroma path=%s collection=%s chunks=%d",
                self.persist_path,
                self.collection_name,
                self.count_chunks(),
            )
            self._validate_embedding_model()

    def _validate_embedding_model(self):
        if self.collection is None:
            return
        stored_model = (self.collection.metadata or {}).get("embedding_model")
        if stored_model is None:
            logger.warning(
                "Collection %s has no embedding model metadata; assuming it is compatible with %s",
                self.collection_name,
                config.EMBEDDING_MODEL,
            )
        elif stored_model != config.EMBEDDING_MODEL:
            raise ValueError(
                f"Collection '{self.collection_name}' uses embedding model '{stored_model}', "
                f"but the configured model is '{config.EMBEDDING_MODEL}'. "
                "Use a separate collection or re-embed its documents."
            )

    def count_chunks(self):
        return self.collection.count() if self.collection is not None else 0

    def count_documents(self):
        return len(self.list_documents())

    @staticmethod
    def _metadata_value(value):
        if value is None:
            return ""
        if isinstance(value, (str, int, float, bool)):
            return value
        return str(value)

    def add_documents(
        self,
        chunks,
        embeddings
    ):
        if not chunks:
            return
        self._ensure_collection()

        ids = [
            chunk.id
            for chunk in chunks
        ]

        documents = [
            chunk.text
            for chunk in chunks
        ]

        metadatas = [
            {
                "source": self._metadata_value(chunk.source),
                "page": self._metadata_value(chunk.page),
                "chunk": self._metadata_value(chunk.chunk_number),
                "doc_id": self._metadata_value(chunk.doc_id),
                "title": self._metadata_value(chunk.title),
                "version": self._metadata_value(chunk.version),
                "status": self._metadata_value(chunk.status),
                "doc_type": self._metadata_value(chunk.doc_type),
                "effective_date": self._metadata_value(chunk.effective_date),
                "source_url": self._metadata_value(chunk.source_url),
                "section": self._metadata_value(chunk.section),
                "page_end": self._metadata_value(chunk.page_end),
            }
            for chunk in chunks
        ]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

    def delete_by_doc_id(self, doc_id):
        self.collection.delete(where={"doc_id": doc_id})

    def get_all_chunks(self, source_name):
        """
        Fetches every chunk belonging to a document directly (chromadb
        .get(), not a similarity-ranked ANN .query()). Used for whole-
        document requests like "summarize this" -- similarity search with
        any top_k cap risks silently missing chunks in a long document,
        which produces an incomplete summary. This guarantees completeness.
        """
        if self.collection is None:
            return []

        data = self.collection.get(
            where={"source": source_name},
            include=["documents", "metadatas"],
        )
        if not data or not data.get("documents"):
            return []

        chunks = list(zip(data["documents"], data["metadatas"]))
        # Preserve original document order (by page, then chunk number)
        # rather than whatever order chromadb happens to return.
        chunks.sort(key=lambda c: (c[1].get("page", 0), c[1].get("chunk", 0)))
        return chunks

    def search(
        self,
        query_embedding,
        top_k=5,
        where=None,
        metadata_filter=None,
    ):
        if where is not None and metadata_filter is not None:
            raise ValueError("Provide either 'where' or 'metadata_filter', not both")

        if self.collection is None:
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        return self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            where=metadata_filter if metadata_filter is not None else where,
        )

    def list_documents(self):
        if self.collection is None:
            return []

        data = self.collection.get(include=["metadatas"])
        if not data or not data.get("metadatas"):
            return []

        documents = sorted(
            {
                metadata["source"]
                for metadata in data["metadatas"]
                if metadata and "source" in metadata
            }
        )
        return documents

    def delete_document(self, document_name):
        if self.collection is None:
            return
        self.collection.delete(
            where={
                "source": document_name
            }
        )