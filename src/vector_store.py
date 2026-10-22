import chromadb
import config


class VectorStore:

    def __init__(self, collection_name=config.COLLECTION_NAME, persist_path=config.CHROMA_PATH):

        self.client = chromadb.PersistentClient(
            path=persist_path
        )

        # Explicitly use cosine distance. Without this, Chroma defaults to
        # squared L2 distance, which is NOT bounded to a 0-1 range -- this
        # silently broke the "score = 1 - distance" relevance threshold used
        # to decide between document vs. web-search fallback, causing even
        # highly relevant matches to score far below the threshold.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        chunks,
        embeddings
    ):

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
                "source": chunk.source,
                "page": chunk.page,
                "chunk": chunk.chunk_number,
                "doc_id": chunk.doc_id if chunk.doc_id is not None else "",
                "title": chunk.title if chunk.title is not None else "",
                "version": chunk.version if chunk.version is not None else "",
                "status": chunk.status if chunk.status is not None else "",
                "doc_type": chunk.doc_type if chunk.doc_type is not None else "",
                "effective_date": chunk.effective_date if chunk.effective_date is not None else "",
                "source_url": chunk.source_url if chunk.source_url is not None else "",
                "section": chunk.section if chunk.section is not None else "",
                "page_end": chunk.page_end if chunk.page_end is not None else "",
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
    ):

        return self.collection.query(
            query_embeddings=[
                query_embedding.tolist()
            ],
            n_results=top_k,
            where=where,
        )

    def list_documents(self):
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
        self.collection.delete(
            where={
                "source": document_name
            }
        )