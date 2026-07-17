import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        # Explicitly use cosine distance. Without this, Chroma defaults to
        # squared L2 distance, which is NOT bounded to a 0-1 range -- this
        # silently broke the "score = 1 - distance" relevance threshold used
        # to decide between document vs. web-search fallback, causing even
        # highly relevant matches to score far below the threshold.
        self.collection = self.client.get_or_create_collection(
            name="research_documents_v2",
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
            }
            for chunk in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
        )

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