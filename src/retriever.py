from src.vector_store import VectorStore
from src.models import RetrievalResult


class Retriever:

    def __init__(self, vector_store=None):

        self.vector_store = vector_store if vector_store is not None else VectorStore()

    def retrieve(
        self,
        query_embedding,
        top_k=5,
        document_filter=None
    ):

        where = None

        if document_filter:

            where = {
                "source": document_filter
            }

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where
        )

        retrieved_chunks = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances
        ):

            retrieved_chunks.append(

                RetrievalResult(
                    text=document,
                    source=metadata["source"],
                    page=metadata["page"],
                    chunk_number=metadata["chunk"],
                    score=1 - distance,
                    doc_id=metadata.get("doc_id", ""),
                )

            )

        return retrieved_chunks