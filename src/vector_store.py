import chromadb


class VectorStore:

    def __init__(self):

        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="research_documents"
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

