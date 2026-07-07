from src.services import ServiceHub


class Embedder:

    def __init__(self):
        self.model = ServiceHub.get_embedding_model()

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode([text])[0]