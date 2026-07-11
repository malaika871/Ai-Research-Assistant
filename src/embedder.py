<<<<<<< HEAD
import logging

import numpy as np

from src.services import ServiceHub
from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Embedder:
    """
    Calls the Hugging Face Inference API's feature-extraction endpoint for
    sentence-transformers/all-MiniLM-L6-v2 instead of loading the model
    in-process. Same model, same weights, same embedding space -> retrieval
    quality is unaffected. The only difference is a network call per text
    instead of a local forward pass.
    """

    def __init__(self):
        self.client = ServiceHub.get_embedding_client()

    def embed_documents(self, texts):
        vectors = [self._embed_one(t) for t in texts]
        return np.array(vectors)

    def embed_query(self, text):
        return self._embed_one(text)

    def _embed_one(self, text: str) -> np.ndarray:
        try:
            result = self.client.feature_extraction(text, model=EMBEDDING_MODEL)
        except Exception:
            logger.exception("Embedding request failed for text of length %d", len(text))
            raise RuntimeError("The embedding service is currently unavailable. Please try again shortly.")

        vec = np.array(result, dtype=np.float32)

        # Some inference backends return per-token embeddings (2D) instead of
        # a single pooled sentence vector (1D). Mean-pool if needed so the
        # rest of the pipeline (which expects one vector per text) still works.
        if vec.ndim == 2:
            vec = vec.mean(axis=0)

        return vec
=======
from src.services import ServiceHub


class Embedder:

    def __init__(self):
        self.model = ServiceHub.get_embedding_model()

    def embed_documents(self, texts):
        return self.model.encode(texts)

    def embed_query(self, text):
        return self.model.encode([text])[0]
>>>>>>> 41cac17 (INitial COmmit)
