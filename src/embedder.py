import logging
import threading
import time

import numpy as np

import config
from src.services import ServiceHub

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding backend cannot produce valid vectors."""


MODEL_REGISTRY = {
    "sentence-transformers/all-MiniLM-L6-v2": {
        "query_prefix": "",
        "document_prefix": "",
        "max_seq_tokens": 256,
    },
    "BAAI/bge-small-en-v1.5": {
        "query_prefix": "Represent this sentence for searching relevant passages: ",
        "document_prefix": "",
        "max_seq_tokens": 512,
    },
    "intfloat/multilingual-e5-small": {
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "max_seq_tokens": 512,
    },
}

_LOCAL_MODELS = {}
_LOCAL_MODELS_LOCK = threading.Lock()
_API_MAX_ATTEMPTS = 4
_API_RETRY_BASE_SECONDS = 0.5


def _model_metadata(model_name):
    metadata = MODEL_REGISTRY.get(model_name)
    if metadata is None:
        logger.warning(
            "Unknown embedding model %r; using no prefixes and a 256-token limit",
            model_name,
        )
        return {"query_prefix": "", "document_prefix": "", "max_seq_tokens": 256}
    return metadata


def _get_local_model():
    cache_key = (config.EMBEDDING_MODEL, config.EMBEDDING_DEVICE)
    model = _LOCAL_MODELS.get(cache_key)
    if model is not None:
        return model

    with _LOCAL_MODELS_LOCK:
        model = _LOCAL_MODELS.get(cache_key)
        if model is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(
                config.EMBEDDING_MODEL,
                device=config.EMBEDDING_DEVICE,
            )
            _LOCAL_MODELS[cache_key] = model
    return model


class Embedder:
    def __init__(self):
        self.client = None

    def get_max_seq_tokens(self, model_name=None):
        return _model_metadata(model_name or config.EMBEDDING_MODEL)["max_seq_tokens"]

    def get_tokenizer_name(self, model_name=None):
        return model_name or config.EMBEDDING_MODEL

    def embed_documents(self, texts):
        texts = list(texts)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        metadata = _model_metadata(config.EMBEDDING_MODEL)
        prepared_texts = [metadata["document_prefix"] + text for text in texts]
        return self._embed_prepared(prepared_texts)

    def _embed_prepared(self, prepared_texts):
        batch_size = max(1, int(config.EMBEDDING_BATCH_SIZE))
        vectors = []

        for start in range(0, len(prepared_texts), batch_size):
            batch = prepared_texts[start : start + batch_size]
            if config.EMBEDDING_BACKEND == "local":
                batch_vectors = self._embed_local(batch, batch_size)
            elif config.EMBEDDING_BACKEND == "api":
                batch_vectors = self._embed_api(batch)
            else:
                raise EmbeddingError(
                    f"Unsupported embedding backend '{config.EMBEDDING_BACKEND}' "
                    f"for model '{config.EMBEDDING_MODEL}'."
                )
            vectors.extend(batch_vectors)

        return self._normalize(np.asarray(vectors, dtype=np.float32))

    def embed_query(self, text):
        prefix = _model_metadata(config.EMBEDDING_MODEL)["query_prefix"]
        return self._embed_prepared([prefix + text])[0]

    def _embed_local(self, texts, batch_size):
        try:
            vectors = _get_local_model().encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
            return self._coerce_batch(vectors, len(texts))
        except Exception as exc:
            logger.warning(
                "Local embedding failed for model %s (%s)",
                config.EMBEDDING_MODEL,
                type(exc).__name__,
            )
            raise EmbeddingError(
                f"Embedding failed for model '{config.EMBEDDING_MODEL}' "
                "on backend 'local'."
            ) from exc

    def _embed_api(self, texts):
        if self.client is None:
            self.client = ServiceHub.get_embedding_client()

        for attempt in range(_API_MAX_ATTEMPTS):
            try:
                result = self.client.feature_extraction(
                    texts,
                    model=config.EMBEDDING_MODEL,
                )
                return self._coerce_batch(result, len(texts))
            except Exception as exc:
                error_kind = self._api_error_kind(exc)
                if error_kind == "credits":
                    raise EmbeddingError(
                        f"Embedding credits are exhausted for model "
                        f"'{config.EMBEDDING_MODEL}' on backend 'api'."
                    ) from exc
                if error_kind == "unavailable":
                    raise EmbeddingError(
                        f"Embedding model '{config.EMBEDDING_MODEL}' is unavailable "
                        "on backend 'api'."
                    ) from exc
                if error_kind != "transient" or attempt == _API_MAX_ATTEMPTS - 1:
                    logger.warning(
                        "API embedding failed for model %s (%s)",
                        config.EMBEDDING_MODEL,
                        type(exc).__name__,
                    )
                    raise EmbeddingError(
                        f"Embedding failed for model '{config.EMBEDDING_MODEL}' "
                        "on backend 'api'."
                    ) from exc

                delay = _API_RETRY_BASE_SECONDS * (2**attempt)
                logger.warning(
                    "Transient API embedding failure for model %s (%s); retry %d/%d",
                    config.EMBEDDING_MODEL,
                    type(exc).__name__,
                    attempt + 1,
                    _API_MAX_ATTEMPTS - 1,
                )
                time.sleep(delay)

        raise EmbeddingError(
            f"Embedding failed for model '{config.EMBEDDING_MODEL}' on backend 'api'."
        )

    @staticmethod
    def _api_error_kind(exc):
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        message = str(exc).lower()

        if any(
            term in message
            for term in (
                "credits are exhausted",
                "credit limit",
                "out of credits",
                "not enough credits",
                "exceeded your monthly included credits",
                "quota exceeded",
            )
        ):
            return "credits"
        if status_code == 404 or any(
            term in message
            for term in (
                "model not found",
                "model is unavailable",
                "model unavailable",
                "model is not supported",
            )
        ):
            return "unavailable"
        if status_code == 429 or status_code in {500, 502, 503, 504}:
            return "transient"
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return "transient"
        if any(term in message for term in ("rate limit", "timed out", "timeout", "temporarily unavailable")):
            return "transient"
        return "permanent"

    @staticmethod
    def _coerce_batch(vectors, expected_count):
        try:
            array = np.asarray(vectors, dtype=np.float32)
        except ValueError:
            items = [np.asarray(item, dtype=np.float32) for item in vectors]
            if len(items) != expected_count:
                raise ValueError("Embedding backend returned an unexpected vector shape")
            pooled = [item.mean(axis=0) if item.ndim == 2 else item for item in items]
            array = np.stack(pooled)
        if array.ndim == 1:
            if expected_count != 1:
                raise ValueError("Embedding backend returned an unexpected vector shape")
            array = array[np.newaxis, :]
        elif array.ndim == 2 and expected_count == 1 and array.shape[0] != 1:
            array = array.mean(axis=0, keepdims=True)
        elif array.ndim == 3:
            array = array.mean(axis=1)

        if array.ndim != 2 or array.shape[0] != expected_count or array.shape[1] == 0:
            raise ValueError("Embedding backend returned an unexpected vector shape")
        return [row for row in array]

    @staticmethod
    def _normalize(vectors):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise EmbeddingError(
                f"Embedding model '{config.EMBEDDING_MODEL}' on backend "
                f"'{config.EMBEDDING_BACKEND}' returned invalid vectors."
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise EmbeddingError(
                f"Embedding model '{config.EMBEDDING_MODEL}' on backend "
                f"'{config.EMBEDDING_BACKEND}' returned a zero vector."
            )
        return (vectors / norms).astype(np.float32, copy=False)