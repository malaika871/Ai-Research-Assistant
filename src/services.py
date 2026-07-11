import os

from huggingface_hub import InferenceClient

from config import EMBEDDING_MODEL, LLM_MODEL


class ServiceHub:
    """
    Single shared Hugging Face Inference client used for both:
    - chat completions (Qwen2.5-7B-Instruct)
    - feature extraction / embeddings (all-MiniLM-L6-v2)

    Both are hosted-inference calls now, so no model weights are downloaded
    or run in-process. This keeps the deployment footprint small (no torch,
    no sentence-transformers) at the cost of a network round trip per call.
    """

    _client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            print("Initializing Hugging Face Inference client...")
            cls._client = InferenceClient(token=os.getenv("HF_TOKEN"))
        return cls._client

    # Kept for backward compatibility with existing call sites.
    @classmethod
    def get_llm_client(cls):
        return cls.get_client()

    @classmethod
    def get_embedding_client(cls):
        return cls.get_client()
