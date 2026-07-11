<<<<<<< HEAD
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
=======
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from config import EMBEDDING_MODEL, LLM_MODEL
import os


class ServiceHub:
    _embedding_model = None
    _llm_client = None

    @classmethod
    def get_embedding_model(cls):

        if cls._embedding_model is None:
            print("Loading embedding model...")
            cls._embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        return cls._embedding_model

    @classmethod
    def get_llm_client(cls):

        if cls._llm_client is None:
            print("Loading LLM client...")
            cls._llm_client = InferenceClient(
                token=os.getenv("HF_TOKEN")
            )

        return cls._llm_client
>>>>>>> 41cac17 (INitial COmmit)
