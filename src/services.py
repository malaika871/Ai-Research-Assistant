import os
from huggingface_hub import InferenceClient
import config


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
    _openai_client = None

    @classmethod
    def get_client(cls):
        if cls._client is None:
            print("Initializing Hugging Face Inference client...")
            kwargs = {"token": os.getenv("HF_TOKEN")}
            if config.LLM_PROVIDER:
                kwargs["provider"] = config.LLM_PROVIDER
            cls._client = InferenceClient(**kwargs)
        return cls._client

    # Kept for backward compatibility with existing call sites.
    @classmethod
    def get_llm_client(cls):
        if config.LLM_BACKEND == "openai_compat":
            if cls._openai_client is None:
                from openai import OpenAI

                cls._openai_client = OpenAI(
                    base_url=config.LLM_BASE_URL or None,
                    api_key=config.LLM_API_KEY,
                )
            return cls._openai_client
        if config.LLM_BACKEND != "hf":
            raise ValueError(f"Unsupported LLM_BACKEND: {config.LLM_BACKEND}")
        return cls.get_client()

    @classmethod
    def get_embedding_client(cls):
        return cls.get_client()