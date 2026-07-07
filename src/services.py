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