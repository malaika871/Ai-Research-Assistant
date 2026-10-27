import config
from src.embedder import Embedder


def _load_fast_tokenizer(model_name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, use_fast=True)


class TokenChunker:
    def __init__(self, model_name=None,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    ):
        self.embedder = Embedder()
        model_name = self.embedder.get_tokenizer_name(model_name)
        self.tokenizer = _load_fast_tokenizer(model_name)
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError(
                f"Tokenizer for model '{model_name}' is not fast; offset mappings are required."
            )
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")

        max_seq_tokens = self.embedder.get_max_seq_tokens(model_name)
        document_prefix = self.embedder.get_document_prefix(model_name)
        prefix_tokens = len(
            self.tokenizer(
                document_prefix,
                add_special_tokens=False,
            )["input_ids"]
        )
        special_tokens = self.tokenizer.num_special_tokens_to_add(pair=False)
        content_limit = max_seq_tokens - prefix_tokens - special_tokens
        if content_limit <= 0:
            raise ValueError(
                f"Model '{model_name}' leaves no token capacity for document text "
                "after its prefix and special tokens."
            )

        self.chunk_size = min(chunk_size, content_limit)
        if overlap >= self.chunk_size:
            raise ValueError(
                "overlap must be smaller than the effective chunk size after "
                "applying the embedding model token limit"
            )
        self.overlap = overlap

    def chunk_text(self, text):
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        tokens = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        chunks = []
        start = 0
        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_text = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if chunk_text:
                chunks.append(chunk_text)
            start += self.chunk_size - self.overlap
        return chunks