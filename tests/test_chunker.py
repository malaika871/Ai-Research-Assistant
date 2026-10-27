import re

import pytest

from src import chunker as chunker_module


class FakeFastTokenizer:
    is_fast = True

    def __call__(self, text, add_special_tokens, return_offsets_mapping):
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [match.span() for match in matches],
        }

    def encode(self, text, add_special_tokens=True):
        token_count = len(re.findall(r"\S+", text))
        return token_count + (2 if add_special_tokens else 0)


class FakeEmbedder:
    def get_tokenizer_name(self, model_name=None):
        return model_name or "test/model"

    def get_max_seq_tokens(self, model_name=None):
        return 12


@pytest.fixture
def make_chunker(monkeypatch):
    monkeypatch.setattr(chunker_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(
        chunker_module,
        "_load_fast_tokenizer",
        lambda model_name: FakeFastTokenizer(),
    )

    def create(chunk_size=4, overlap=1):
        return chunker_module.TokenChunker(chunk_size=chunk_size, overlap=overlap)

    return create


def test_chunks_are_substrings_of_original_text(make_chunker):
    page_text = "Opening   section, with unusual spacing. Then another part!"
    chunks = make_chunker().chunk_text(page_text)

    assert chunks
    assert all(chunk in page_text for chunk in chunks)


def test_chunks_preserve_capitalization_spacing_and_punctuation(make_chunker):
    page_text = "MiXeD   Case, stays! Exactly as-written."
    chunks = make_chunker(chunk_size=3, overlap=0).chunk_text(page_text)

    assert chunks[0] == "MiXeD   Case, stays!"
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert "MiXeD" in " ".join(chunks)
    assert "Case," in " ".join(chunks)


def test_chunks_fit_embedder_token_limit_with_special_tokens(make_chunker):
    chunks = make_chunker(chunk_size=4, overlap=1).chunk_text(
        "One two three four five six seven eight nine ten"
    )

    assert all(FakeFastTokenizer().encode(chunk) <= 12 for chunk in chunks)


def test_overlapping_windows_share_original_text(make_chunker):
    chunks = make_chunker(chunk_size=4, overlap=1).chunk_text(
        "zero one two three four five six"
    )

    assert chunks == ["zero one two three", "three four five six", "six"]


def test_non_fast_tokenizer_raises_clear_error(monkeypatch):
    class SlowTokenizer:
        is_fast = False

    monkeypatch.setattr(chunker_module, "Embedder", FakeEmbedder)
    monkeypatch.setattr(chunker_module, "_load_fast_tokenizer", lambda model_name: SlowTokenizer())

    with pytest.raises(ValueError, match="is not fast"):
        chunker_module.TokenChunker(chunk_size=4, overlap=1)