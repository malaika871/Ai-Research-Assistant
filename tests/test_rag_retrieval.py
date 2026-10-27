from unittest.mock import Mock, patch

from src.models import RetrievalResult
from src.rag_engine import RAGEngine


def make_engine(documents):
    engine = RAGEngine.__new__(RAGEngine)
    engine.vector_store = Mock()
    engine.vector_store.list_document_records.return_value = [
        {"source": document, "title": document, "doc_id": document}
        for document in documents
    ]
    engine.embedder = Mock()
    engine.embedder.embed_query.return_value = "query-vector"
    engine.retriever = Mock()
    engine.retriever.retrieve.return_value = []
    return engine


def test_normal_question_uses_top_k_even_when_document_named():
    engine = make_engine(["Refund-Policy.pdf"])

    engine._retrieve_for_intent("What does Refund-Policy.pdf say about refunds?", "DOCUMENT")

    engine.retriever.retrieve.assert_called_once_with(
        "query-vector",
        top_k=5,
        metadata_filter={"doc_id": "Refund-Policy.pdf"},
    )


def test_summary_uses_summary_top_k_only_for_named_document():
    engine = make_engine(["Refund-Policy.pdf", "Graduate-Policy.pdf"])

    engine._retrieve_for_intent("Give an overview of Refund-Policy.pdf", "DOCUMENT")

    engine.retriever.retrieve.assert_called_once_with(
        "query-vector",
        top_k=15,
        metadata_filter={"doc_id": "Refund-Policy.pdf"},
    )


def test_unnamed_summary_uses_normal_top_k():
    engine = make_engine(["Refund-Policy.pdf", "Graduate-Policy.pdf"])

    engine._retrieve_for_intent("Give an overview of the document", "DOCUMENT")

    engine.retriever.retrieve.assert_called_once_with("query-vector", top_k=5, metadata_filter=None)


def test_summary_mentioning_multiple_documents_does_not_choose_arbitrarily():
    engine = make_engine(["Refund-Policy.pdf", "Graduate-Policy.pdf"])

    engine._retrieve_for_intent(
        "Give an overview of Refund-Policy.pdf and Graduate-Policy.pdf",
        "DOCUMENT",
    )

    assert engine.retriever.retrieve.call_count == 2
    engine.retriever.retrieve.assert_any_call(
        "query-vector",
        top_k=5,
        metadata_filter={"doc_id": "Refund-Policy.pdf"},
    )
    engine.retriever.retrieve.assert_any_call(
        "query-vector",
        top_k=5,
        metadata_filter={"doc_id": "Graduate-Policy.pdf"},
    )


def test_document_title_and_doc_id_match_for_scoped_retrieval():
    engine = make_engine([])
    engine.vector_store.list_document_records.return_value = [
        {"source": "refund.pdf", "title": "National Fee Refund Policy", "doc_id": "refund-2024"}
    ]

    engine._retrieve_for_intent("What does the National Fee Refund Policy say?", "GENERAL")

    engine.retriever.retrieve.assert_called_once_with(
        "query-vector",
        top_k=5,
        metadata_filter={"doc_id": "refund-2024"},
    )


def test_score_margin_and_context_limit_keep_highest_scores():
    results = [
        RetrievalResult("low", "doc.pdf", 1, 0, 0.60),
        RetrievalResult("best", "doc.pdf", 1, 1, 0.90),
        RetrievalResult("second", "doc.pdf", 2, 0, 0.80),
        RetrievalResult("outside margin", "doc.pdf", 3, 0, 0.70),
    ]

    with (
        patch("src.rag_engine.config.RETRIEVAL_SCORE_MARGIN", 0.15),
        patch("src.rag_engine.config.CONTEXT_MAX_CHUNKS", 2),
    ):
        limited = RAGEngine._limit_context(results)

    assert [result.text for result in limited] == ["best", "second"]


def test_document_sources_deduplicate_by_document_and_page_using_best_score():
    results = [
        RetrievalResult("low", "doc.pdf", 1, 0, 0.70, doc_id="doc-id"),
        RetrievalResult("best", "doc.pdf", 1, 1, 0.90, doc_id="doc-id"),
        RetrievalResult("other page", "doc.pdf", 2, 0, 0.80, doc_id="doc-id"),
        RetrievalResult("other doc", "other.pdf", 1, 0, 0.85, doc_id="other-id"),
    ]

    sources = RAGEngine._document_sources(results)

    assert sources == [
        {"source": "doc.pdf", "page": 1, "url": None},
        {"source": "doc.pdf", "page": 2, "url": None},
        {"source": "other.pdf", "page": 1, "url": None},
    ]


def test_cross_document_context_interleaves_chunks_from_each_document():
    results = [
        RetrievalResult("a1", "a.pdf", 1, 0, 0.99, doc_id="a"),
        RetrievalResult("a2", "a.pdf", 2, 0, 0.98, doc_id="a"),
        RetrievalResult("a3", "a.pdf", 3, 0, 0.97, doc_id="a"),
        RetrievalResult("b1", "b.pdf", 1, 0, 0.82, doc_id="b"),
        RetrievalResult("b2", "b.pdf", 2, 0, 0.81, doc_id="b"),
    ]

    with (
        patch("src.rag_engine.config.RETRIEVAL_SCORE_MARGIN", 0.20),
        patch("src.rag_engine.config.CONTEXT_MAX_CHUNKS", 4),
    ):
        selected = RAGEngine._limit_context_across_documents(results)

    assert [result.text for result in selected] == ["a1", "b1", "a2", "b2"]