import logging
import re

from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
from src.vector_store import VectorStore
from src.web_search import WebSearcher
from src.models import RetrievalResult
import config

logger = logging.getLogger(__name__)


class RAGEngine:

    def __init__(self, create_if_missing=None):
        if create_if_missing is None:
            create_if_missing = config.INTENT_MODE != "document"
        self.vector_store = VectorStore(create_if_missing=create_if_missing)
        self.processor = DocumentProcessor(vector_store=self.vector_store)
        self.embedder = Embedder()
        self.retriever = Retriever(vector_store=self.vector_store)
        self.generator = Generator()
        self.web_searcher = WebSearcher()
        if config.INTENT_MODE == "document" and self.vector_store.count_chunks() == 0:
            logger.warning(
                "Document intent mode is enabled, but Chroma collection '%s' at '%s' "
                "is missing or empty; document retrieval will have no indexed chunks.",
                self.vector_store.collection_name,
                self.vector_store.persist_path,
            )

    def index_documents(self, files, display_names=None):
        return self.processor.index(files, display_names=display_names)

    def _has_documents(self) -> bool:
        return bool(self.vector_store.list_documents())

    def _retrieve_for_intent(self, question, intent):
        summary_request = bool(
            re.search(
                r"\b(summarize|summary|overview)\b|\bwhat\s+is\s+this\s+document\s+about\b",
                question,
                flags=re.IGNORECASE,
            )
        )
        documents = self._document_records()
        named_documents = self._find_named_documents(question, documents)
        top_k = (
            config.SUMMARY_TOP_K
            if summary_request and len(named_documents) == 1
            else config.TOP_K
        )
        query_embedding = self.embedder.embed_query(question)
        cross_document = len(named_documents) > 1 or self._is_cross_document_question(question)
        if len(named_documents) > 1:
            per_document_results = []
            for record in named_documents:
                per_document_results.extend(
                    self.retriever.retrieve(
                        query_embedding,
                        top_k=config.TOP_K,
                        metadata_filter=self._document_filter(record),
                    )
                )
            return self._limit_context_across_documents(per_document_results)

        if cross_document:
            candidate_count = max(config.TOP_K, config.CONTEXT_MAX_CHUNKS * 3)
            results = self.retriever.retrieve(query_embedding, top_k=candidate_count)
            return self._limit_context_across_documents(results)

        metadata_filter = self._document_filter(named_documents[0]) if named_documents else None
        results = self.retriever.retrieve(
            query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )
        return self._limit_context(results)

    def _document_records(self):
        if hasattr(self.vector_store, "list_document_records"):
            return self.vector_store.list_document_records()
        return [
            {"source": source, "title": source, "doc_id": source}
            for source in self.vector_store.list_documents()
        ]

    @staticmethod
    def _normalize_document_label(value):
        return " ".join(re.sub(r"[^\w]+", " ", value.casefold()).split())

    @classmethod
    def _find_named_documents(cls, question, documents):
        normalized_question = f" {cls._normalize_document_label(question)} "
        matches = []
        for document in documents:
            aliases = (document.get("source"), document.get("title"), document.get("doc_id"))
            if any(
                alias
                and f" {cls._normalize_document_label(alias)} " in normalized_question
                for alias in aliases
            ):
                matches.append(document)
        return matches

    @staticmethod
    def _document_filter(document):
        doc_id = document.get("doc_id")
        if doc_id:
            return {"doc_id": doc_id}
        return {"source": document["source"]}

    @classmethod
    def _is_cross_document_question(cls, question):
        return bool(
            re.search(
                r"\b(compare|comparison|contrast|across|between|among|each|both|all)\b",
                question,
                flags=re.IGNORECASE,
            )
        )

    @staticmethod
    def _limit_context(results):
        if not results:
            return []
        top_score = max(result.score for result in results)
        score_floor = top_score - config.RETRIEVAL_SCORE_MARGIN
        filtered = [result for result in results if result.score >= score_floor]
        filtered.sort(key=lambda result: result.score, reverse=True)
        return filtered[: config.CONTEXT_MAX_CHUNKS]

    @staticmethod
    def _limit_context_across_documents(results):
        if not results:
            return []

        by_document = {}
        for result in results:
            key = result.doc_id or result.source
            by_document.setdefault(key, []).append(result)

        ranked_groups = []
        for group in by_document.values():
            top_score = max(result.score for result in group)
            score_floor = top_score - config.RETRIEVAL_SCORE_MARGIN
            ranked_groups.append(
                sorted(
                    (result for result in group if result.score >= score_floor),
                    key=lambda result: result.score,
                    reverse=True,
                )
            )

        selected = []
        while len(selected) < config.CONTEXT_MAX_CHUNKS and any(ranked_groups):
            for group in ranked_groups:
                if group and len(selected) < config.CONTEXT_MAX_CHUNKS:
                    selected.append(group.pop(0))
        return selected

    @staticmethod
    def _document_sources(results):
        best_by_document_page = {}
        for result in results:
            key = (result.doc_id or result.source, result.page)
            previous = best_by_document_page.get(key)
            if previous is None or result.score > previous.score:
                best_by_document_page[key] = result
        return [
            {"source": result.source, "page": result.page, "url": None}
            for result in best_by_document_page.values()
        ]

    @staticmethod
    def _has_relevant_results(results) -> bool:
        """
        A vector store always returns the nearest chunks, even when every
        chunk is unrelated to the question. This pre-check avoids wasting an
        LLM call generating a document answer for a GENERAL question when
        retrieval clearly found nothing relevant -- go straight to web
        instead.
        """
        return bool(results) and max(c.score for c in results) >= config.RETRIEVAL_SCORE_THRESHOLD

    def _web_fallback(self, question):
        if not config.USE_FALLBACK or not config.WEB_SEARCH_ENABLED:
            return None
        return self.web_searcher.search(question, max_results=config.WEB_SEARCH_MAX_RESULTS) or None

    def _classify_intent(self, question):
        if config.INTENT_MODE == "document":
            return "DOCUMENT"
        records = self._document_records()
        if self._find_named_documents(question, records):
            return "DOCUMENT"
        if len(records) > 1 and self._is_cross_document_question(question):
            return "DOCUMENT"
        if len(records) == 1 and re.search(r"\b(document|pdf|file|upload)\b", question, re.IGNORECASE):
            return "DOCUMENT"
        return self.generator.classify_intent(question)

    def ask(self, question):
        has_docs = self._has_documents()

        if not has_docs:
            web_results = self._web_fallback(question)
            if web_results:
                return self.generator.generate_from_web(question, web_results)
            return self.generator.generate(question, [])  # honest "no info" response

        intent = self._classify_intent(question)

        if intent == "WEB":
            web_results = self._web_fallback(question)
            if web_results:
                return self.generator.generate_from_web(question, web_results)
            # No web results even though web was explicitly requested -- fall
            # through to trying the documents rather than returning nothing.

        results = self._retrieve_for_intent(question, intent)

        if intent == "GENERAL" and not self._has_relevant_results(results):
            web_results = self._web_fallback(question)
            if web_results:
                return self.generator.generate_from_web(question, web_results)
            results = []  # don't cite an unrelated paper as this answer's source

        doc_answer = self.generator.generate(question, results)
        doc_answer["sources"] = self._document_sources(results)

        if self.generator.answer_is_insufficient(doc_answer["answer"]):
            web_results = self._web_fallback(question)
            if web_results:
                return self.generator.generate_from_web(question, web_results)
            # Web fallback also failed/unavailable. The answer is an honest
            # "I don't know" -- it should NOT still cite the uploaded
            # document as a source, since that document did not actually
            # answer the question. Showing "I don't have this information"
            # next to "Sources: research_paper.pdf" is misleading.
            doc_answer["sources"] = []

        return doc_answer

    def ask_stream(self, question):
        has_docs = self._has_documents()

        if not has_docs:
            web_results = self._web_fallback(question)
            if web_results:
                return self._stream_web(question, web_results)
            return self._stream_document(question, [])

        intent = self._classify_intent(question)

        if intent == "WEB":
            web_results = self._web_fallback(question)
            if web_results:
                return self._stream_web(question, web_results)

        results = self._retrieve_for_intent(question, intent)

        if intent == "GENERAL" and not self._has_relevant_results(results):
            web_results = self._web_fallback(question)
            if web_results:
                return self._stream_web(question, web_results)
            results = []  # don't cite an unrelated paper as this answer's source

        # Generate the full document answer first (not yet streamed to the
        # client) so we can check whether it's genuinely insufficient before
        # deciding whether to show it or switch to web search. The already-
        # generated text is then replayed as a stream, rather than
        # regenerating it, so this costs one extra LLM call only on the
        # (less common) insufficient-info path, not on every request.
        doc_answer = self.generator.generate(question, results)
        doc_answer["sources"] = self._document_sources(results)

        if self.generator.answer_is_insufficient(doc_answer["answer"]):
            web_results = self._web_fallback(question)
            if web_results:
                return self._stream_web(question, web_results)
            # Web fallback also failed -- don't attach the uploaded document
            # as a source for an "I don't know" answer.
            doc_answer["sources"] = []

        sources = doc_answer["sources"]
        token_gen = self._replay_as_stream(doc_answer["answer"])
        return token_gen, sources, "document"

    def _stream_web(self, question, web_results):
        sources = [
            {"source": r["title"], "page": None, "url": r["url"]}
            for r in web_results
        ]
        token_gen = self.generator.generate_stream_from_web(question, web_results)
        return token_gen, sources, "web"

    def _stream_document(self, question, results):
        sources = self._document_sources(results)
        token_gen = self.generator.generate_stream(question, results)
        return token_gen, sources, "document"

    @staticmethod
    def _replay_as_stream(text):
        """Yields an already-generated answer word by word, so the client's
        existing token-streaming UI still gets a typing effect, without a
        second LLM call for text that's already been generated and checked."""
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word if i == 0 else " " + word

    def list_documents(self):
        return self.vector_store.list_documents()

    def delete_document(self, document):
        self.vector_store.delete_document(document)