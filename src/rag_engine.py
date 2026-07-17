from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
from src.vector_store import VectorStore
from src.web_search import WebSearcher
from src.models import RetrievalResult
import config


class RAGEngine:

    def __init__(self):
        self.processor = DocumentProcessor()
        self.embedder = Embedder()
        self.retriever = Retriever()
        self.generator = Generator()
        self.vector_store = VectorStore()
        self.web_searcher = WebSearcher()

    def index_documents(self, files, display_names=None):
        return self.processor.index(files, display_names=display_names)

    def _has_documents(self) -> bool:
        return bool(self.vector_store.list_documents())

    def _retrieve_for_intent(self, question, intent):
        if intent == "DOCUMENT":
            docs = self.vector_store.list_documents()
            if len(docs) == 1:
                # Unambiguous: "the uploaded document" can only mean this
                # one. Fetch every chunk directly instead of a similarity-
                # ranked top-k, so a long document can't get silently
                # truncated into an incomplete summary.
                raw_chunks = self.vector_store.get_all_chunks(docs[0])
                return [
                    RetrievalResult(
                        text=text,
                        source=meta.get("source", docs[0]),
                        page=meta.get("page", 1),
                        chunk_number=meta.get("chunk", 0),
                        score=1.0,
                    )
                    for text, meta in raw_chunks
                ]
            # Multiple documents indexed: which one "the document" refers
            # to is ambiguous without conversation state. Falls back to a
            # broad similarity search across all of them as a reasonable
            # default, though this can still miss content in a specific
            # long document -- a known limitation until per-document
            # selection is added.
            query_embedding = self.embedder.embed_query(question)
            return self.retriever.retrieve(query_embedding, top_k=config.SUMMARY_TOP_K)

        query_embedding = self.embedder.embed_query(question)
        return self.retriever.retrieve(query_embedding, top_k=config.TOP_K)

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
        if not config.WEB_SEARCH_ENABLED:
            return None
        return self.web_searcher.search(question, max_results=config.WEB_SEARCH_MAX_RESULTS) or None

    def ask(self, question):
        has_docs = self._has_documents()

        if not has_docs:
            web_results = self._web_fallback(question)
            if web_results:
                return self.generator.generate_from_web(question, web_results)
            return self.generator.generate(question, [])  # honest "no info" response

        intent = self.generator.classify_intent(question)

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

        intent = self.generator.classify_intent(question)

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
        sources = [
            {"source": r.source, "page": r.page, "url": None}
            for r in results
        ]
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