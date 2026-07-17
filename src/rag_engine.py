from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
from src.vector_store import VectorStore
from src.web_search import WebSearcher
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
        query_embedding = self.embedder.embed_query(question)
        top_k = config.SUMMARY_TOP_K if intent == "DOCUMENT" else config.TOP_K
        return self.retriever.retrieve(query_embedding, top_k=top_k)

    @staticmethod
    def _has_relevant_results(results) -> bool:
        """Return whether retrieval found a plausible document match.

        A vector store will always return the nearest chunks, even when every
        chunk is unrelated to the question.  Treating those nearest chunks as
        context is what caused general/web questions to be answered with a
        refusal while still showing the uploaded paper as its source.
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

        # GENERAL questions are ambiguous: use the uploaded documents only
        # when retrieval actually found a relevant match.  Otherwise go to
        # Tavily directly so an unrelated paper is never presented as the
        # answer's source.
        if intent == "GENERAL" and not self._has_relevant_results(results):
            web_results = self._web_fallback(question)
            if web_results:
                return self.generator.generate_from_web(question, web_results)
            # Do not expose unrelated paper chunks if Tavily is unavailable;
            # generate an honest no-context response with no paper sources.
            results = []

        doc_answer = self.generator.generate(question, results)

        if self.generator.answer_is_insufficient(doc_answer["answer"]):
            web_results = self._web_fallback(question)
            if web_results:
                return self.generator.generate_from_web(question, web_results)

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
            results = []

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
