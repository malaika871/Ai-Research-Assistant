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

    def _decide_context(self, question):
        """
        Routing, in priority order:

        1. No documents uploaded at all -> web, no classifier call needed
           (trivial, deterministic, saves an LLM round trip).
        2. Documents exist -> classify intent:
             DOCUMENT -> always answer from documents. Broad retrieval
                         (SUMMARY_TOP_K) since these are often whole-document
                         requests. Never silently falls back to web, even if
                         the documents don't cover it -- the generator's
                         prompt handles that by saying so honestly.
             WEB       -> always web search, regardless of what's in the
                         vector store (the question explicitly wants
                         current/external info).
             GENERAL   -> ambiguous standalone question. Try documents
                         (normal TOP_K) first; only fall back to web if
                         the best match is below the relevance threshold.

        Returns (context_mode, retrieved_chunks, web_results).
        """
        has_docs = self._has_documents()

        if not has_docs:
            if not config.WEB_SEARCH_ENABLED:
                return "document", [], []  # nothing to answer from, honest "no info" response
            web_results = self.web_searcher.search(question, max_results=config.WEB_SEARCH_MAX_RESULTS)
            return ("web", [], web_results) if web_results else ("document", [], [])

        intent = self.generator.classify_intent(question)

        if intent == "DOCUMENT":
            query_embedding = self.embedder.embed_query(question)
            results = self.retriever.retrieve(query_embedding, top_k=config.SUMMARY_TOP_K)
            return "document", results, []

        if intent == "WEB":
            if not config.WEB_SEARCH_ENABLED:
                # Fall back to whatever's in the documents rather than
                # returning nothing, even though this isn't ideal.
                query_embedding = self.embedder.embed_query(question)
                results = self.retriever.retrieve(query_embedding, top_k=config.TOP_K)
                return "document", results, []
            web_results = self.web_searcher.search(question, max_results=config.WEB_SEARCH_MAX_RESULTS)
            return ("web", [], web_results) if web_results else ("document", [], [])

        # intent == "GENERAL": similarity-threshold-based, as before, but
        # now scoped ONLY to genuinely ambiguous questions instead of
        # every single question.
        query_embedding = self.embedder.embed_query(question)
        results = self.retriever.retrieve(query_embedding, top_k=config.TOP_K)

        if not config.WEB_SEARCH_ENABLED:
            return "document", results, []

        if results and max(c.score for c in results) >= config.RETRIEVAL_SCORE_THRESHOLD:
            return "document", results, []

        web_results = self.web_searcher.search(question, max_results=config.WEB_SEARCH_MAX_RESULTS)
        if web_results and results:
            return "combined", results, web_results
        if web_results:
            return "web", [], web_results
        return "document", results, []

    def ask(self, question):
        context_mode, results, web_results = self._decide_context(question)

        if context_mode == "web":
            return self.generator.generate_from_web(question, web_results)
        if context_mode == "combined":
            return self.generator.generate_combined(question, results, web_results)
        return self.generator.generate(question, results)

    def ask_stream(self, question):
        context_mode, results, web_results = self._decide_context(question)

        if context_mode == "web":
            sources = [
                {"source": r["title"], "page": None, "url": r["url"]}
                for r in web_results
            ]
            token_gen = self.generator.generate_stream_from_web(question, web_results)
            return token_gen, sources, "web"

        if context_mode == "combined":
            sources = [
                {"source": r.source, "page": r.page, "url": None}
                for r in results
            ] + [
                {"source": r["title"], "page": None, "url": r["url"]}
                for r in web_results
            ]
            token_gen = self.generator.generate_stream_combined(question, results, web_results)
            return token_gen, sources, "combined"

        sources = [
            {"source": r.source, "page": r.page, "url": None}
            for r in results
        ]
        token_gen = self.generator.generate_stream(question, results)
        return token_gen, sources, "document"

    def list_documents(self):
        return self.vector_store.list_documents()

    def delete_document(self, document):
        self.vector_store.delete_document(document)