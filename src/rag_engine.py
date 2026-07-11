from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
from src.vector_store import VectorStore
<<<<<<< HEAD
from src.web_search import WebSearcher
import config
=======
>>>>>>> 41cac17 (INitial COmmit)


class RAGEngine:

    def __init__(self):
<<<<<<< HEAD
        self.processor = DocumentProcessor()
        self.embedder = Embedder()
        self.retriever = Retriever()
        self.generator = Generator()
        self.vector_store = VectorStore()
        self.web_searcher = WebSearcher()

    def index_documents(self, files):
        return self.processor.index(files)

    def _should_use_web(self, retrieved_chunks) -> bool:
        """
        Fall back to web search when:
        - web search is enabled, AND
        - either no chunks were retrieved at all (e.g. no documents
          uploaded), or the best match is below the relevance threshold
          (uploaded documents exist but don't cover this question).
        """
        if not config.WEB_SEARCH_ENABLED:
            return False
        if not retrieved_chunks:
            return True
        best_score = max(c.score for c in retrieved_chunks)
        return best_score < config.RETRIEVAL_SCORE_THRESHOLD

    def ask(self, question):
        query_embedding = self.embedder.embed_query(question)
        results = self.retriever.retrieve(query_embedding)

        if self._should_use_web(results):
            web_results = self.web_searcher.search(
                question, max_results=config.WEB_SEARCH_MAX_RESULTS
            )
            if web_results:
                return self.generator.generate_from_web(question, web_results)
            # Web search also came up empty -- fall through to the normal
            # document path, which will honestly say it doesn't know
            # rather than silently failing.

        return self.generator.generate(question, results)

    def ask_stream(self, question):
        query_embedding = self.embedder.embed_query(question)
        results = self.retriever.retrieve(query_embedding)

        if self._should_use_web(results):
            web_results = self.web_searcher.search(
                question, max_results=config.WEB_SEARCH_MAX_RESULTS
            )
            if web_results:
                sources = [
                    {"source": r["title"], "page": None, "url": r["url"]}
                    for r in web_results
                ]
                token_gen = self.generator.generate_stream_from_web(question, web_results)
                return token_gen, sources, "web"

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
=======

        self.processor = DocumentProcessor()

        self.embedder = Embedder()

        self.retriever = Retriever()

        self.generator = Generator()

        self.vector_store = VectorStore()

    def index_documents(self, files):

        return self.processor.index(files)

    def ask(self, question):

        query_embedding = self.embedder.embed_query(
            question
        )

        results = self.retriever.retrieve(
            query_embedding
        )

        return self.generator.generate(
            question,
            results
        )

    def ask_stream(self, question):
        query_embedding = self.embedder.embed_query(
            question
        )
        results = self.retriever.retrieve(
            query_embedding
        )
        return self.generator.generate_stream(question, results), results

    def list_documents(self):

        return self.vector_store.list_documents()

    def delete_document(self, document):

        self.vector_store.delete_document(
            document
        )
>>>>>>> 41cac17 (INitial COmmit)
