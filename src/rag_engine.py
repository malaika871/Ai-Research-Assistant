from src.document_processor import DocumentProcessor
from src.embedder import Embedder
from src.retriever import Retriever
from src.generator import Generator
from src.vector_store import VectorStore


class RAGEngine:

    def __init__(self):

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