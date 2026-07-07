import uuid
from pathlib import Path

from src.loader_factory import LoaderFactory
from src.chunker import TokenChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.models import DocumentChunk


class DocumentProcessor:

    def __init__(self):

        self.chunker = TokenChunker()

        self.embedder = Embedder()

        self.vector_store = VectorStore()

    def index(self, files):

        indexed_files = 0
        total_chunks = 0

        for file_path in files:

            loader = LoaderFactory.get_loader(file_path)

            pages = loader.extract_pages()

            document_chunks = []

            for page in pages:

                chunks = self.chunker.chunk_text(
                    page["text"]
                )

                for chunk_number, chunk in enumerate(chunks):

                    document_chunks.append(

                        DocumentChunk(

                            id=str(uuid.uuid4()),

                            source=Path(file_path).name,

                            page=page["page"],

                            chunk_number=chunk_number,

                            text=chunk

                        )

                    )

            embeddings = self.embedder.embed_documents(

                [c.text for c in document_chunks]

            )

            self.vector_store.add_documents(

                document_chunks,

                embeddings

            )

            indexed_files += 1

            total_chunks += len(document_chunks)

        return {

            "indexed_files": indexed_files,

            "total_chunks": total_chunks

        }