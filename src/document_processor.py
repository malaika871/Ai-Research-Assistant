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

    def index(self, files, display_names=None):
        """
        display_names: optional dict {file_path: original_filename}. Needed
        because uploaded files are saved to disk with a UUID-prefixed name
        (to avoid collisions/overwrites), but users should see and be able
        to delete documents by their original filename, not the on-disk one.
        """
        display_names = display_names or {}

        indexed_files = 0
        total_chunks = 0

        for file_path in files:

            loader = LoaderFactory.get_loader(file_path)

            pages = loader.extract_pages()

            document_chunks = []
            source_name = display_names.get(file_path, Path(file_path).name)

            for page in pages:

                chunks = self.chunker.chunk_text(
                    page["text"]
                )

                for chunk_number, chunk in enumerate(chunks):

                    document_chunks.append(

                        DocumentChunk(

                            id=str(uuid.uuid4()),

                            source=source_name,

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