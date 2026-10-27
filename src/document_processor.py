import uuid
from pathlib import Path

from src.loader_factory import LoaderFactory
from src.chunker import TokenChunker
from src.embedder import Embedder
from src.vector_store import VectorStore
from src.models import DocumentChunk
import config


class DocumentProcessor:

    def __init__(self, vector_store=None):

        self.chunker = TokenChunker(
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
        )

        self.embedder = Embedder()

        self.vector_store = vector_store if vector_store is not None else VectorStore()

    def index(self, files, display_names=None, metadata=None):
        """
        display_names: optional dict {file_path: original_filename}. Needed
        because uploaded files are saved to disk with a UUID-prefixed name
        (to avoid collisions/overwrites), but users should see and be able
        to delete documents by their original filename, not the on-disk one.
        """
        display_names = display_names or {}
        stable_metadata = metadata is not None
        metadata = metadata or {}

        indexed_files = 0
        total_chunks = 0
        total_pages = 0

        for file_path in files:

            loader = LoaderFactory.get_loader(file_path)

            pages = loader.extract_pages()
            total_pages += len(pages)

            document_chunks = []
            source_name = display_names.get(file_path, Path(file_path).name)
            doc_id = metadata.get("doc_id") or f"upload-{uuid.uuid4().hex}"

            for page in pages:

                chunks = self.chunker.chunk_text(
                    page["text"]
                )

                for chunk_number, chunk in enumerate(chunks):

                    document_chunks.append(

                        DocumentChunk(

                            id=(
                                f"{doc_id}::p{page['page']}::c{chunk_number}"
                                if stable_metadata
                                else str(uuid.uuid4())
                            ),

                            source=source_name,

                            page=page["page"],

                            doc_id=doc_id,

                            title=metadata.get("title") or source_name,

                            version=metadata.get("version") or "",

                            status=metadata.get("status") or "",

                            doc_type=metadata.get("doc_type") or "",

                            effective_date=metadata.get("effective_date") or "",

                            source_url=metadata.get("source_url") or "",

                            section=metadata.get("section") or "",

                            page_end=page["page"],

                            chunk_number=chunk_number,

                            text=chunk

                        )

                    )

            if document_chunks:
                embeddings = self.embedder.embed_documents(
                    [c.text for c in document_chunks]
                )
                self.vector_store.add_documents(document_chunks, embeddings)

            indexed_files += 1

            total_chunks += len(document_chunks)

        return {

            "indexed_files": indexed_files,

            "total_chunks": total_chunks,

            "total_pages": total_pages

        }