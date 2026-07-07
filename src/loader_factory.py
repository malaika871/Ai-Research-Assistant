from pathlib import Path

from src.pdf_loader import PDFLoader
from src.doc_loader import DocLoader
from src.txt_loader import TXTLoader


class LoaderFactory:

    LOADERS = {
        ".pdf": PDFLoader,
        ".docx": DocLoader,
        ".txt": TXTLoader,
    }

    @classmethod
    def get_loader(cls, file_path: str):

        extension = Path(file_path).suffix.lower()

        loader = cls.LOADERS.get(extension)

        if loader is None:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return loader(file_path)