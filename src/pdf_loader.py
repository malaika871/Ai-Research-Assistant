import fitz
from pathlib import Path


class PDFLoader:
    def __init__(self, pdf_path: str):
        self.pdf_path = Path(pdf_path)

    def extract_pages(self):

        if not self.pdf_path.exists():
            raise FileNotFoundError(
                f"{self.pdf_path} not found."
            )

        document = fitz.open(self.pdf_path)

        pages = []

        for page_number, page in enumerate(document):

            pages.append(
                {
                    "page": page_number + 1,
                    "text": page.get_text()
                }
            )

        document.close()

        return pages

    def extract_text(self):

        pages = self.extract_pages()

        return "\n".join(
            page["text"]
            for page in pages
        )