from pathlib import Path
from docx import Document


class DocLoader:

    def __init__(self, doc_path: str):
        self.doc_path = Path(doc_path)

    def extract_pages(self):
        """
        DOCX doesn't have real pages.
        We'll treat the whole document as page 1.
        """

        if not self.doc_path.exists():
            raise FileNotFoundError(
                f"{self.doc_path} not found."
            )

        document = Document(self.doc_path)

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )

        return [
            {
                "page": 1,
                "text": text
            }
        ]

    def extract_text(self):

        pages = self.extract_pages()

        return "\n".join(
            page["text"]
            for page in pages
        )