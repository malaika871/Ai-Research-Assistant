from pathlib import Path


class TXTLoader:

    def __init__(self, txt_path: str):
        self.txt_path = Path(txt_path)

    def extract_pages(self):

        if not self.txt_path.exists():
            raise FileNotFoundError(
                f"{self.txt_path} not found."
            )

        with open(
            self.txt_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

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