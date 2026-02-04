from pypdf import PdfReader
from pathlib import Path

class PDFLoader:

    """A simple PDF loader to read and extract text from PDF files."""

    def __init__(self, pdf_path: Path):
        print(f"Loading PDF from path: {pdf_path}")
        if not pdf_path.exists() or not pdf_path.is_file():
            raise FileNotFoundError(f"The file {pdf_path} does not exist.")
        self.pdf_path = pdf_path

    def load(self) -> str:
        reader = PdfReader(self.pdf_path)
        pages_text = []

        for page in reader.pages:
            text=page.extract_text()
            if text:
                pages_text.append(text)

        return "\n".join(pages_text)
    