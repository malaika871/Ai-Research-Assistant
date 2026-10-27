from dataclasses import dataclass


@dataclass
class DocumentChunk:
    id: str
    source: str
    page: int
    chunk_number: int
    text: str
    doc_id: str = ""
    title: str = ""
    version: str = ""
    status: str = ""
    doc_type: str = ""
    effective_date: str = ""
    source_url: str = ""
    section: str = ""
    page_end: int = 0


@dataclass
class RetrievalResult:
    text: str
    source: str
    page: int
    chunk_number: int
    score: float
    doc_id: str = ""