from dataclasses import dataclass


@dataclass
class DocumentChunk:
    id: str
    source: str
    page: int
    chunk_number: int
    text: str


@dataclass
class RetrievalResult:
    text: str
    source: str
    page: int
    chunk_number: int
    score: float