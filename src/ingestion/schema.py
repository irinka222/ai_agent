# src/ingestion/schema.py

from dataclasses import dataclass
from typing import Optional


@dataclass
class DocumentBlock:
    paper_id: str
    page: int
    content: str
    block_type: str = "text"
    source: str = "unknown"  # pdfplumber | ocr
    language: Optional[str] = None