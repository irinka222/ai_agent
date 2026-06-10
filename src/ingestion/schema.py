from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class DocumentBlock:
    """
    Блок текста (параграф, секция, формула и т.д.), извлечённый из документа.
    """
    paper_id: str                     # идентификатор документа (обычно имя файла без расширения)
    content: str                      # текст блока
    block_type: str = "paragraph"     # paragraph, section, formula, equation_block, title, authors, abstract, table
    page: Optional[int] = None
    section: Optional[str] = None     # название секции (если есть)
    formulas: List[Dict[str, Any]] = field(default_factory=list)  # список формул, связанных с блоком
    source: str = "unknown"           # откуда получен блок: "grobid", "pdfplumber", "ocr", "heuristic"
    language: Optional[str] = None    # язык текста (если определён)
    metadata: Dict[str, Any] = field(default_factory=dict)  # дополнительные данные (например, confidence, ast и т.д.)

    def to_dict(self) -> Dict[str, Any]:
        """Преобразует в словарь для совместимости со старым кодом."""
        return {
            "type": self.block_type,
            "content": self.content,
            "page": self.page,
            "section": self.section,
            "formulas": self.formulas,
            "source": self.source,
            "language": self.language,
            "metadata": self.metadata,
            "paper_id": self.paper_id,
        }



    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentBlock":
        """Создаёт экземпляр из словаря (для обратной совместимости)."""
        return cls(
            paper_id=data.get("paper_id", data.get("source", "")),
            content=data.get("content", ""),
            block_type=data.get("type", "paragraph"),
            page=data.get("page"),
            section=data.get("section"),
            formulas=data.get("formulas", []),
            source=data.get("source", "unknown"),
            language=data.get("language"),
            metadata=data.get("metadata", {}),
        )