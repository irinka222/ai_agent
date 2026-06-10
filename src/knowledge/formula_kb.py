import uuid
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, field

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    cosine_similarity = None


@dataclass
class FormulaEntry:
    id: str
    canonical: str
    embedding: Optional[np.ndarray] = None
    variants: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    pages: List[int] = field(default_factory=list)
    score: float = 0.0
    domain: str = "unknown"


class FormulaKnowledgeBase:
    """
    Stores normalized formulas and merges duplicates across documents.
    """

    def __init__(self, similarity_threshold: float = 0.88):
        self.items: List[FormulaEntry] = []
        self.similarity_threshold = similarity_threshold

    def add_formula(
        self,
        formula: str,
        embedding: Optional[np.ndarray] = None,
        source: str = "",
        page: int = 0,
        domain: str = "electrochemistry",
        score: float = 0.0
    ) -> str:
        formula_norm = self._normalize(formula)

        existing = self._find_similar(formula_norm, embedding)
        if existing:
            self._merge(existing, formula_norm, source, page, score)
            return existing.id

        entry = FormulaEntry(
            id=str(uuid.uuid4()),
            canonical=formula_norm,
            embedding=embedding,
            variants=[formula_norm],
            sources=[source] if source else [],
            pages=[page] if page else [],
            score=score,
            domain=domain
        )
        self.items.append(entry)
        return entry.id

    def get_all(self) -> List[FormulaEntry]:
        return self.items

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[FormulaEntry]:
        if not self.items or cosine_similarity is None:
            return []
        scored = []
        for item in self.items:
            if item.embedding is None:
                continue
            sim = cosine_similarity([query_embedding], [item.embedding])[0][0]
            scored.append((sim, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:top_k]]

    def _find_similar(self, formula: str, embedding: Optional[np.ndarray]):
        for item in self.items:
            if self._normalize(item.canonical) == formula:
                return item
            if embedding is not None and item.embedding is not None and cosine_similarity is not None:
                if cosine_similarity([embedding], [item.embedding])[0][0] >= self.similarity_threshold:
                    return item
        return None

    def _merge(self, item: FormulaEntry, formula: str, source: str, page: int, score: float):
        if formula not in item.variants:
            item.variants.append(formula)
        if source and source not in item.sources:
            item.sources.append(source)
        if page and page not in item.pages:
            item.pages.append(page)
        item.score = max(item.score, score)

    def _normalize(self, formula: str) -> str:
        if not formula:
            return ""
        f = formula.lower()
        f = f.replace(" ", "")
        f = f.replace("\\", "")
        f = f.replace("{", "")
        f = f.replace("}", "")
        return f