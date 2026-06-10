import logging
from typing import List, Tuple, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        pairs = [[query, chunk.get("content", "")] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda item: item[1], reverse=True)

        return [chunk for chunk, _ in scored[:top_k]]

    def rerank_with_scores(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Tuple[Dict[str, Any], float]]:
        if not chunks:
            return []

        pairs = [[query, chunk.get("content", "")] for chunk in chunks]
        scores = self.model.predict(pairs)

        scored = list(zip(chunks, scores))
        scored.sort(key=lambda item: item[1], reverse=True)

        return scored[:top_k]