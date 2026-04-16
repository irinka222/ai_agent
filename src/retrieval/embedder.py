import numpy as np
from sentence_transformers import SentenceTransformer
import re


class Embedder:
    def __init__(self, model_name="intfloat/multilingual-e5-large"):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_formula(self, formula: dict) -> np.ndarray:
        semantic = formula.get('semantic_description', '')
        latex = formula.get('latex', '')
        text_for_embed = f"{semantic} | formula: {latex}"
        return self.model.encode(text_for_embed, normalize_embeddings=True)

    def is_math_query(self, query: str) -> bool:
        math_patterns = [
            r'[=+\-*/^]', r'd/dt', r'd/dx', r'integral', r'уравн',
            r'формул', r'производн', r'дифференц', r'экспонен'
        ]
        return any(re.search(p, query.lower()) for p in math_patterns)