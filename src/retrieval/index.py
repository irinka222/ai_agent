import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any
from src.retrieval.embedder import Embedder


class DualIndex:
    def __init__(self, embedder: Embedder, dim=1024):
        self.embedder = embedder
        self.dim = dim
        self.text_index = faiss.IndexFlatIP(dim)
        self.formula_index = faiss.IndexFlatIP(dim)
        self.text_metadata = []
        self.formula_metadata = []

    def add_chunk(self, chunk: Dict[str, Any]):
        chunk_id = chunk['chunk_id']
        text_emb = self.embedder.embed_text(chunk.get('search_text', chunk['content']))
        self.text_index.add(text_emb.reshape(1, -1))
        self.text_metadata.append({'chunk_id': chunk_id, 'type': 'text', 'chunk': chunk})

        for f in chunk.get('formulas', []):
            formula_emb = self.embedder.embed_formula(f)
            self.formula_index.add(formula_emb.reshape(1, -1))
            self.formula_metadata.append({
                'chunk_id': chunk_id,
                'type': 'formula',
                'chunk': chunk,
                'formula': f
            })

    def search(self, query: str, top_k: int = 5):
        is_math = self.embedder.is_math_query(query)
        q_emb = self.embedder.embed_text(query).reshape(1, -1)

        text_scores, text_idx = self.text_index.search(q_emb, top_k * 2)
        text_results = []
        for i, score in enumerate(text_scores[0]):
            if score > 0:
                text_results.append({
                    'score': float(score),
                    'metadata': self.text_metadata[text_idx[0][i]]
                })

        formula_scores, formula_idx = self.formula_index.search(q_emb, top_k * 2)
        formula_results = []
        for i, score in enumerate(formula_scores[0]):
            if score > 0:
                formula_results.append({
                    'score': float(score),
                    'metadata': self.formula_metadata[formula_idx[0][i]]
                })

        weight_text = 0.3 if is_math else 0.7
        weight_formula = 0.7 if is_math else 0.3

        for r in text_results:
            r['score'] *= weight_text
        for r in formula_results:
            r['score'] *= weight_formula

        all_results = text_results + formula_results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:top_k]

    def save(self, path: str):
        faiss.write_index(self.text_index, f"{path}_text.faiss")
        faiss.write_index(self.formula_index, f"{path}_formula.faiss")
        with open(f"{path}_metadata.pkl", 'wb') as f:
            pickle.dump({'text': self.text_metadata, 'formula': self.formula_metadata}, f)

    def load(self, path: str):
        self.text_index = faiss.read_index(f"{path}_text.faiss")
        self.formula_index = faiss.read_index(f"{path}_formula.faiss")
        with open(f"{path}_metadata.pkl", 'rb') as f:
            data = pickle.load(f)
            self.text_metadata = data['text']
            self.formula_metadata = data['formula']