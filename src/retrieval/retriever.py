from typing import List, Dict
from src.retrieval.index import DualIndex
from src.retrieval.embedder import Embedder


class Retriever:
    def __init__(self, index_path: str = None):
        self.embedder = Embedder()
        self.index = DualIndex(self.embedder)
        if index_path:
            self.index.load(index_path)

    def index_chunks(self, chunks: List[Dict]):
        for chunk in chunks:
            self.index.add_chunk(chunk)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        results = self.index.search(query, top_k)
        seen = set()
        unique_results = []
        for r in results:
            cid = r['metadata']['chunk_id']
            if cid not in seen:
                seen.add(cid)
                unique_results.append(r['metadata']['chunk'])
        return unique_results[:top_k]

    def save(self, path: str):
        self.index.save(path)