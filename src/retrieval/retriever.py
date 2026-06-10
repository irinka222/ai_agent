import logging
import os
import pickle
import re
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def get_query_math_score(query: str) -> float:
    """
    Оценивает, насколько запрос похож на математический / формульный.
    0.0 — обычный текстовый запрос.
    1.0 — явно формульный запрос.
    """
    score = 0.0
    query_lower = query.lower()

    math_symbols = re.findall(
        r'[=∂∇∑∫±×÷√∞≤≥≠]|\\[a-z]+|_\{|d/d[xt]|\b(div|grad|rot)\b',
        query_lower,
    )
    score += min(len(math_symbols) * 0.25, 0.6)

    formula_terms = [
        'уравнение', 'equation',
        'формула', 'formula',
        'нернста', 'nernst',
        'планка', 'planck',
        'пуассона', 'poisson',
        'навье', 'navier',
        'стокса', 'stokes',
        'коэффициент', 'coefficient',
        'концентрация', 'concentration',
        'поток', 'flux',
        'потенциал', 'potential',
        'плотность тока', 'current density',
        'диссоциация', 'dissociation',
        'рекомбинация', 'recombination',
        'k_d', 'kd',
        'k_r', 'kr',
        'k_w', 'kw',
        'константа', 'constant',
        'вода', 'water',
        'ph',
        'водородный',
        'гидроксил',
        'гидроксида',
        'ион', 'ion',
        'электродиализ', 'electrodialysis',
        'мембрана', 'membrane',
        'опз',
        'область пространственного заряда',
        'space charge',
        'space-charge',
        'дебаев', 'debye',
        'вах',
        'вольт-ампер',
        'current-voltage',
        'overlimiting current',
        'limiting current',
    ]

    term_hits = sum(1 for term in formula_terms if term in query_lower)
    score += min(term_hits * 0.1, 0.4)

    return min(score, 1.0)


def _default_tokenizer(text: str) -> List[str]:
    """
    Простой токенизатор для BM25.
    Сохраняет русские/английские слова, цифры и часть математических символов.
    """
    text = text.lower()
    text = re.sub(r'[^a-zа-яё0-9_+\-*/=∂∇]+', ' ', text)
    return [token for token in text.split() if token]


def _safe_chunk_copy(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """
    Делает безопасную копию чанка и добавляет обязательные поля,
    чтобы агент не падал при отсутствии source/citation/formulas.
    """
    result = dict(chunk)

    result.setdefault("chunk_id", "")
    result.setdefault("content", "")
    result.setdefault("formulas", [])
    result.setdefault("search_text", result.get("content", ""))
    result.setdefault("source", "")
    result.setdefault("section", None)
    result.setdefault("citation", {})
    result.setdefault("context_window", {})

    return result


class Embedder:
    """
    Эмбеддер текста и формул.
    Использует multilingual-e5-large.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        local_files_only: Optional[bool] = None,
    ):
        if local_files_only is None:
            local_files_only = (
                os.environ.get("HF_HUB_OFFLINE") == "1"
                or os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            )

        logger.info(
            "Загрузка embedding-модели: %s, local_files_only=%s",
            model_name,
            local_files_only,
        )

        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            local_files_only=local_files_only,
        )

    def embed_text(self, text: str) -> np.ndarray:
        emb = self.model.encode(text or "", normalize_embeddings=True)
        return np.asarray(emb, dtype="float32")

    def embed_formula(self, formula: Dict[str, Any]) -> np.ndarray:
        semantic = (
            formula.get("semantic_description")
            or formula.get("description")
            or formula.get("context")
            or ""
        )
        latex = formula.get("latex", "")

        text_for_embed = f"{semantic} | formula: {latex}"
        return self.embed_text(text_for_embed)

    def get_math_score(self, query: str) -> float:
        return get_query_math_score(query)


class DualIndex:
    """
    Двойной FAISS-индекс:
    1. text_index — для обычных текстовых чанков.
    2. formula_index — для формул из поля formulas.
    """

    def __init__(self, embedder: Embedder, dim: int = 1024):
        self.embedder = embedder
        self.dim = dim

        self.text_index = faiss.IndexFlatIP(dim)
        self.formula_index = faiss.IndexFlatIP(dim)

        self.text_metadata: List[Dict[str, Any]] = []
        self.formula_metadata: List[Dict[str, Any]] = []

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        chunk = _safe_chunk_copy(chunk)

        chunk_id = chunk.get("chunk_id", "")
        source = chunk.get("source", "")
        section = chunk.get("section", "")
        content = chunk.get("content", "")
        search_text = chunk.get("search_text") or content
        formulas = chunk.get("formulas", []) or []

        text_emb = self.embedder.embed_text(search_text)

        if text_emb.shape[0] != self.dim:
            raise ValueError(
                f"Неверная размерность text embedding: {text_emb.shape[0]}, ожидалось {self.dim}"
            )

        self.text_index.add(text_emb.reshape(1, -1))
        self.text_metadata.append(
            {
                "chunk_id": chunk_id,
                "type": "text",
                "chunk": chunk,
                "source": source,
                "section": section,
            }
        )

        for formula in formulas:
            if not isinstance(formula, dict):
                continue

            latex = formula.get("latex", "")
            if not latex:
                continue

            formula_emb = self.embedder.embed_formula(formula)

            if formula_emb.shape[0] != self.dim:
                raise ValueError(
                    f"Неверная размерность formula embedding: {formula_emb.shape[0]}, ожидалось {self.dim}"
                )

            self.formula_index.add(formula_emb.reshape(1, -1))
            self.formula_metadata.append(
                {
                    "chunk_id": chunk_id,
                    "type": "formula",
                    "chunk": chunk,
                    "formula": formula,
                    "source": source,
                    "section": section,
                }
            )

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.text_index.ntotal == 0:
            return []

        math_score = get_query_math_score(query)

        # Плавное распределение весов:
        # обычный запрос: text=0.7, formula=0.3
        # формульный запрос: text=0.2, formula=0.8
        weight_formula = 0.3 + 0.5 * math_score
        weight_text = 1.0 - weight_formula

        q_emb = self.embedder.embed_text(query).reshape(1, -1)

        all_results: List[Dict[str, Any]] = []

        text_limit = min(max(top_k * 3, top_k), self.text_index.ntotal)
        text_scores, text_indices = self.text_index.search(q_emb, text_limit)

        for rank, raw_score in enumerate(text_scores[0]):
            meta_idx = int(text_indices[0][rank])

            if meta_idx < 0 or meta_idx >= len(self.text_metadata):
                continue

            score = float(raw_score)
            if score <= 0:
                continue

            all_results.append(
                {
                    "score": score * weight_text,
                    "raw_score": score,
                    "rank": rank + 1,
                    "search_type": "dense_text",
                    "math_score": math_score,
                    "metadata": self.text_metadata[meta_idx].copy(),
                }
            )

        if self.formula_index.ntotal > 0:
            formula_limit = min(max(top_k * 3, top_k), self.formula_index.ntotal)
            formula_scores, formula_indices = self.formula_index.search(q_emb, formula_limit)

            for rank, raw_score in enumerate(formula_scores[0]):
                meta_idx = int(formula_indices[0][rank])

                if meta_idx < 0 or meta_idx >= len(self.formula_metadata):
                    continue

                score = float(raw_score)
                if score <= 0:
                    continue

                all_results.append(
                    {
                        "score": score * weight_formula,
                        "raw_score": score,
                        "rank": rank + 1,
                        "search_type": "dense_formula",
                        "math_score": math_score,
                        "metadata": self.formula_metadata[meta_idx].copy(),
                    }
                )

        all_results.sort(key=lambda item: item["score"], reverse=True)
        return all_results[:top_k]

    def save(self, base_path: str) -> None:
        faiss.write_index(self.text_index, f"{base_path}_text.faiss")
        faiss.write_index(self.formula_index, f"{base_path}_formula.faiss")

        with open(f"{base_path}_metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "text": self.text_metadata,
                    "formula": self.formula_metadata,
                    "dim": self.dim,
                },
                f,
            )

    def load(self, base_path: str) -> None:
        self.text_index = faiss.read_index(f"{base_path}_text.faiss")

        formula_path = f"{base_path}_formula.faiss"
        if os.path.exists(formula_path):
            self.formula_index = faiss.read_index(formula_path)
        else:
            logger.warning("Формульный FAISS-индекс не найден: %s", formula_path)
            self.formula_index = faiss.IndexFlatIP(self.dim)

        with open(f"{base_path}_metadata.pkl", "rb") as f:
            data = pickle.load(f)

        self.text_metadata = data.get("text", [])
        self.formula_metadata = data.get("formula", [])

        logger.info(
            "FAISS загружен: text=%s, formula=%s",
            self.text_index.ntotal,
            self.formula_index.ntotal,
        )


class BM25Retriever:
    """
    BM25-ретривер.
    Работает не со строками, а с полными чанками, чтобы не терять chunk_id/source/citation.
    """

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or _default_tokenizer
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []

    def index(self, chunks: List[Dict[str, Any]]) -> None:
        self.documents = [_safe_chunk_copy(chunk) for chunk in chunks]

        tokenized_corpus = []

        for chunk in self.documents:
            content = chunk.get("content", "")
            search_text = chunk.get("search_text", "")

            formulas = " ".join(
                formula.get("latex", "")
                for formula in chunk.get("formulas", []) or []
                if isinstance(formula, dict)
            )

            combined = f"{content} {search_text} {formulas}"
            tokenized_corpus.append(self.tokenizer(combined))

        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        with_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.documents:
            return []

        tokenized_query = self.tokenizer(query)
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for rank, idx in enumerate(top_indices):
            score = float(scores[idx])

            if score <= 0:
                continue

            chunk = _safe_chunk_copy(self.documents[int(idx)])

            if with_scores:
                chunk["_score"] = score
                chunk["_rank"] = rank + 1
                chunk["_search_type"] = "bm25"

            results.append(chunk)

        return results

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "documents": self.documents,
                    "tokenizer": self.tokenizer,
                },
                f,
            )

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.documents = data.get("documents", [])
        self.tokenizer = data.get("tokenizer", _default_tokenizer)

        self.index(self.documents)

        logger.info("BM25 загружен и перестроен: documents=%s", len(self.documents))


class Retriever:
    """
    Основной ретривер:
    - dense FAISS search;
    - formula FAISS search;
    - BM25 search;
    - RRF hybrid search;
    - weighted hybrid search;
    - save/load master_index.
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        embedding_model: str = "intfloat/multilingual-e5-large",
        dim: int = 1024,
        local_files_only: Optional[bool] = None,
    ):
        self.embedder = Embedder(
            model_name=embedding_model,
            local_files_only=local_files_only,
        )

        self.index = DualIndex(self.embedder, dim=dim)
        self.bm25 = BM25Retriever()
        self._id_to_chunk: Dict[str, Dict[str, Any]] = {}

        if index_path and os.path.exists(f"{index_path}_text.faiss"):
            self.load(index_path)

    def index_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        self._id_to_chunk = {}

        for chunk in chunks:
            chunk = _safe_chunk_copy(chunk)
            self.index.add_chunk(chunk)

            chunk_id = chunk.get("chunk_id")
            if chunk_id:
                self._id_to_chunk[chunk_id] = chunk

        self.bm25.index(chunks)

        logger.info(
            "Проиндексировано %s чанков: FAISS text=%s, FAISS formula=%s, BM25=%s.",
            len(chunks),
            self.index.text_index.ntotal,
            self.index.formula_index.ntotal,
            len(self.bm25.documents),
        )

    def dense_search(
        self,
        query: str,
        top_k: int = 10,
        with_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        raw_results = self.index.search(query, top_k=top_k)

        seen = set()
        results = []

        for item in raw_results:
            metadata = item.get("metadata", {})
            chunk = _safe_chunk_copy(metadata.get("chunk", {}))

            chunk_id = chunk.get("chunk_id") or metadata.get("chunk_id")
            if not chunk_id or chunk_id in seen:
                continue

            seen.add(chunk_id)

            chunk["chunk_id"] = chunk_id
            chunk["source"] = metadata.get("source", chunk.get("source", ""))
            chunk["section"] = metadata.get("section", chunk.get("section"))

            if with_scores:
                chunk["_score"] = item.get("score")
                chunk["_raw_score"] = item.get("raw_score")
                chunk["_rank"] = item.get("rank")
                chunk["_search_type"] = item.get("search_type")
                chunk["_math_score"] = item.get("math_score")

            results.append(chunk)

        return results[:top_k]

    def bm25_search(
        self,
        query: str,
        top_k: int = 10,
        with_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        return self.bm25.retrieve(query, top_k=top_k, with_scores=with_scores)

    def hybrid_search_rrf(
        self,
        query: str,
        top_k: int = 10,
        k_rrf: int = 60,
        with_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        dense_results = self.dense_search(
            query,
            top_k=top_k * 3,
            with_scores=True,
        )
        bm25_results = self.bm25_search(
            query,
            top_k=top_k * 3,
            with_scores=True,
        )

        scores: Dict[str, float] = {}
        debug: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_results):
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                continue

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)

            debug.setdefault(chunk_id, {})["dense_rank"] = rank + 1
            debug.setdefault(chunk_id, {})["dense_score"] = doc.get("_score")
            debug.setdefault(chunk_id, {})["dense_type"] = doc.get("_search_type")

        for rank, doc in enumerate(bm25_results):
            chunk_id = doc.get("chunk_id")
            if not chunk_id:
                continue

            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k_rrf + rank + 1)

            debug.setdefault(chunk_id, {})["bm25_rank"] = rank + 1
            debug.setdefault(chunk_id, {})["bm25_score"] = doc.get("_score")

        sorted_ids = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []

        for chunk_id, score in sorted_ids:
            chunk = self._id_to_chunk.get(chunk_id)

            if not chunk:
                continue

            chunk = _safe_chunk_copy(chunk)

            if with_scores:
                chunk["_score"] = score
                chunk["_search_type"] = "hybrid_rrf"
                chunk["_math_score"] = get_query_math_score(query)
                chunk["_debug"] = debug.get(chunk_id, {})

            results.append(chunk)

        return results

    def hybrid_search_weighted(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.6,
        with_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        dense_results = self.dense_search(
            query,
            top_k=top_k * 3,
            with_scores=True,
        )
        bm25_results = self.bm25_search(
            query,
            top_k=top_k * 3,
            with_scores=True,
        )

        dense_ranks = {
            doc["chunk_id"]: rank
            for rank, doc in enumerate(dense_results)
            if doc.get("chunk_id")
        }

        bm25_ranks = {
            doc["chunk_id"]: rank
            for rank, doc in enumerate(bm25_results)
            if doc.get("chunk_id")
        }

        all_ids = set(dense_ranks) | set(bm25_ranks)
        combined: Dict[str, float] = {}

        max_rank = top_k * 3 + 1

        for chunk_id in all_ids:
            dense_rank = dense_ranks.get(chunk_id, max_rank)
            bm25_rank = bm25_ranks.get(chunk_id, max_rank)

            dense_norm = 1.0 / (dense_rank + 1)
            bm25_norm = 1.0 / (bm25_rank + 1)

            combined[chunk_id] = alpha * dense_norm + (1.0 - alpha) * bm25_norm

        sorted_ids = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results = []

        for chunk_id, score in sorted_ids:
            chunk = self._id_to_chunk.get(chunk_id)

            if not chunk:
                continue

            chunk = _safe_chunk_copy(chunk)

            if with_scores:
                chunk["_score"] = score
                chunk["_search_type"] = "hybrid_weighted"
                chunk["_math_score"] = get_query_math_score(query)
                chunk["_debug"] = {
                    "dense_rank": dense_ranks.get(chunk_id),
                    "bm25_rank": bm25_ranks.get(chunk_id),
                    "alpha": alpha,
                }

            results.append(chunk)

        return results

    def search(
        self,
        query: str,
        top_k: int = 10,
        mode: str = "rrf",
        alpha: float = 0.6,
        with_scores: bool = False,
    ) -> List[Dict[str, Any]]:
        if mode == "dense":
            return self.dense_search(
                query,
                top_k=top_k,
                with_scores=with_scores,
            )

        if mode == "bm25":
            return self.bm25_search(
                query,
                top_k=top_k,
                with_scores=with_scores,
            )

        if mode == "weighted":
            return self.hybrid_search_weighted(
                query,
                top_k=top_k,
                alpha=alpha,
                with_scores=with_scores,
            )

        return self.hybrid_search_rrf(
            query,
            top_k=top_k,
            with_scores=with_scores,
        )

    def save(self, base_path: str) -> None:
        self.index.save(base_path)
        self.bm25.save(f"{base_path}_bm25.pkl")

        with open(f"{base_path}_id_map.pkl", "wb") as f:
            pickle.dump(self._id_to_chunk, f)

        logger.info("Retriever сохранён как %s_*", base_path)

    def load(self, base_path: str) -> None:
        self.index.load(base_path)

        bm25_path = f"{base_path}_bm25.pkl"
        if os.path.exists(bm25_path):
            self.bm25.load(bm25_path)
        else:
            logger.warning("BM25 индекс не найден: %s", bm25_path)

        id_map_path = f"{base_path}_id_map.pkl"
        if os.path.exists(id_map_path):
            with open(id_map_path, "rb") as f:
                self._id_to_chunk = pickle.load(f)
        else:
            logger.warning("id_map не найден: %s", id_map_path)
            self._id_to_chunk = {}

        logger.info(
            "Retriever загружен из %s: chunks=%s",
            base_path,
            len(self._id_to_chunk),
        )