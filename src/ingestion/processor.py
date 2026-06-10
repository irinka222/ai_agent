import os
import glob
import pickle
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.ingestion.pdf_parser import unified_parse_pdf
from src.processing.cleaner import clean_blocks
from src.processing.normalizer import normalize_blocks
from src.processing.chunker import chunk_blocks
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache/parsed_pdfs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_pdf_hash(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def process_pdf_for_indexing(pdf_path: str, retriever: Retriever, use_cache: bool = True) -> bool:
    pdf_path = str(pdf_path)
    file_hash = get_pdf_hash(pdf_path)
    cache_file = CACHE_DIR / f"{file_hash}.pkl"

    if use_cache and cache_file.exists():
        logger.info(f"Загрузка чанков из кэша: {pdf_path}")
        with open(cache_file, "rb") as f:
            chunks = pickle.load(f)
    else:
        blocks, metadata = unified_parse_pdf(pdf_path)
        if not blocks:
            logger.error(f"Не удалось извлечь контент из {pdf_path}")
            return False

        cleaned = clean_blocks(blocks)
        normalized = normalize_blocks(cleaned)
        paper_id = metadata.get('source', Path(pdf_path).stem)
        authors = metadata.get('authors', [])
        title = metadata.get('title', paper_id)
        chunks = chunk_blocks(
            blocks=normalized,
            paper_id=paper_id,
            authors=authors,
            title=title,
        )
        if use_cache:
            with open(cache_file, "wb") as f:
                pickle.dump(chunks, f)
            logger.info(f"Кэш сохранён: {cache_file}")

    if not chunks:
        logger.warning(f"Нет чанков для {pdf_path}")
        return False

    retriever.index_chunks(chunks)
    logger.info(f"Проиндексировано {len(chunks)} чанков из {pdf_path}")
    return True


def build_index_from_directory(input_dir: str, index_name: str = "master_index", use_cache: bool = True) -> Retriever:
    pdf_files = glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True)
    logger.info(f"Найдено {len(pdf_files)} PDF файлов")

    retriever = Retriever()
    for pdf_path in pdf_files:
        try:
            process_pdf_for_indexing(pdf_path, retriever, use_cache=use_cache)
        except Exception as e:
            logger.exception(f"Ошибка при обработке {pdf_path}: {e}")

    retriever.save(index_name)
    logger.info(f"Индекс сохранён в {index_name}_*.faiss, {index_name}_metadata.pkl и {index_name}_bm25.pkl")
    return retriever