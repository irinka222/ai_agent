import json
import glob
import logging
from pathlib import Path
from typing import List, Dict, Any

from src.processing.chunker import (
    build_search_text,
    make_chunk_id,
    add_context_windows,
)
from src.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)


def load_chunks_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        if isinstance(data.get("chunks"), list):
            return data["chunks"]
        if isinstance(data.get("data"), list):
            return data["data"]

    raise ValueError(f"Unsupported chunks JSON format: {path}")


def normalize_ready_chunk(raw: Dict[str, Any], source_name: str, index: int) -> Dict[str, Any]:
    content = str(raw.get("content") or "").strip()
    citation = raw.get("citation", {}) or {}
    section = raw.get("section")
    formulas = raw.get("formulas", []) or []

    paper_id = (
        raw.get("source")
        or citation.get("paper_id")
        or citation.get("title")
        or source_name
    )

    pages = raw.get("pages") or []
    if not pages and citation.get("pages"):
        pages = citation.get("pages")

    chunk_id = raw.get("chunk_id") or make_chunk_id(
        paper_id=paper_id,
        content=content,
        section=section,
        pages=pages,
    )

    search_text = raw.get("search_text") or build_search_text(content, formulas)

    return {
        "chunk_id": chunk_id,
        "content": content,
        "formulas": formulas,
        "search_text": search_text,
        "citation": citation,
        "type": "chunk",
        "source": paper_id,
        "section": section,
        "pages": pages,
        "metadata": {
            **(raw.get("metadata", {}) or {}),
            "has_formula": bool(formulas),
            "formula_count": len(formulas),
            "is_formula_only": bool(formulas) and len(content) < 200,
            "manual_or_prepared": True,
        },
    }


def load_prepared_chunks_from_directory(input_dir: str) -> List[Dict[str, Any]]:
    paths = glob.glob(str(Path(input_dir) / "*_chunks.json"))
    all_chunks: List[Dict[str, Any]] = []

    for path in paths:
        source_name = Path(path).stem.replace("_chunks", "")

        try:
            raw_chunks = load_chunks_json(path)
        except Exception as e:
            logger.warning("Skip %s: %s", path, e)
            continue

        normalized = [
            normalize_ready_chunk(raw, source_name, i)
            for i, raw in enumerate(raw_chunks, 1)
            if raw.get("content")
        ]

        normalized = add_context_windows(normalized)
        all_chunks.extend(normalized)

        logger.info("Loaded %d chunks from %s", len(normalized), path)

    return all_chunks


def build_index_from_ready_chunks(
    input_dir: str = "data/processed_chunks",
    index_name: str = "master_index",
) -> Retriever:
    chunks = load_prepared_chunks_from_directory(input_dir)

    if not chunks:
        raise RuntimeError(f"No chunks found in {input_dir}")

    retriever = Retriever()
    retriever.index_chunks(chunks)
    retriever.save(index_name)

    logger.info("Indexed %d prepared chunks", len(chunks))
    logger.info("Index saved as %s_*", index_name)

    return retriever