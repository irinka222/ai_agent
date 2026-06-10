#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple

from src.processing.formula_extractor import (
    clean_formula,
    is_valid_formula,
)

logger = logging.getLogger(__name__)

MIN_CHUNK_SIZE = 300
MAX_CHUNK_SIZE = 1500
OVERLAP_SIZE = 150

FORMULA_BLOCK_TYPES = {"formula", "formula_candidate", "equation_block", "formula_fragment"}


def make_chunk_id(
    paper_id: str,
    content: str,
    section: Optional[str] = None,
    pages: Optional[List[int]] = None,
) -> str:
    raw = "|".join([
        paper_id or "unknown",
        section or "",
        ",".join(map(str, pages or [])),
        (content or "")[:500],
    ])

    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{paper_id}_{digest}"


def is_garbage_formula(text: str) -> bool:
    if not text:
        return True

    s = str(text).strip()

    if len(s) < 3:
        return True

    if re.match(r"^\d+$", s):
        return True

    garbage_patterns = [
        r"^CEM_",
        r"^AEM_",
        r"^DL_",
        r"^PS,$",
        r"^PD$",
        r"^NaCl$",
        r"^KCl$",
        r"^CatD$",
        r"^NCatD$",
        r"^RD$",
        r"^_$",
        r"^[.,;:]+$",
    ]

    return any(re.match(pattern, s) for pattern in garbage_patterns)


def normalize_page_list(pages: List[Any]) -> List[int]:
    result = []

    for page in pages:
        if page is None:
            continue

        try:
            p = int(page)
        except Exception:
            continue

        if p not in result:
            result.append(p)

    return result


def formula_key(formula: Dict[str, Any]) -> str:
    return str(
        formula.get("id")
        or formula.get("latex")
        or formula.get("description")
        or formula.get("raw")
        or ""
    ).strip()


def deduplicate_formulas(formulas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    seen = set()

    for formula in formulas or []:
        if not isinstance(formula, dict):
            continue

        key = re.sub(r"\s+", "", formula_key(formula).lower())

        if not key:
            continue

        if key in seen:
            continue

        seen.add(key)
        result.append(formula)

    return result


def normalize_formula_object(
    formula: Dict[str, Any],
    fallback_latex: str = "",
    fallback_page: Optional[int] = None,
    fallback_source: Optional[str] = None,
    fallback_description: str = "",
    fallback_semantic: str = "",
) -> Optional[Dict[str, Any]]:
    if not isinstance(formula, dict):
        return None

    latex = (
        formula.get("latex")
        or formula.get("normalized")
        or formula.get("content")
        or fallback_latex
        or ""
    )

    latex = str(latex).strip()

    if latex and not is_garbage_formula(latex):
        latex = clean_formula(latex) if is_valid_formula(latex) else latex
    else:
        latex = ""

    if not latex:
        return None

    return {
        "id": formula.get("id") or formula.get("formula_id"),
        "number": formula.get("number") or formula.get("label"),
        "latex": latex,
        "raw": formula.get("raw") or fallback_latex or latex,
        "description": (
            formula.get("description")
            or formula.get("semantic_description")
            or fallback_description
            or fallback_semantic
            or ""
        ),
        "semantic_description": (
            formula.get("semantic_description")
            or formula.get("description")
            or fallback_semantic
            or fallback_description
            or ""
        ),
        "variables": formula.get("variables", {}),
        "context": formula.get("context") or formula.get("context_before"),
        "source": formula.get("source") or fallback_source,
        "page": formula.get("page") or fallback_page,
        "confidence": formula.get("confidence", 0.5),
        "type": formula.get("type", "formula_candidate"),
        "extraction_method": formula.get("extraction_method"),
        "link_score": formula.get("link_score"),
    }


def build_search_text(content: str, formulas: List[Dict[str, Any]]) -> str:
    parts = [content or ""]

    for formula in formulas or []:
        parts.append(formula.get("description", ""))
        parts.append(formula.get("semantic_description", ""))
        parts.append(formula.get("latex", ""))

        variables = formula.get("variables", {})
        if isinstance(variables, dict):
            parts.extend(str(k) for k in variables.keys())
            parts.extend(str(v) for v in variables.values())
        elif isinstance(variables, list):
            parts.extend(str(v) for v in variables)

    return "\n".join(str(p) for p in parts if p).strip()


def create_chunk(
    content: str,
    section: Optional[str],
    pages: List[int],
    paper_id: str,
    authors: List[str],
    title: str,
    year: Optional[str] = None,
    formulas: Optional[List[Dict[str, Any]]] = None,
    source: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    content = (content or "").strip()
    formulas = deduplicate_formulas(formulas or [])
    pages = normalize_page_list(pages)

    if len(content) < MIN_CHUNK_SIZE and not formulas:
        return None

    search_text = build_search_text(content, formulas)

    citation = {
        "authors": authors or [],
        "paper_id": paper_id,
        "title": title,
        "year": year,
        "section": section if section else "unknown",
        "pages": pages[:2] if len(pages) > 2 else pages,
    }

    return {
        "chunk_id": make_chunk_id(
            paper_id=paper_id,
            content=content,
            section=section,
            pages=pages,
    ),
        "content": content,
        "formulas": formulas,
        "search_text": search_text,
        "citation": citation,
        "type": "chunk",
        "source": source or paper_id,
        "section": section,
        "pages": pages,
        "metadata": {
            "has_formula": bool(formulas),
            "formula_count": len(formulas),
            "is_formula_only": bool(formulas) and len(content) < 200,
        },
    }


def equation_block_to_text_and_formulas(block: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    latex = str(block.get("latex") or block.get("normalized") or "").strip()
    context_before = str(block.get("context_before") or "").strip()
    description = str(block.get("description") or "").strip()
    semantic_description = str(block.get("semantic_description") or "").strip()
    label = str(block.get("label") or "").strip()
    page = block.get("page")
    source = block.get("source")

    normalized_latex = ""

    if latex and not is_garbage_formula(latex):
        normalized_latex = clean_formula(latex) if is_valid_formula(latex) else latex

    formulas = []

    existing_formulas = block.get("formulas") or []
    for formula in existing_formulas:
        normalized_formula = normalize_formula_object(
            formula=formula,
            fallback_latex=normalized_latex,
            fallback_page=page,
            fallback_source=source,
            fallback_description=description,
            fallback_semantic=semantic_description or context_before,
        )
        if normalized_formula:
            formulas.append(normalized_formula)

    if not formulas and normalized_latex:
        formulas.append({
            "id": block.get("id") or block.get("formula_id"),
            "number": block.get("number") or label,
            "latex": normalized_latex,
            "raw": block.get("raw") or latex,
            "description": description or semantic_description or context_before,
            "semantic_description": semantic_description or description or context_before,
            "variables": block.get("variables", {}),
            "context": context_before,
            "source": source,
            "page": page,
            "confidence": block.get("confidence", 0.5),
            "type": block.get("type", "formula_candidate"),
            "extraction_method": block.get("extraction_method"),
        })

    formulas = deduplicate_formulas(formulas)

    parts = []

    if context_before:
        parts.append(context_before)

    if normalized_latex:
        formula_line = f"Формула: {normalized_latex}"
        if label:
            formula_line += f" {label}"
        parts.append(formula_line)

    if description:
        parts.append(description)

    if semantic_description and semantic_description != description:
        parts.append(f"Смысл формулы: {semantic_description}")

    content = "\n".join(parts).strip()

    return content, formulas


def formula_block_to_text_and_formulas(block: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    latex = str(block.get("latex") or block.get("normalized") or block.get("content") or "").strip()

    if is_garbage_formula(latex) or not is_valid_formula(latex):
        return "", []

    latex = clean_formula(latex)

    formula = normalize_formula_object(
        formula=block,
        fallback_latex=latex,
        fallback_page=block.get("page"),
        fallback_source=block.get("source"),
        fallback_description=block.get("description", ""),
        fallback_semantic=block.get("semantic_description", ""),
    )

    if not formula:
        return "", []

    content = f"Формула: {formula['latex']}"
    if formula.get("number"):
        content += f" {formula['number']}"

    if formula.get("description"):
        content += f"\n{formula['description']}"

    return content, [formula]


def split_large_chunk(
    content: str,
    section: Optional[str],
    pages: List[int],
    paper_id: str,
    authors: List[str],
    title: str,
    year: Optional[str] = None,
    formulas: Optional[List[Dict[str, Any]]] = None,
    source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    chunks = []
    formulas = formulas or []

    sentences = re.split(r"(?<=[.!?])\s+", content)
    current_chunk = ""
    current_pages = pages.copy()

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > MAX_CHUNK_SIZE and current_chunk:
            chunk = create_chunk(
                current_chunk,
                section,
                current_pages,
                paper_id,
                authors,
                title,
                year,
                formulas=formulas,
                source=source,
            )
            if chunk:
                chunks.append(chunk)

            overlap_start = max(0, len(current_chunk) - OVERLAP_SIZE)
            current_chunk = current_chunk[overlap_start:] + " " + sentence
        else:
            current_chunk = f"{current_chunk} {sentence}".strip()

    if current_chunk:
        chunk = create_chunk(
            current_chunk,
            section,
            current_pages,
            paper_id,
            authors,
            title,
            year,
            formulas=formulas,
            source=source,
        )
        if chunk:
            chunks.append(chunk)

    return chunks


def flush_current_chunk(
    chunks: List[Dict[str, Any]],
    current_chunk: str,
    current_formulas: List[Dict[str, Any]],
    current_section: Optional[str],
    current_pages: List[int],
    paper_id: str,
    authors: List[str],
    title: str,
    year: Optional[str],
    source: str,
) -> None:
    if not current_chunk and not current_formulas:
        return

    if len(current_chunk) > MAX_CHUNK_SIZE:
        chunks.extend(
            split_large_chunk(
                current_chunk,
                current_section,
                current_pages,
                paper_id,
                authors,
                title,
                year,
                formulas=current_formulas,
                source=source,
            )
        )
    else:
        chunk = create_chunk(
            current_chunk,
            current_section,
            current_pages,
            paper_id,
            authors,
            title,
            year,
            formulas=current_formulas,
            source=source,
        )
        if chunk:
            chunks.append(chunk)


def should_skip_references(block_type: str, content: str) -> bool:
    if block_type != "section":
        return False

    content_lower = (content or "").lower().strip()

    return bool(
        re.search(
            r"^(список литературы|библиографический список|references|литература)$",
            content_lower,
        )
    )
def add_context_windows(
    chunks: List[Dict[str, Any]],
    window_chars: int = 500,
) -> List[Dict[str, Any]]:
    for i, chunk in enumerate(chunks):
        prev_content = chunks[i - 1].get("content", "") if i > 0 else ""
        next_content = chunks[i + 1].get("content", "") if i + 1 < len(chunks) else ""

        chunk["context_window"] = {
            "prev": prev_content[:window_chars],
            "next": next_content[:window_chars],
        }

    return chunks

def chunk_blocks(
    blocks: List[Dict[str, Any]],
    paper_id: str,
    authors: List[str],
    title: str,
    year: Optional[str] = None,
    include_references: bool = False,
) -> List[Dict[str, Any]]:
    chunks = []

    current_chunk = ""
    current_formulas: List[Dict[str, Any]] = []
    current_section = None
    current_pages: List[int] = []

    source = paper_id

    for block in blocks:
        block_type = block.get("type", "paragraph")
        block_page = block.get("page", 1)
        block_source = block.get("source") or paper_id
        raw_content = str(block.get("content") or "")

        source = block_source or source

        if not include_references and should_skip_references(block_type, raw_content):
            logger.info("Skipping references section")
            break

        if block_type == "section":
            flush_current_chunk(
                chunks,
                current_chunk,
                current_formulas,
                current_section,
                current_pages,
                paper_id,
                authors,
                title,
                year,
                source,
            )

            current_chunk = ""
            current_formulas = []
            current_pages = []
            current_section = raw_content[:180]
            continue

        if block_type == "equation_block":
            block_content, block_formulas = equation_block_to_text_and_formulas(block)

        elif block_type in {"formula", "formula_candidate"}:
            block_content, block_formulas = formula_block_to_text_and_formulas(block)

        elif block_type == "formula_fragment":
            continue

        elif block_type == "paragraph":
            block_content = raw_content.strip()
            block_formulas = []

            for formula in block.get("formulas", []) or []:
                normalized_formula = normalize_formula_object(
                    formula=formula,
                    fallback_page=block_page,
                    fallback_source=block_source,
                )
                if normalized_formula:
                    block_formulas.append(normalized_formula)

        else:
            continue

        if not block_content and not block_formulas:
            continue

        test_len = len(current_chunk) + len(block_content)

        if test_len > MAX_CHUNK_SIZE and current_chunk:
            flush_current_chunk(
                chunks,
                current_chunk,
                current_formulas,
                current_section,
                current_pages,
                paper_id,
                authors,
                title,
                year,
                source,
            )

            current_chunk = block_content
            current_formulas = block_formulas.copy()
            current_pages = normalize_page_list([block_page])
        else:
            if current_chunk and block_content:
                current_chunk += "\n\n" + block_content
            elif block_content:
                current_chunk = block_content

            current_formulas.extend(block_formulas)
            current_formulas = deduplicate_formulas(current_formulas)

            if block_page is not None:
                try:
                    p = int(block_page)
                    if p not in current_pages:
                        current_pages.append(p)
                except Exception:
                    pass

    flush_current_chunk(
        chunks,
        current_chunk,
        current_formulas,
        current_section,
        current_pages,
        paper_id,
        authors,
        title,
        year,
        source,
    )

    chunks = add_context_windows(chunks)

    logger.info("Created %d chunks from %d blocks", len(chunks), len(blocks))
    return chunks