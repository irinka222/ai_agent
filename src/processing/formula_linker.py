# src/processing/formula_linker.py
# -*- coding: utf-8 -*-

import re
import logging
from typing import List, Dict, Any, Optional, Set

logger = logging.getLogger(__name__)


STOP_WORDS = {
    "для", "при", "это", "как", "или", "что", "где", "также", "который",
    "equation", "formula", "model", "where", "with", "from", "this",
    "формула", "уравнение", "модель", "глава", "стр", "страница",
}


def normalize_text(text: Any) -> str:
    if text is None:
        return ""

    text = str(text).lower()
    text = text.replace("ё", "е")

    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "×": "x",
        "·": " ",
        "\\": " ",
        "{": " ",
        "}": " ",
        "_": " ",
        "^": " ",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_page_from_context(context: str) -> Optional[int]:
    if not context:
        return None

    match = re.search(r"(стр\.?|страница|page)\s*\.?\s*(\d+)", context.lower())
    if match:
        return int(match.group(2))

    return None


def get_chunk_pages(chunk: Dict[str, Any]) -> Set[int]:
    pages = set()

    if chunk.get("page") is not None:
        try:
            pages.add(int(chunk["page"]))
        except Exception:
            pass

    for p in chunk.get("pages", []) or []:
        try:
            pages.add(int(p))
        except Exception:
            pass

    citation = chunk.get("citation", {}) or {}
    for p in citation.get("pages", []) or []:
        try:
            pages.add(int(p))
        except Exception:
            pass

    return pages


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = re.findall(r"[a-zа-я0-9]+", text, flags=re.IGNORECASE)

    return [
        t for t in tokens
        if len(t) >= 3 and t not in STOP_WORDS
    ]


def formula_keywords(formula: Dict[str, Any]) -> Set[str]:
    parts = []

    for field in ["id", "number", "latex", "description", "semantic_description", "context", "section"]:
        value = formula.get(field)
        if value:
            parts.append(str(value))

    variables = formula.get("variables", {})
    if isinstance(variables, dict):
        parts.extend(variables.keys())
        parts.extend(variables.values())
    elif isinstance(variables, list):
        parts.extend(map(str, variables))

    return set(tokenize(" ".join(parts)))


def formula_to_linked_object(formula: Dict[str, Any], score: float, reasons: List[str]) -> Dict[str, Any]:
    return {
        "id": formula.get("id"),
        "number": formula.get("number"),
        "latex": formula.get("latex"),
        "description": formula.get("description"),
        "semantic_description": formula.get("semantic_description", ""),
        "variables": formula.get("variables", {}),
        "context": formula.get("context"),
        "source": formula.get("source"),
        "page": formula.get("page") or extract_page_from_context(formula.get("context", "")),
        "confidence": formula.get("confidence", 1.0),
        "link_score": round(score, 3),
        "link_reasons": reasons,
        "type": "curated_formula",
    }


def score_formula_for_chunk(
    formula: Dict[str, Any],
    formula_kw: Set[str],
    chunk: Dict[str, Any],
) -> tuple[float, List[str]]:
    score = 0.0
    reasons = []

    chunk_text = normalize_text(
        " ".join([
            chunk.get("content", ""),
            chunk.get("section", ""),
            str(chunk.get("citation", {})),
            chunk.get("search_text", ""),
        ])
    )

    chunk_tokens = set(tokenize(chunk_text))

    common = formula_kw & chunk_tokens
    if common:
        score += min(len(common) * 0.8, 8.0)
        reasons.append(f"keyword_overlap:{len(common)}")

    number = formula.get("number")
    if number and str(number).lower() in chunk_text:
        score += 5.0
        reasons.append("formula_number")

    formula_page = formula.get("page") or extract_page_from_context(formula.get("context", ""))
    chunk_pages = get_chunk_pages(chunk)

    if formula_page and chunk_pages:
        if int(formula_page) in chunk_pages:
            score += 6.0
            reasons.append("same_page")
        elif any(abs(int(formula_page) - p) <= 1 for p in chunk_pages):
            score += 2.5
            reasons.append("near_page")

    formula_context = normalize_text(formula.get("context", ""))
    chunk_section = normalize_text(chunk.get("section", ""))

    if formula_context and chunk_section:
        context_tokens = set(tokenize(formula_context))
        section_tokens = set(tokenize(chunk_section))
        section_overlap = context_tokens & section_tokens

        if section_overlap:
            score += min(len(section_overlap) * 1.0, 4.0)
            reasons.append(f"section_overlap:{len(section_overlap)}")

    description = normalize_text(formula.get("description", ""))
    if description:
        desc_tokens = set(tokenize(description))
        desc_overlap = desc_tokens & chunk_tokens
        if desc_overlap:
            score += min(len(desc_overlap) * 0.7, 5.0)
            reasons.append(f"description_overlap:{len(desc_overlap)}")

    latex = normalize_text(formula.get("latex", ""))
    if latex:
        latex_tokens = set(tokenize(latex))
        latex_overlap = latex_tokens & chunk_tokens
        if latex_overlap:
            score += min(len(latex_overlap) * 0.6, 3.0)
            reasons.append(f"latex_overlap:{len(latex_overlap)}")

    return score, reasons


def link_formulas_to_chunks(
    chunks: List[Dict[str, Any]],
    formulas: List[Dict[str, Any]],
    min_score: float = 4.0,
    max_formulas_per_chunk: int = 5,
) -> List[Dict[str, Any]]:
    """
    Связывает чанки с curated-базой формул.

    Вход:
    - chunks: чанки после chunker.py
    - formulas: список формул из JSON-базы

    Выход:
    - каждый chunk получает:
        chunk["formulas"]
        chunk["search_text"]
        chunk["metadata"]["has_formula"]
        chunk["metadata"]["formula_count"]
    """
    if not chunks:
        return []

    if not formulas:
        logger.warning("No formulas provided for linking")
        return chunks

    prepared = [
        {
            "formula": formula,
            "keywords": formula_keywords(formula),
        }
        for formula in formulas
        if formula.get("latex")
    ]

    linked_total = 0
    result = []

    for chunk in chunks:
        new_chunk = chunk.copy()
        existing_formulas = list(new_chunk.get("formulas", []) or [])

        candidates = []

        for item in prepared:
            formula = item["formula"]
            keywords = item["keywords"]

            score, reasons = score_formula_for_chunk(
                formula=formula,
                formula_kw=keywords,
                chunk=new_chunk,
            )

            if score >= min_score:
                candidates.append(
                    formula_to_linked_object(formula, score, reasons)
                )

        candidates.sort(key=lambda x: x.get("link_score", 0), reverse=True)

        merged = merge_formula_lists(
            existing_formulas,
            candidates[:max_formulas_per_chunk],
        )

        new_chunk["formulas"] = merged
        new_chunk["search_text"] = build_search_text(new_chunk)
        new_chunk.setdefault("metadata", {})
        new_chunk["metadata"]["has_formula"] = bool(merged)
        new_chunk["metadata"]["formula_count"] = len(merged)

        linked_total += len(merged)
        result.append(new_chunk)

    logger.info(
        "Formula linking completed: %d formulas linked to %d chunks",
        linked_total,
        len(result),
    )

    return result


def merge_formula_lists(
    existing: List[Dict[str, Any]],
    linked: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = []
    seen = set()

    for formula in existing + linked:
        key = (
            formula.get("id")
            or formula.get("latex")
            or formula.get("description")
        )

        if not key:
            continue

        normalized_key = normalize_text(key)

        if normalized_key in seen:
            continue

        seen.add(normalized_key)
        merged.append(formula)

    return merged


def build_search_text(chunk: Dict[str, Any]) -> str:
    parts = [chunk.get("content", "")]

    for formula in chunk.get("formulas", []) or []:
        parts.append(formula.get("description", ""))
        parts.append(formula.get("semantic_description", ""))
        parts.append(formula.get("latex", ""))

        variables = formula.get("variables", {})
        if isinstance(variables, dict):
            parts.extend(variables.keys())
            parts.extend(variables.values())
        elif isinstance(variables, list):
            parts.extend(map(str, variables))

    return "\n".join(str(p) for p in parts if p).strip()