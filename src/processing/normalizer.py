# src/processing/normalizer.py
# -*- coding: utf-8 -*-

from typing import List, Dict, Any, Optional


FORMULA_TYPES = {"formula", "formula_candidate", "formula_fragment"}


def normalize_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not blocks:
        return []

    blocks = merge_equation_blocks(blocks)
    blocks = remove_empty_blocks(blocks)
    blocks = propagate_sections(blocks)

    return blocks


def merge_equation_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    i = 0

    while i < len(blocks):
        block = blocks[i]
        block_type = block.get("type", "paragraph")

        if block_type == "equation_block":
            result.append(normalize_equation_block(block))
            i += 1
            continue

        if (
            i + 2 < len(blocks)
            and blocks[i].get("type") == "paragraph"
            and blocks[i + 1].get("type") in FORMULA_TYPES
            and blocks[i + 2].get("type") == "formula_label"
        ):
            result.append(
                build_equation_block(
                    paragraph=blocks[i],
                    formula=blocks[i + 1],
                    label_block=blocks[i + 2],
                    description_block=blocks[i + 3] if i + 3 < len(blocks) and blocks[i + 3].get("type") == "paragraph" else None,
                )
            )
            i += 4 if i + 3 < len(blocks) and blocks[i + 3].get("type") == "paragraph" else 3
            continue

        if (
            i + 1 < len(blocks)
            and blocks[i].get("type") == "paragraph"
            and blocks[i + 1].get("type") in FORMULA_TYPES
        ):
            result.append(
                build_equation_block(
                    paragraph=blocks[i],
                    formula=blocks[i + 1],
                    label_block=None,
                    description_block=None,
                )
            )
            i += 2
            continue

        if block_type in FORMULA_TYPES:
            result.append(
                build_equation_block(
                    paragraph=None,
                    formula=block,
                    label_block=None,
                    description_block=None,
                )
            )
            i += 1
            continue

        result.append(block)
        i += 1

    return result


def build_equation_block(
    paragraph: Optional[Dict[str, Any]],
    formula: Dict[str, Any],
    label_block: Optional[Dict[str, Any]] = None,
    description_block: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    latex = formula.get("latex") or formula.get("normalized") or formula.get("content", "")
    context_before = paragraph.get("content", "") if paragraph else ""
    description = description_block.get("content", "") if description_block else ""

    source = (
        formula.get("source")
        or (paragraph or {}).get("source")
        or (description_block or {}).get("source")
    )

    paper_id = (
        formula.get("paper_id")
        or (paragraph or {}).get("paper_id")
        or (description_block or {}).get("paper_id")
    )

    section = (
        formula.get("section")
        or (paragraph or {}).get("section")
        or (description_block or {}).get("section")
    )

    page = (
        formula.get("page")
        or (paragraph or {}).get("page")
        or (description_block or {}).get("page")
    )

    label = label_block.get("content", "") if label_block else formula.get("label")

    eq_block = {
        "type": "equation_block",
        "latex": latex,
        "raw": formula.get("raw", latex),
        "source": source,
        "paper_id": paper_id,
        "section": section,
        "page": page,
        "label": label,
        "confidence": formula.get("confidence", 0.5),
        "score": formula.get("score"),
        "ast": formula.get("ast", False),
        "context_before": context_before,
        "description": description,
        "semantic_description": formula.get("semantic_description") or context_before[:300],
        "formulas": [
    {
        "id": formula.get("id"),
        "number": formula.get("number"),
        "latex": latex,
        "raw": formula.get("raw", latex),
        "description": formula.get("description") or formula.get("semantic_description") or context_before[:300],
        "semantic_description": formula.get("semantic_description") or formula.get("description") or context_before[:300],
        "variables": formula.get("variables", []),
        "context": formula.get("context"),
        "source": source,
        "page": page,
        "confidence": formula.get("confidence", 0.5),
        "type": formula.get("type", "formula_candidate"),
    }
],
        "metadata": {
            "has_formula": True,
            "formula_count": 1,
            "from_formula_candidate": formula.get("type") == "formula_candidate",
        },
    }

    return eq_block


def normalize_equation_block(block: Dict[str, Any]) -> Dict[str, Any]:
    b = block.copy()
    latex = b.get("latex") or b.get("normalized") or ""

    formulas = b.get("formulas") or []
    if not formulas and latex:
        formulas = [
            {
                "latex": latex,
                "raw": b.get("raw", latex),
                "description": b.get("semantic_description", ""),
                "variables": b.get("variables", []),
                "source": b.get("source"),
                "page": b.get("page"),
                "confidence": b.get("confidence", 0.5),
                "type": "formula_candidate",
            }
        ]

    b["type"] = "equation_block"
    b["latex"] = latex
    b["formulas"] = formulas
    b.setdefault("metadata", {})
    b["metadata"]["has_formula"] = bool(formulas)
    b["metadata"]["formula_count"] = len(formulas)

    return b


def remove_empty_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []

    for block in blocks:
        block_type = block.get("type", "paragraph")

        has_content = bool(str(block.get("content", "")).strip())
        has_latex = bool(str(block.get("latex", "")).strip())
        has_formulas = bool(block.get("formulas"))

        if block_type in {"equation_block", "formula", "formula_candidate"}:
            if not has_latex and not has_content and not has_formulas:
                continue

        elif block_type == "paragraph":
            if not has_content and not has_formulas:
                continue

        elif block_type == "section":
            if not has_content:
                continue

        result.append(block)

    return result


def propagate_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    current_section = None
    result = []

    for block in blocks:
        b = block.copy()

        if b.get("type") == "section":
            current_section = b.get("content") or b.get("section")
            b["section"] = current_section
            result.append(b)
            continue

        if current_section and not b.get("section"):
            b["section"] = current_section

        result.append(b)

    return result