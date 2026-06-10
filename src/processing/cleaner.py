# src/processing/cleaner.py
# -*- coding: utf-8 -*-

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

FORMULA_TYPES = {"formula", "equation_block", "formula_candidate", "formula_fragment"}


def clean_text_advanced(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\(cid:\d+\)", "", text)
    text = re.sub(r"([A-Za-zА-Яа-яЁё])-+\s*\n\s*([A-Za-zА-Яа-яЁё])", r"\1\2", text)
    text = re.sub(r"([A-Za-zА-Яа-яЁё])-+\s+([A-Za-zА-Яа-яЁё])", r"\1\2", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

    replacements = {
        "\uf0b7": "-",
        "\ufeff": "",
        "\u200b": "",
        "\u00a0": " ",
        "": " ",
        "…": "...",
        "−": "-",
        "–": "-",
        "—": "-",
        "∙": "·",
        "⋅": "·",
        "§": "",
        "™": "",
        "®": "",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def clean_latex_light(latex: str) -> str:
    if not latex:
        return ""

    latex = re.sub(r"\(cid:\d+\)", "", latex)
    latex = latex.replace("\u00a0", " ")
    latex = latex.replace("\u200b", "")
    latex = latex.replace("\ufeff", "")
    latex = re.sub(r"[ \t]+", " ", latex)

    return latex.strip()


def is_formula_fragment(text: str) -> bool:
    if not text:
        return False

    s = text.strip()

    fragments = [
        r"^[+\-]H$",
        r"^[+\-]OH$",
        r"^H[+\-]$",
        r"^OH[+\-]$",
        r"^[+\-]?\d*OH[+\-]?$",
        r"^[A-ZА-Я][0-9]+[A-ZА-Я]?$",
        r"^[A-ZА-Я][0-9]*[+\-]$",
        r"^[+\-][A-ZА-Я][0-9]*$",
        r"^[cdjxytuvCEPIRFDkKwWrTφϕερν]+[_\d]*$",
        r"^[=+\-*/^_{}()[\].,;:]+$",
    ]

    return any(re.match(pattern, s) for pattern in fragments)


def is_garbage_formula(text: str) -> bool:
    if not text:
        return True

    s = text.strip()

    if len(s) < 2:
        return True

    if re.match(r"^\d{1,4}$", s):
        return True

    words = re.findall(r"[A-Za-zА-Яа-яЁё]{2,}", s)
    if len(words) > 35:
        return True

    garbage_patterns = [
        r"^_$",
        r"^[.,;:]+$",
        r"^(NaCl|KCl|CEM|AEM|EDL|SCR|DL|PD|PS)$",
    ]

    return any(re.match(pattern, s) for pattern in garbage_patterns)


def is_likely_section_title(text: str) -> bool:
    if not text:
        return False

    text_clean = text.strip()

    if len(text_clean) > 220:
        return False

    exact_patterns = [
        r"^(Введение|INTRODUCTION)$",
        r"^(Аннотация|ABSTRACT|Реферат|РЕФЕРАТ)$",
        r"^(Ключевые слова|KEYWORDS)$",
        r"^(Заключение|CONCLUSION|ВЫВОДЫ|Conclusions)$",
        r"^(Литература|Список литературы|REFERENCES|БИБЛИОГРАФИЧЕСКИЙ СПИСОК)$",
        r"^(Результаты|RESULTS|РЕЗУЛЬТАТЫ)$",
        r"^(Обсуждение|DISCUSSION|ОБСУЖДЕНИЕ)$",
        r"^(Методы|METHODS|МЕТОДЫ|Материалы и методы)$",
        r"^(Теоретическая часть|THEORETICAL)$",
        r"^(Математическая модель|MATHEMATICAL MODEL)$",
        r"^(Постановка задачи|Problem statement)$",
    ]

    for pattern in exact_patterns:
        if re.match(pattern, text_clean, re.IGNORECASE):
            return True

    numbered_patterns = [
        r"^Глава\s+\d+\.?\s+.+",
        r"^\d+\.\d+\.\d+\s+[A-ZА-ЯЁ].+",
        r"^\d+\.\d+\s+[A-ZА-ЯЁ].+",
        r"^\d+\.\s+[A-ZА-ЯЁ].+",
        r"^\d+\s+[A-ZА-ЯЁ][^.!?]{5,180}$",
    ]

    for pattern in numbered_patterns:
        if re.match(pattern, text_clean):
            return True

    if (
        8 <= len(text_clean) <= 120
        and not text_clean.endswith(".")
        and re.match(r"^[A-ZА-ЯЁ]", text_clean)
        and len(re.findall(r"\w+", text_clean)) <= 12
    ):
        keywords = [
            "модель", "уравнения", "электроконвекция", "результаты",
            "концентрация", "потенциал", "диффузионный слой",
            "model", "equations", "electroconvection", "results",
            "diffusion layer", "mathematical"
        ]
        if any(k.lower() in text_clean.lower() for k in keywords):
            return True

    return False


def split_embedded_sections(block: Dict[str, Any]) -> List[Dict[str, Any]]:
    content = block.get("content", "")
    block_type = block.get("type", "paragraph")

    if block_type != "paragraph":
        return [block]

    patterns = [
        r"^(Глава\s+\d+\.?\s+[А-ЯЁA-Z][^.!?]{10,120})\s+(.+)$",
        r"^(\d+\.\d+\.\d+\s+[А-ЯЁA-Z][^.!?]{10,120})\s+(.+)$",
        r"^(\d+\.\d+\s+[А-ЯЁA-Z][^.!?]{10,120})\s+(.+)$",
        r"^(\d+\.\s+[А-ЯЁA-Z][^.!?]{10,120})\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.match(pattern, content, re.DOTALL)
        if not match:
            continue

        title_part = match.group(1).strip()
        text_part = match.group(2).strip()

        if text_part and len(title_part) < 160:
            section_block = block.copy()
            section_block["type"] = "section"
            section_block["content"] = title_part
            section_block["section"] = title_part

            paragraph_block = block.copy()
            paragraph_block["type"] = "paragraph"
            paragraph_block["content"] = text_part
            paragraph_block["section"] = title_part

            return [section_block, paragraph_block]

    return [block]


def detect_sections(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    section_count = 0
    current_section = None
    result = []

    for block in blocks:
        b = block.copy()
        block_type = b.get("type", "paragraph")

        if block_type in FORMULA_TYPES:
            if current_section and not b.get("section"):
                b["section"] = current_section
            result.append(b)
            continue

        content = b.get("content", "").strip()
        if not content:
            continue

        if is_likely_section_title(content):
            b["type"] = "section"
            b["section"] = content[:180]
            current_section = b["section"]
            section_count += 1
        else:
            b["type"] = "paragraph"
            if current_section and not b.get("section"):
                b["section"] = current_section

        result.append(b)

    logger.info("Total detected sections: %d", section_count)
    return result


def mark_front_matter(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    intro_index = None

    for i, block in enumerate(blocks):
        content = block.get("content", "").strip().lower()

        if content == "введение" or content.startswith("введение"):
            intro_index = i
            break

        if content == "introduction" or content.startswith("introduction"):
            intro_index = i
            break

    if intro_index is None:
        return blocks

    marked = []

    for i, block in enumerate(blocks):
        b = block.copy()

        if i < intro_index:
            b.setdefault("metadata", {})
            b["metadata"]["front_matter"] = True
            if not b.get("section"):
                b["section"] = "front_matter"

        marked.append(b)

    logger.info("Marked %d blocks as front_matter", intro_index)
    return marked


def should_merge_with_previous(prev: str, current: str) -> bool:
    if not prev or not current:
        return False

    current_stripped = current.strip()
    prev_stripped = prev.strip()

    if not current_stripped:
        return False

    if current_stripped[0].islower():
        return True

    if prev_stripped.endswith((",", ";", ":", "и", "или")):
        return True

    if not prev_stripped.endswith((".", "!", "?", ":", ";")):
        if len(current_stripped.split()) > 3:
            return True

    return False


def merge_split_paragraphs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not blocks:
        return []

    merged = []
    current = None

    for block in blocks:
        block_type = block.get("type", "paragraph")
        content = block.get("content", "").strip()
        latex = block.get("latex", "").strip()
        formulas = block.get("formulas", [])

        if not content and not latex and not formulas:
            continue

        if content and is_formula_fragment(content) and block_type == "paragraph":
            if current:
                merged.append(current)
                current = None

            fragment = block.copy()
            fragment["type"] = "formula_fragment"
            fragment.setdefault("metadata", {})
            fragment["metadata"]["formula_fragment"] = True
            merged.append(fragment)
            continue

        if block_type in FORMULA_TYPES or block_type == "section":
            if current:
                merged.append(current)
                current = None
            merged.append(block)
            continue

        if block_type == "paragraph":
            if current and should_merge_with_previous(current.get("content", ""), content):
                current["content"] += " " + content

                prev_page = current.get("page")
                new_page = block.get("page")

                if prev_page != new_page:
                    pages = current.get("pages")
                    if not pages:
                        pages = [prev_page] if prev_page is not None else []
                    if new_page is not None and new_page not in pages:
                        pages.append(new_page)
                    current["pages"] = pages

                if not current.get("section") and block.get("section"):
                    current["section"] = block.get("section")
            else:
                if current:
                    merged.append(current)
                current = block.copy()

    if current:
        merged.append(current)

    logger.debug("Merged paragraphs: %d -> %d blocks", len(blocks), len(merged))
    return merged


def remove_empty_and_duplicate_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    prev_signature = None

    for block in blocks:
        content = str(block.get("content", "")).strip()
        latex = str(block.get("latex", "")).strip()
        formulas = block.get("formulas", [])

        if not content and not latex and not formulas:
            continue

        signature = (
            block.get("type", "paragraph"),
            content[:200],
            latex[:200],
            block.get("page"),
        )

        if signature == prev_signature:
            continue

        result.append(block)
        prev_signature = signature

    return result


def clean_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not blocks:
        return []

    cleaned = []

    for block in blocks:
        b = block.copy()

        if "content" in b:
            b["content"] = clean_text_advanced(b.get("content", ""))

        if "latex" in b:
            b["latex"] = clean_latex_light(b.get("latex", ""))

        if "source" not in b:
            b["source"] = b.get("paper_id", "unknown")

        block_type = b.get("type", "paragraph")

        if block_type == "formula":
            formula_text = b.get("latex") or b.get("content", "")
            if is_garbage_formula(formula_text):
                continue

        has_content = bool(str(b.get("content", "")).strip())
        has_latex = bool(str(b.get("latex", "")).strip())
        has_formulas = bool(b.get("formulas"))

        if not has_content and not has_latex and not has_formulas:
            continue

        cleaned.extend(split_embedded_sections(b))

    cleaned = remove_empty_and_duplicate_blocks(cleaned)
    cleaned = detect_sections(cleaned)
    cleaned = mark_front_matter(cleaned)
    cleaned = merge_split_paragraphs(cleaned)
    cleaned = remove_empty_and_duplicate_blocks(cleaned)

    logger.info("Cleaned blocks: %d remain from original %d", len(cleaned), len(blocks))
    return cleaned