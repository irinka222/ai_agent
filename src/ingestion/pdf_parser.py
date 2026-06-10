#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import fitz

from src.config import OCR_ENABLED, OCR_LANG
from src.processing.equation_database import EQUATIONS, KEYWORDS
from src.processing.formula_extractor import clean_formula, is_valid_formula
from src.knowledge.kb_instance import KB

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Замена ссылок на уравнения из базы
# ----------------------------------------------------------------------
def replace_equations_with_latex(text: str) -> str:
    result = text
    for eq_id, keywords in KEYWORDS.items():
        latex = EQUATIONS.get(eq_id, "")
        if not latex:
            continue
        for kw in keywords:
            if kw in result:
                replacement = f"\n[ФОРМУЛА: {latex}]\n"
                result = result.replace(kw, replacement)
    return result


# ----------------------------------------------------------------------
# Обработка формул через KB
# ----------------------------------------------------------------------
from src.processing.formula_extractor import get_formula_semantic_description

def process_formula_block(block_text: str, page: int, source: str) -> Dict:
    if not is_valid_formula(block_text):
        return None

    stripped = block_text.strip()
    # Отбрасываем деградировавшие формулы PyMuPDF:
    # много одиночных символов/переменных, но нет нормальной структуры LaTeX.
    tokens = stripped.split()

    if len(tokens) > 25:
        single_char_tokens = [
            t for t in tokens
            if len(t) <= 2 and re.match(r"^[A-Za-zА-Яа-я0-9_=+\-.,()]+$", t)
        ]

        if len(single_char_tokens) / max(len(tokens), 1) > 0.55:
            return None

    # Жёсткая фильтрация: настоящие формулы обычно содержат '=', '∂', '∇', '∫', '∑', или русские буквы только в описании констант
    # Если нет символа '=', и нет '∂∇∫∑', и есть хотя бы одно русское слово из списка — отбрасываем
    math_ops = '=∂∇∫∑'
    has_math_op = any(op in stripped for op in math_ops)

    # Русские слова, характерные для текста, а не для формул
    text_words = ['скорость', 'процесс', 'молекула', 'вода', 'ион', 'концентрация', 'где', 'рисунок', 'таблица']
    has_text_word = any(word in stripped.lower() for word in text_words)

    if not has_math_op and has_text_word:
        return None

    # Если строка содержит '=', но в ней больше 5 слов и нет других мат.символов — скорее текст
    if '=' in stripped:
        words = len(stripped.split())
        if words > 5 and len(re.findall(r'[∂∇∑∫]', stripped)) == 0:
            return None

    # Если строка содержит более 3 русских слов — отбрасываем
    russian_words = re.findall(r'[А-Яа-я]{3,}', stripped)
    if len(russian_words) > 3:
        return None

    # Если строка состоит в основном из букв и пробелов, и нет '=', отбрасываем
    if stripped.count('=') == 0:
        letter_ratio = len(re.findall(r'[A-Za-zА-Яа-я]', stripped)) / len(stripped) if stripped else 0
        if letter_ratio > 0.6:
            return None

    # Если длина меньше 10 и нет цифр — отбрасываем
    if len(stripped) < 10 and not re.search(r'\d', stripped):
        return None

    normalized = clean_formula(stripped)
    # Если после нормализации формула всё ещё выглядит как набор обрывков,
    # не считаем её пригодной для формульного индекса.
    if len(normalized.split()) > 30 and normalized.count("\\") < 2:
        return None
    if len(normalized) < 5:
        return None

    semantic = get_formula_semantic_description(normalized)

    formula_id = KB.add_formula(
        formula=normalized,
        embedding=None,
        source=source,
        page=page,
        domain="electrochemistry",
        score=0.7
    )
    return {
        "type": "formula",
        "latex": normalized,
        "content": f"[FORMULA_ID: {formula_id}] {normalized}",
        "page": page,
        "source": source,
        "formula_id": formula_id,
        "semantic_description": semantic
    }
# ----------------------------------------------------------------------
# Метаданные (авторы, год, заголовок) – без изменений
# ----------------------------------------------------------------------
def normalize_author_name(name: str) -> str:
    name = re.sub(r'\s+', ' ', name).strip()
    name = re.sub(r'\d+$', '', name)
    name = re.sub(r'\s*[a-z]?,\s*\d?\*?$', '', name)
    name = re.sub(r'\(?orcid[^)]*\)?', '', name, flags=re.I)
    name = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '', name)
    match = re.match(r'^([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)$', name)
    if match:
        last = match.group(1)
        first_initial = match.group(2)[0]
        middle_initial = match.group(3)[0]
        return f"{last} {first_initial}.{middle_initial}."
    match = re.match(r'^([А-ЯЁ][а-яё]+)\s+([А-ЯЁ]\.(?:[А-ЯЁ]\.)?)$', name)
    if match:
        return name
    match = re.match(r'^([А-ЯЁ]\.(?:[А-ЯЁ]\.)?)\s+([А-ЯЁ][а-яё]+)$', name)
    if match:
        initials = match.group(1)
        last = match.group(2)
        return f"{last} {initials}"
    match = re.match(r'^([A-Z]\.(?:[A-Z]\.)?)\s+([A-Z][a-z]+)$', name)
    if match:
        initials = match.group(1)
        last = match.group(2)
        return f"{last} {initials}"
    match = re.match(r'^([A-Z][a-z]+)\s+([A-Z]\.(?:[A-Z]\.)?)$', name)
    if match:
        last = match.group(1)
        initials = match.group(2)
        return f"{last} {initials}"
    return name


def is_valid_author(name: str) -> bool:
    if not name or len(name) < 3 or len(name) > 50:
        return False
    if not any(c.isalpha() for c in name):
        return False
    if any(c.isdigit() for c in name):
        return False
    if '.' not in name and ' ' not in name:
        return False
    stop_words = {
        'университет', 'university', 'институт', 'institute', 'лаборатория',
        'кафедра', 'department', 'факультет', 'faculty', 'министерство',
        'профессор', 'professor', 'доцент', 'доктор', 'кандидат', 'аспирант',
        'заведующий', 'руководитель', 'научный', 'ректор', 'декан',
        'федеральное', 'государственное', 'бюджетное', 'учреждение',
        'page', 'figure', 'table', 'equation', 'section',
        'научный руководитель', 'оппонент', 'рецензент', 'ведущая организация'
    }
    name_lower = name.lower()
    for sw in stop_words:
        if sw in name_lower:
            return False
    return True


def clean_authors_list(authors: List[str]) -> List[str]:
    seen = set()
    cleaned = []
    for raw in authors:
        raw = re.sub(r'\d+$', '', raw)
        raw = re.sub(r'\s*[a-z]?,\s*\d?\*?$', '', raw)
        raw = re.sub(r'\s+', ' ', raw).strip()
        if '@' in raw:
            continue
        if not is_valid_author(raw):
            continue
        norm = normalize_author_name(raw)
        if any(word in norm.lower() for word in ['университет', 'university', 'институт', 'institute']):
            continue
        if norm not in seen:
            seen.add(norm)
            cleaned.append(norm)
    return cleaned   # Убрано [:5]


def extract_dissertation_author(lines: List[str]) -> List[str]:
    for i, line in enumerate(lines):
        if "на правах рукописи" in line.lower():
            for j in range(i + 1, min(i + 15, len(lines))):
                candidate = lines[j].strip()
                if not candidate:
                    continue
                match = re.match(r'^([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)\s+([А-ЯЁ][а-яё]+)$', candidate)
                if match:
                    return clean_authors_list([candidate])
                match = re.match(r'^([А-ЯЁ][а-яё]+)\s+([А-ЯЁ]\.(?:[А-ЯЁ]\.)?)$', candidate)
                if match:
                    return clean_authors_list([candidate])
                match = re.match(r'^([А-ЯЁ]\.(?:[А-ЯЁ]\.)?)\s+([А-ЯЁ][а-яё]+)$', candidate)
                if match:
                    return clean_authors_list([candidate])
            break
    return []


def extract_article_authors(text: str) -> List[str]:
    authors = []
    credit_match = re.search(r'CRediT authorship contribution statement\s*\n(.*?)(?:\n\n|\Z)', text, re.DOTALL | re.IGNORECASE)
    if credit_match:
        for line in credit_match.group(1).split('\n'):
            if ':' in line:
                name = line.split(':')[0].strip()
                if name and is_valid_author(name):
                    authors.append(name)
        if authors:
            return clean_authors_list(authors)
    name_with_initials = re.findall(r'([A-Z]\.(?:[A-Z]\.)?)\s+([A-Z][a-z]+)', text[:3000])
    for init, last in name_with_initials:
        authors.append(f"{init} {last}")
    if authors:
        return clean_authors_list(authors)
    return []


def extract_authors_advanced(text: str, source_name: str = "") -> List[str]:
    lines = text.split('\n')
    is_dissertation = "на правах рукописи" in text.lower()
    if is_dissertation:
        authors = extract_dissertation_author(lines)
        if authors:
            return authors
    return extract_article_authors(text)


def extract_dissertation_title(lines: List[str]) -> str:
    start = None
    for i, line in enumerate(lines):
        if "на правах рукописи" in line.lower():
            start = i
            break
    if start is None:
        return None
    candidate_lines = []
    for line in lines[start + 1:start + 30]:
        line = line.strip()
        if not line:
            continue
        if re.search(r'(диссертаци|автореферат|на соискание|ученой степени|специальность)', line, re.I):
            break
        if re.match(r'^[А-ЯЁ][а-яё]+(\s+[А-ЯЁ][а-яё]+){1,2}$', line):
            continue
        if re.match(r'^\d{2}\.\d{2}\.\d{2}', line):
            break
        candidate_lines.append(line)
    if not candidate_lines:
        return None
    title = " ".join(candidate_lines)
    title = re.sub(r'\s+', ' ', title)
    title = title.strip(' .;:-')
    if len(title) > 20:
        return title
    return None


def extract_article_title(doc: fitz.Document, first_page_text: str, source_name: str) -> str:
    if doc and doc.metadata:
        meta = doc.metadata
        if meta.get("title") and len(meta["title"]) > 5:
            title_candidate = meta["title"]
            if title_candidate.lower() not in ["на правах рукописи", "title", "untitled"]:
                title_candidate = re.sub(r'\s+', ' ', title_candidate)
                if len(title_candidate) > 10:
                    return title_candidate
    latex_match = re.search(r'\\title\{(.*?)\}', first_page_text)
    if latex_match:
        return latex_match.group(1)
    lines = first_page_text.split('\n')
    for line in lines[:30]:
        line = line.strip()
        if len(line) > 20 and len(line) < 200:
            if not re.search(r'(abstract|keywords|doi|received|revised|accepted)', line, re.I):
                return line
    return source_name.replace('_', ' ')


def extract_title_smart(doc: fitz.Document, source_name: str, first_page_text: str) -> str:
    lines = first_page_text.split('\n')
    is_dissertation = "на правах рукописи" in first_page_text.lower()
    if is_dissertation:
        title = extract_dissertation_title(lines)
        if title:
            return title
    return extract_article_title(doc, first_page_text, source_name)


def extract_year_smart(text: str, metadata: dict = None) -> str:
    first_page = text[:5000] if len(text) > 5000 else text
    years = re.findall(r'\b(19[0-9]{2}|20[0-2][0-9]|2030)\b', first_page)
    if years:
        filtered = [y for y in years if 1950 <= int(y) <= 2030]
        if filtered:
            return filtered[-1]
    doi_match = re.search(r'10\.\d{4}/[^\s]+\.(\d{4})\.', first_page)
    if doi_match:
        return doi_match.group(1)
    acc_match = re.search(r'Accepted\s*:\s*\d{1,2}\s+[A-Za-z]+\s+(\d{4})', text, re.I)
    if acc_match:
        return acc_match.group(1)
    rec_match = re.search(r'Received\s*:\s*\d{1,2}\s+[A-Za-z]+\s+(\d{4})', text, re.I)
    if rec_match:
        return rec_match.group(1)
    copy_match = re.search(r'©\s*(\d{4})', first_page)
    if copy_match:
        year = copy_match.group(1)
        if 1950 <= int(year) <= 2030:
            return year
    if metadata and metadata.get("creationDate"):
        y = re.search(r'(19[0-9]{2}|20[0-2][0-9])', metadata["creationDate"])
        if y:
            year = y.group(1)
            if 1950 <= int(year) <= 2030:
                return year
    return None


# ----------------------------------------------------------------------
# Извлечение текстовых блоков
# ----------------------------------------------------------------------
def extract_text_blocks(page) -> List[Dict]:
    blocks = []
    text_dict = page.get_text("dict")
    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        lines = []
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                spans.append(span.get("text", ""))
            if spans:
                lines.append("".join(spans))
        block_text = "\n".join(lines).strip()
        if not block_text:
            continue
        if re.match(r'^\s*\d+\s*$', block_text):
            continue
        if re.search(r'\.{3,}\s*\d+$', block_text):
            continue
        if re.match(r'^\d+(\.\d+)*\s+.+\s+\d+$', block_text):
            continue
        blocks.append({
            "text": block_text,
            "bbox": block.get("bbox", (0,0,0,0))
        })
    return blocks


def merge_page_blocks(blocks: List[Dict]) -> List[Dict]:
    if not blocks:
        return blocks
    merged = []
    current = blocks[0].copy()
    for block in blocks[1:]:
        prev_y = current["bbox"][3]
        curr_y = block["bbox"][1]
        prev_x = current["bbox"][0]
        curr_x = block["bbox"][0]
        same_column = abs(prev_x - curr_x) < 120
        distance = curr_y - prev_y
        if distance < 15 and same_column:
            current["text"] += " " + block["text"]
            current["bbox"] = (
                min(current["bbox"][0], block["bbox"][0]),
                min(current["bbox"][1], block["bbox"][1]),
                max(current["bbox"][2], block["bbox"][2]),
                max(current["bbox"][3], block["bbox"][3])
            )
        else:
            merged.append(current)
            current = block.copy()
    merged.append(current)
    return merged


# ----------------------------------------------------------------------
# Основной парсер
# ----------------------------------------------------------------------
def parse_pdf_pymupdf(pdf_path: str, source_name: str) -> Tuple[List[Dict], Dict]:
    doc = fitz.open(pdf_path)
    full_text = ""
    first_page_text = ""
    blocks = []

    for page_num, page in enumerate(doc, start=1):
        page_blocks = extract_text_blocks(page)
        if not page_blocks:
            continue
        page_blocks = merge_page_blocks(page_blocks)

        if page_num == 1:
            first_page_text = page.get_text("text")

        page_text = " ".join([b["text"] for b in page_blocks])
        full_text += page_text + "\n"

        for page_block in page_blocks:
            block_text = page_block["text"]

            # 1. Сначала пробуем обработать как формулу через KB
            if is_valid_formula(block_text):
                formula_block = process_formula_block(block_text, page_num, source_name)
                if formula_block:
                    blocks.append(formula_block)
                    continue

            # 2. Иначе заменяем известные уравнения из базы
            block_text = replace_equations_with_latex(block_text)

            blocks.append({
                "type": "paragraph",
                "content": block_text,
                "page": page_num,
                "source": source_name,
                "formulas": []
            })

    title = extract_title_smart(doc, source_name, first_page_text)
    year = extract_year_smart(full_text, doc.metadata)
    authors = extract_authors_advanced(first_page_text, source_name)

    doc.close()

    if not blocks and OCR_ENABLED:
        logger.info(f"Text too short, trying OCR")
        try:
            from src.ingestion.ocr import ocr_full_pdf
            ocr_text = ocr_full_pdf(pdf_path, lang=OCR_LANG)
            if ocr_text:
                blocks = [{
                    "type": "paragraph",
                    "content": ocr_text,
                    "page": 1,
                    "source": source_name,
                    "formulas": []
                }]
                title = extract_title_smart(None, source_name, ocr_text[:5000])
                year = extract_year_smart(ocr_text, None)
                authors = extract_authors_advanced(ocr_text, source_name)
        except Exception as e:
            logger.error(f"OCR failed: {e}")

    metadata = {
        "source": source_name,
        "title": title,
        "authors": authors,
        "year": year
    }
    logger.info(f"Extracted: authors={authors}, year={year}, title={title[:60] if title else None}")
    logger.info(f"Total blocks: {len(blocks)}")
    return blocks, metadata


def unified_parse_pdf(pdf_path: str, use_grobid: bool = False) -> Tuple[List[Dict], Dict]:
    source = Path(pdf_path).stem
    return parse_pdf_pymupdf(pdf_path, source)