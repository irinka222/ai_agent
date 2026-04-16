import requests
import pdfplumber
import re
import logging
from pathlib import Path
from src.config import GROBID_URL, OCR_ENABLED
from src.processing.cleaner import clean_text
from src.processing.formula_extractor import is_valid_formula, parse_ast_safe, compute_confidence

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# GROBID парсер
# ----------------------------------------------------------------------
def parse_pdf_grobid(pdf_path: str):
    try:
        with open(pdf_path, "rb") as f:
            files = {"input": f}
            data = {"teiCoordinates": "formula,head,p,table"}
            response = requests.post(GROBID_URL, files=files, data=data, timeout=120)
        if response.status_code != 200:
            logger.warning(f"GROBID status {response.status_code}")
            return None
        if len(response.text) < 1000:
            logger.warning("GROBID returned too short response")
            return None
        return response.text
    except Exception as e:
        logger.error(f"GROBID error: {e}")
        return None

# ----------------------------------------------------------------------
# Вспомогательные функции для двухколоночного PDF
# ----------------------------------------------------------------------
def extract_columns(page):
    width = page.width
    left_bbox = (0, 0, width/2, page.height)
    right_bbox = (width/2, 0, width, page.height)
    left = page.within_bbox(left_bbox).extract_text() or ""
    right = page.within_bbox(right_bbox).extract_text() or ""
    return left + "\n" + right

# ----------------------------------------------------------------------
# Фильтры для отсева мусора (ослабленные)
# ----------------------------------------------------------------------
def is_garbage_formula(expr: str) -> bool:
    words = re.findall(r'[A-Za-zА-Яа-я]+', expr)
    if len(words) > 8:
        return True
    cyrillic = len(re.findall(r'[А-Яа-я]', expr))
    if cyrillic > len(expr) * 0.4:
        return True
    return False

def has_math_structure(expr: str) -> bool:
    return any([
        re.search(r'.+=.+', expr),
        re.search(r'[A-Za-z]\s*[_^]\s*\d+', expr),
        re.search(r'[()]{2,}', expr),
        re.search(r'\d+\s*[*/+-]\s*\d+', expr),
        re.search(r'[A-Za-z]\d+', expr),
        re.search(r'\d+[A-Za-z]', expr),
        re.search(r'd\s*[A-Za-z]', expr),   # производная dN
    ])

def is_too_simple(expr: str) -> bool:
    # одна буква
    if re.fullmatch(r'[A-Za-z]', expr):
        return True
    tokens = re.findall(r'[A-Za-z0-9]+', expr)
    if len(tokens) < 2 and not re.search(r'[=+\-*/^]', expr):
        return True
    return False

def has_operator(expr: str) -> bool:
    return bool(re.search(r'[=+\-*/^]|d\s*[A-Za-z]|∂', expr))

def math_density(expr: str) -> float:
    if not expr:
        return 0.0
    math_chars = len(re.findall(r'[=+\-*/^_{}()]', expr))
    return math_chars / len(expr)

def compact_math(expr: str) -> str:
    expr = re.sub(r'\b([a-zA-Z])\s+([a-zA-Z])\b', r'\1\2', expr)
    expr = re.sub(r'\s+', ' ', expr)
    return expr.strip()

def normalize_formula_heuristic(expr: str) -> str:
    expr = expr.replace('–', '-').replace('—', '-').replace('−', '-')
    expr = expr.replace('×', '*').replace('÷', '/')
    expr = re.sub(r'\s*=\s*', ' = ', expr)
    expr = re.sub(r'\s*\+\s*', ' + ', expr)
    expr = re.sub(r'\s*\-\s*', ' - ', expr)
    expr = re.sub(r'\s*\*\s*', ' * ', expr)
    expr = re.sub(r'\s*/\s*', ' / ', expr)
    expr = re.sub(r'\s+', ' ', expr).strip()
    expr = re.sub(r'([A-Za-z])(\d+)', r'\1_{\2}', expr)
    return expr

# ----------------------------------------------------------------------
# Извлечение формул из текста (эвристика)
# ----------------------------------------------------------------------
def extract_formulas_from_text_heuristic(text: str):
    formulas = []
    candidates = re.findall(
        r".{0,80}=[^=]{0,80}"
        r"|[A-Za-z][A-Za-z0-9_]*\s*=\s*[^=\n]+"
        r"|\b[A-Za-z]\d+\b"
        r"|\b(?=[A-Za-z]*\d+[A-Za-z]*\b)[A-Za-z0-9]{4,30}\b"
        r"|\b[A-Za-z]+\d+[A-Za-z]+\b",
        text
    )
    candidates = list(set(candidates))

    for cand in candidates:
        cand = cand.strip()
        if len(cand.split()) > 20:
            continue
        if is_garbage_formula(cand):
            continue
        if is_too_simple(cand):
            continue
        if not has_operator(cand):
            continue
        if not has_math_structure(cand):
            continue
        if math_density(cand) < 0.05:
            continue

        cand = compact_math(cand)
        if not is_valid_formula(cand):
            continue

        ast = parse_ast_safe(cand)
        confidence = compute_confidence(cand, ast, source='heuristic')
        if confidence < 0.3:
            continue

        formulas.append({
            "latex": normalize_formula_heuristic(cand),
            "raw": cand,
            "source": "heuristic",
            "confidence": confidence,
            "ast": ast is not None
        })

    # уникализация
    unique = {}
    for f in formulas:
        key = f["latex"]
        if key not in unique:
            unique[key] = f
    return list(unique.values())

# ----------------------------------------------------------------------
# Fallback парсер (pdfplumber + эвристика)
# ----------------------------------------------------------------------
def parse_pdf_fallback(pdf_path: str):
    blocks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = extract_columns(page)
            if (not text or len(text.strip()) < 50) and OCR_ENABLED:
                try:
                    from src.processing.ocr import ocr_page
                    text = ocr_page(pdf_path, page_num) or ""
                except Exception as e:
                    logger.error(f"OCR failed: {e}")
                    text = ""
            text = clean_text(text)
            if text:
                formulas = extract_formulas_from_text_heuristic(text)
                logger.info(f"Page {page_num}: extracted {len(formulas)} formulas")
                blocks.append({
                    "type": "paragraph",
                    "content": text,
                    "page": page_num,
                    "section": None,
                    "formulas": formulas
                })
    return blocks

# ----------------------------------------------------------------------
# Унифицированный парсер (GROBID + fallback)
# ----------------------------------------------------------------------
def unified_parse_pdf(pdf_path: str):
    tei_blocks = []
    tei_xml = parse_pdf_grobid(pdf_path)
    if tei_xml:
        with open("debug_tei.xml", "w", encoding="utf-8") as f:
            f.write(tei_xml)
        from src.processing.tei_parser import parse_tei
        tei_blocks = parse_tei(tei_xml, paper_id=Path(pdf_path).stem)
        logger.info(f"GROBID blocks: {len(tei_blocks)}")

    fallback_blocks = parse_pdf_fallback(pdf_path)
    logger.info(f"Fallback blocks: {len(fallback_blocks)}")

    # Объединяем оба источника
    return tei_blocks + fallback_blocks