import re
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def is_valid_formula(latex: str) -> bool:
    if not latex or len(latex) < 3:
        return False
    if '(cid:' in latex:
        return False
    if re.search(r'[§©¨•]', latex):
        return False
    # Разрешаем короткие русские слова (до 3 букв)
    if re.search(r'[а-яА-Я]{4,}', latex):
        return False
    if len(latex) > 200:
        return False
    return True

def parse_ast_safe(latex: str):
    try:
        import sympy as sp
        from sympy.parsing.latex import parse_latex
        return parse_latex(latex)
    except Exception as e:
        logger.debug(f"AST parse failed: {latex[:50]}... | {e}")
        return None

def compute_confidence(latex: str, ast, source: str) -> float:
    source_weights = {
        'grobid': 0.4,
        'mathml': 0.35,
        'heuristic': 0.2,
        'ocr': 0.05
    }
    score = source_weights.get(source, 0.1)
    if any(c in latex for c in ['=', '\\frac', '^', '_', '\\int', '\\sum']):
        score += 0.3
    if ast is not None:
        score += 0.2
    else:
        score += 0.05
    if len(latex) < 100:
        score += 0.1
    if '=' in latex or re.search(r'd\s*[A-Za-z]', latex):
        score += 0.1
    return min(score, 1.0)

def extract_full_formula(elem, source='grobid') -> Optional[Dict[str, Any]]:
    latex = elem.attrib.get('tex') or elem.text
    if not latex:
        return None
    latex = latex.strip()
    if not is_valid_formula(latex):
        return None
    ast = parse_ast_safe(latex)
    confidence = compute_confidence(latex, ast, source)
    if confidence < 0.3:
        return None
    return {
        "latex": latex,
        "raw": latex,
        "source": source,
        "confidence": confidence,
        "ast": ast is not None,
        "variables": [],
        "dimension_signature": None
    }