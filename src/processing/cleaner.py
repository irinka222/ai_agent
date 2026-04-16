import re
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)
MIN_TEXT_LEN = 3
MAX_CHUNK_LEN = 1000

def _remove_cid(text: str) -> str:
    return re.sub(r'\(cid:\d+\)', '', text)

def _normalize_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'([a-zа-я])([A-ZА-Я])', r'\1 \2', text)
    return text.strip()

def clean_blocks(blocks: List[Dict]) -> List[Dict]:
    for b in blocks:
        if 'content' not in b and 'text' in b:
            b['content'] = b['text']
        if 'formulas' not in b:
            b['formulas'] = []
        if 'content' in b:
            b['content'] = _remove_cid(b['content'])
            b['content'] = _normalize_text(b['content'])

    blocks = _remove_garbage(blocks)
    blocks = _detect_sections(blocks)
    blocks = _merge_lines(blocks)
    return blocks

def _remove_garbage(blocks: List[Dict]) -> List[Dict]:
    cleaned = []
    for b in blocks:
        if b.get('formulas') or b.get('type') in ('formula', 'equation_block', 'formula_label'):
            cleaned.append(b)
            continue
        text = b.get('content', '').strip()
        if len(text) < MIN_TEXT_LEN:
            continue
        if re.fullmatch(r"\[\d+\]", text):
            continue
        if re.fullmatch(r"\d+", text):
            continue
        if "http://" in text or "www." in text:
            continue
        if "УДК" in text or "UDC" in text:
            continue
        if re.search(r"(Cand\.|Dr\.|к\.|д\.)", text):
            continue
        if re.search(r"\b(19|20)\d{2}\b", text):
            continue
        if re.search(r"(journal|vol\.|volume|no\.|issue|pp\.|pages|doi:|elsevier|springer)", text.lower()):
            continue
        cleaned.append(b)
    return cleaned

def _detect_sections(blocks: List[Dict]) -> List[Dict]:
    for b in blocks:
        if b.get('type') not in ('paragraph', 'text'):
            continue
        text = b.get('content', '').strip()
        if re.match(r"^\d+(\.\d+)?\s+[A-ZА-Я]", text):
            b['type'] = 'section'
            b['is_section'] = True
    return blocks

def _merge_lines(blocks: List[Dict]) -> List[Dict]:
    if not blocks:
        return blocks
    merged = []
    buffer = None
    for b in blocks:
        if b.get('formulas') or b.get('type') in ('formula', 'equation_block', 'formula_label', 'section', 'title', 'authors', 'abstract'):
            if buffer is not None:
                merged.append(buffer)
                buffer = None
            merged.append(b)
            continue
        if b.get('type') not in ('paragraph', 'text'):
            if buffer is not None:
                merged.append(buffer)
                buffer = None
            merged.append(b)
            continue
        text = b.get('content', '').strip()
        if not text:
            continue
        if buffer is None:
            buffer = b.copy()
            if 'formulas' not in buffer:
                buffer['formulas'] = []
            continue
        if len(buffer['content']) + len(text) + 2 > MAX_CHUNK_LEN:
            merged.append(buffer)
            buffer = b.copy()
            continue
        if not buffer['content'].rstrip().endswith(('.', '!', '?')):
            buffer['content'] += ' ' + text
        else:
            merged.append(buffer)
            buffer = b.copy()
    if buffer is not None:
        merged.append(buffer)
    return merged

def clean_text(text: str) -> str:
    text = _remove_cid(text)
    return _normalize_text(text)