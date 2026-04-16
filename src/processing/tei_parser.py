import re
import logging
from lxml import etree as ET
from typing import List, Dict, Any
from src.processing.formula_extractor import is_valid_formula, parse_ast_safe, compute_confidence

logger = logging.getLogger(__name__)
NS = {"tei": "http://www.tei-c.org/ns/1.0"}

def get_text(el):
    return "".join(el.itertext()).strip() if el is not None else ""

def parse_tei(xml: str, paper_id: str) -> List[Dict[str, Any]]:
    if not xml:
        return []
    try:
        root = ET.fromstring(xml.encode('utf-8'))
    except Exception as e:
        logger.error(f"TEI parse error: {e}")
        return []

    blocks = []
    title = root.find(".//tei:titleStmt/tei:title", NS)
    if title is not None:
        blocks.append({"type": "title", "content": get_text(title)})

    authors = []
    for author in root.findall(".//tei:titleStmt/tei:author", NS):
        name = get_text(author)
        if name and len(name) < 100 and not re.search(r'\d|\[|\]', name):
            authors.append(name)
    if authors:
        blocks.append({"type": "authors", "authors": authors})

    abstract = root.find(".//tei:abstract", NS)
    if abstract is not None:
        blocks.append({"type": "abstract", "content": get_text(abstract)})

    current_section = None
    for div in root.findall(".//tei:div", NS):
        head = div.find("tei:head", NS)
        if head is not None:
            current_section = get_text(head)
            blocks.append({"type": "section", "content": current_section})

        for elem in div:
            tag = elem.tag
            if tag.endswith('p'):
                text = get_text(elem)
                if text:
                    blocks.append({"type": "paragraph", "content": text, "section": current_section})
            elif tag.endswith('formula'):
                raw = get_text(elem)
                latex = elem.attrib.get('tex', raw)
                if not is_valid_formula(latex):
                    continue
                ast = parse_ast_safe(latex)
                confidence = compute_confidence(latex, ast, source='grobid')
                if confidence < 0.3:
                    continue
                blocks.append({
                    "type": "formula",
                    "content": raw,
                    "latex": latex,
                    "raw": raw,
                    "source": "grobid",
                    "confidence": confidence,
                    "ast": ast is not None,
                    "section": current_section
                })
            elif tag.endswith('label'):
                if elem.getparent() is not None and elem.getparent().tag.endswith('formula'):
                    continue
                blocks.append({"type": "formula_label", "content": get_text(elem), "section": current_section})
            elif tag.endswith('table'):
                rows = []
                for row in elem.findall(".//tei:row", NS):
                    cells = [get_text(c) for c in row.findall(".//tei:cell", NS)]
                    if cells:
                        rows.append(cells)
                if rows:
                    blocks.append({"type": "table", "content": str(rows), "section": current_section})
    return blocks