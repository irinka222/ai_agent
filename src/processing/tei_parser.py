# src/processing/tei_parser.py
# -*- coding: utf-8 -*-

import re
import logging
from typing import List, Dict, Any, Optional
from lxml import etree as ET

from src.processing.formula_extractor import build_formula_candidate

logger = logging.getLogger(__name__)

NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def get_text(el) -> str:
    return "".join(el.itertext()).strip() if el is not None else ""


def parse_tei(xml: str, paper_id: str) -> List[Dict[str, Any]]:
    if not xml:
        return []

    try:
        root = ET.fromstring(xml.encode("utf-8"))
    except Exception as e:
        logger.error("TEI parse error: %s", e)
        return []

    blocks: List[Dict[str, Any]] = []

    metadata = extract_metadata(root, paper_id)
    blocks.extend(metadata_to_blocks(metadata, paper_id))

    current_section = None

    body = root.find(".//tei:body", NS)
    if body is None:
        body = root

    for div in body.findall(".//tei:div", NS):
        head = div.find("tei:head", NS)

        if head is not None:
            current_section = get_text(head)
            if current_section:
                blocks.append({
                    "type": "section",
                    "content": current_section,
                    "section": current_section,
                    "source": paper_id,
                    "paper_id": paper_id,
                    "page": None,
                })

        for elem in div:
            local_name = strip_namespace(elem.tag)

            if local_name == "head":
                continue

            if local_name == "p":
                text = get_text(elem)
                if text:
                    blocks.append({
                        "type": "paragraph",
                        "content": text,
                        "section": current_section,
                        "source": paper_id,
                        "paper_id": paper_id,
                        "page": get_page(elem),
                        "formulas": [],
                    })

            elif local_name == "formula":
                formula_block = parse_formula_element(
                    elem=elem,
                    paper_id=paper_id,
                    section=current_section,
                )
                if formula_block:
                    blocks.append(formula_block)

            elif local_name in {"figure", "table"}:
                caption = extract_caption(elem)
                if caption:
                    blocks.append({
                        "type": local_name,
                        "content": caption,
                        "section": current_section,
                        "source": paper_id,
                        "paper_id": paper_id,
                        "page": get_page(elem),
                    })

    return blocks


def extract_metadata(root, paper_id: str) -> Dict[str, Any]:
    title = get_text(root.find(".//tei:titleStmt/tei:title", NS))

    authors = []
    for author in root.findall(".//tei:titleStmt/tei:author", NS):
        name = get_text(author)
        if name and len(name) < 120:
            authors.append(name)

    if not authors:
        for contrib in root.findall(".//tei:titleStmt/tei:contributor", NS):
            name = get_text(contrib)
            if name and len(name) < 120:
                authors.append(name)

    year = None
    for date_elem in root.findall(".//tei:date", NS):
        date_str = get_text(date_elem)
        match = re.search(r"\b(19\d{2}|20\d{2})\b", date_str)
        if match:
            year = match.group(1)
            break

    abstract = get_text(root.find(".//tei:abstract", NS))

    return {
        "paper_id": paper_id,
        "title": title,
        "authors": authors,
        "year": year,
        "abstract": abstract,
    }


def metadata_to_blocks(metadata: Dict[str, Any], paper_id: str) -> List[Dict[str, Any]]:
    blocks = []

    if metadata.get("title"):
        blocks.append({
            "type": "title",
            "content": metadata["title"],
            "section": "metadata",
            "source": paper_id,
            "paper_id": paper_id,
            "page": None,
        })

    if metadata.get("authors"):
        blocks.append({
            "type": "authors",
            "authors": metadata["authors"],
            "content": ", ".join(metadata["authors"]),
            "section": "metadata",
            "source": paper_id,
            "paper_id": paper_id,
            "page": None,
        })

    if metadata.get("year"):
        blocks.append({
            "type": "publication_date",
            "content": metadata["year"],
            "section": "metadata",
            "source": paper_id,
            "paper_id": paper_id,
            "page": None,
        })

    if metadata.get("abstract"):
        blocks.append({
            "type": "abstract",
            "content": metadata["abstract"],
            "section": "abstract",
            "source": paper_id,
            "paper_id": paper_id,
            "page": None,
        })

    return blocks


def parse_formula_element(elem, paper_id: str, section: Optional[str]) -> Optional[Dict[str, Any]]:
    latex = (
        elem.attrib.get("tex")
        or elem.attrib.get("latex")
        or elem.attrib.get("formula")
        or get_text(elem)
    )

    label = elem.attrib.get("label") or elem.attrib.get("n")
    page = get_page(elem)

    candidate = build_formula_candidate(
        latex=latex,
        source="grobid",
        page=page,
        label=label,
    )

    if not candidate:
        return None

    candidate["source"] = "grobid"
    candidate["paper_id"] = paper_id
    candidate["section"] = section
    candidate["content"] = candidate.get("latex", "")

    return candidate


def extract_caption(elem) -> str:
    head = elem.find(".//tei:head", NS)
    if head is not None:
        return get_text(head)

    fig_desc = elem.find(".//tei:figDesc", NS)
    if fig_desc is not None:
        return get_text(fig_desc)

    return ""


def get_page(elem) -> Optional[int]:
    for key in ["page", "n", "from", "{http://www.w3.org/XML/1998/namespace}id"]:
        value = elem.attrib.get(key)
        if not value:
            continue

        match = re.search(r"\d+", value)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                pass

    return None


def strip_namespace(tag: str) -> str:
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag