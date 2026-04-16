import sys
sys.path.insert(0, 'src')

import logging
from src.config import LOG_LEVEL
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(levelname)s:%(name)s:%(message)s')

from src.ingestion.pdf_parser import unified_parse_pdf
from src.processing.cleaner import clean_blocks
from src.processing.normalizer import normalize_blocks
from src.processing.chunker import chunk_blocks
from src.processing.formula_extractor import is_valid_formula
from src.retrieval.retriever import Retriever

def attach_formulas(blocks):
    merged = []
    current_para = None
    for b in blocks:
        if b.get('type') == 'formula':
            latex = b.get('latex', '')
            if not is_valid_formula(latex):
                continue
            if current_para and current_para.get('section') == b.get('section'):
                current_para.setdefault('formulas', []).append(b)
            else:
                merged.append({
                    'type': 'equation_block',
                    'formulas': [b],
                    'content': '',
                    'latex': latex,
                    'section': b.get('section')
                })
            continue
        current_para = b.copy()
        current_para.setdefault('formulas', [])
        merged.append(current_para)
    return merged

def count_formulas(blocks):
    total = 0
    for b in blocks:
        total += len(b.get('formulas', []))
        if b.get('type') == 'formula':
            total += 1
    return total

def process_pdf(pdf_path: str, index_name: str = "my_index"):
    print(f"Processing {pdf_path}...")
    all_blocks = unified_parse_pdf(pdf_path)
    if not all_blocks:
        print("No content extracted")
        return None

    all_blocks = attach_formulas(all_blocks)
    print(f"After attach: {len(all_blocks)} blocks, formulas: {count_formulas(all_blocks)}")

    cleaned = clean_blocks(all_blocks)
    print(f"Cleaner: {len(cleaned)} blocks, formulas: {count_formulas(cleaned)}")

    normalized = normalize_blocks(cleaned)
    print(f"Normalizer: {len(normalized)} blocks, formulas: {count_formulas(normalized)}")

    authors = []
    for b in normalized:
        if b.get('type') == 'authors':
            authors = b.get('authors', [])
            break

    paper_id = pdf_path.split('/')[-1].replace('.pdf', '')
    chunks = chunk_blocks(normalized, paper_id=paper_id, authors=authors)
    print(f"Chunker: {len(chunks)} chunks")
    total_f = sum(len(c.get('formulas', [])) for c in chunks)
    chunks_with_f = sum(1 for c in chunks if c.get('formulas'))
    print(f"Total formulas in chunks: {total_f}, chunks with formulas: {chunks_with_f}/{len(chunks)}")

    if chunks:
        example = next((c for c in chunks if c.get('formulas')), chunks[0])
        print("\n--- Example Chunk ---")
        print(f"Section: {example.get('section')}")
        print(f"Content preview: {example['content'][:300]}...")
        print(f"Formulas count: {len(example.get('formulas', []))}")
        if example.get('formulas'):
            print(f"First formula latex: {example['formulas'][0].get('latex', '')[:100]}")
        print(f"Citation: {example.get('citation')}")

    print("\nBuilding search index...")
    retriever = Retriever()
    retriever.index_chunks(chunks)
    retriever.save(index_name)
    print(f"Index saved to {index_name}_text.faiss, {index_name}_formula.faiss")

    import json
    with open("chunks_output.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print("\nChunks saved to chunks_output.json")
    return chunks, retriever

if __name__ == "__main__":
    pdf_path = "data/raw/collection_2/baulinamm_+Journal+of+the+Belarusian+State+University_+Mathemat.pdf"
    chunks, retriever = process_pdf(pdf_path)