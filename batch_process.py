import sys
sys.path.insert(0, 'src')

import os
import glob
import logging
from src.config import LOG_LEVEL
from src.ingestion.pdf_parser import unified_parse_pdf
from src.processing.cleaner import clean_blocks
from src.processing.normalizer import normalize_blocks
from src.processing.chunker import chunk_blocks
from src.processing.formula_extractor import is_valid_formula
from src.retrieval.retriever import Retriever

logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

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

def process_single_pdf(pdf_path: str):
    """Обрабатывает один PDF и возвращает список чанков."""
    logger.info(f"Processing {pdf_path}...")
    all_blocks = unified_parse_pdf(pdf_path)
    if not all_blocks:
        logger.warning(f"No content extracted from {pdf_path}")
        return []

    all_blocks = attach_formulas(all_blocks)
    logger.debug(f"After attach: {len(all_blocks)} blocks, formulas: {count_formulas(all_blocks)}")

    cleaned = clean_blocks(all_blocks)
    normalized = normalize_blocks(cleaned)
    # Извлекаем авторов (можно пропустить или собрать позже)
    authors = []
    for b in normalized:
        if b.get('type') == 'authors':
            authors = b.get('authors', [])
            break

    paper_id = os.path.basename(pdf_path).replace('.pdf', '')
    chunks = chunk_blocks(normalized, paper_id=paper_id, authors=authors)
    logger.info(f"Chunks: {len(chunks)}, formulas: {sum(len(c.get('formulas', [])) for c in chunks)}")
    return chunks

def batch_process(input_dir: str, index_name: str = "master_index"):
    all_chunks = []
    # Ищем все PDF в директории (рекурсивно)
    pdf_files = glob.glob(os.path.join(input_dir, "**", "*.pdf"), recursive=True)
    logger.info(f"Found {len(pdf_files)} PDF files")
    for pdf_path in pdf_files:
        chunks = process_single_pdf(pdf_path)
        all_chunks.extend(chunks)
        # Можно периодически сохранять промежуточный индекс, но для простоты – в конце

    logger.info(f"Total chunks collected: {len(all_chunks)}")
    logger.info(f"Building master index...")
    retriever = Retriever()
    retriever.index_chunks(all_chunks)
    retriever.save(index_name)
    logger.info(f"Index saved to {index_name}_text.faiss, {index_name}_formula.faiss, {index_name}_metadata.pkl")

    # Опционально: сохранить все чанки в JSON для отладки
    import json
    with open("all_chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info("All chunks saved to all_chunks.json")
    return retriever

if __name__ == "__main__":
    input_directory = "data/raw/collection_2"  # измените на вашу папку
    retriever = batch_process(input_directory)
    # Теперь можно использовать retriever для поиска
    # Например, интерактивный режим:
    while True:
        query = input("\nВведите запрос (или 'exit'): ")
        if query.lower() == 'exit':
            break
        results = retriever.retrieve(query, top_k=5)
        print(f"Found {len(results)} relevant chunks:")
        for i, ch in enumerate(results):
            print(f"\n--- Result {i+1} ---")
            print(f"Source: {ch.get('source')}, Section: {ch.get('section')}")
            print(f"Content: {ch['content'][:300]}...")
            if ch.get('formulas'):
                print(f"Formulas: {[f['latex'] for f in ch['formulas']]}")