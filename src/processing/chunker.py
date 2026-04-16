import uuid
import logging
from typing import List, Dict

from src.config import MAX_CHUNK_SIZE, WINDOW_SIZE

logger = logging.getLogger(__name__)


def _new_chunk(section: str = None) -> Dict:
    """Создаёт пустой чанк с базовой структурой."""
    return {
        "id": str(uuid.uuid4()),
        "section": section,
        "content": "",
        "formulas": [],
        "metadata": {"has_formula": False},
    }


def _finalize_chunk(chunk: Dict, paper_id: str, authors: List[str]) -> Dict | None:
    """
    Завершает чанк: очищает content, добавляет метаданные, citation, search_text.
    Возвращает None, если чанк пуст.
    """
    chunk["content"] = chunk["content"].strip()
    if not chunk["content"] and not chunk["formulas"]:
        return None
    if not chunk["content"] and chunk["formulas"]:
        chunk["content"] = "[Formula(s) without surrounding text]"
        chunk["metadata"]["is_formula_only"] = True
    if chunk["formulas"]:
        chunk["metadata"]["has_formula"] = True
    chunk["chunk_id"] = f"{paper_id}_{chunk['id'][:8]}"
    chunk["citation"] = {
        "authors": authors,
        "paper_id": paper_id,
        "section": chunk.get("section"),
    }
    chunk["source"] = paper_id
    # Универсальное поле для поиска
    chunk["search_text"] = chunk["content"]
    if chunk["formulas"]:
        chunk["search_text"] += " " + " ".join(f.get("latex", "") for f in chunk["formulas"])
    return chunk


def _add_context_windows(chunks: List[Dict]) -> List[Dict]:
    """Добавляет контекстные окна (предыдущий/следующий чанк)."""
    for i, ch in enumerate(chunks):
        prev_texts = []
        for j in range(max(0, i - WINDOW_SIZE), i):
            if "content" in chunks[j] and chunks[j]["content"]:
                prev_texts.append(chunks[j]["content"])
        next_texts = []
        for j in range(i + 1, min(len(chunks), i + 1 + WINDOW_SIZE)):
            if "content" in chunks[j] and chunks[j]["content"]:
                next_texts.append(chunks[j]["content"])
        ch["context_window"] = {
            "prev": " ".join(prev_texts),
            "next": " ".join(next_texts),
        }
    return chunks


def chunk_blocks(blocks: List[Dict], paper_id: str, authors: List[str]) -> List[Dict]:
    """
    Основная функция: разбивает блоки на чанки.
    Стратегия:
      1. Текстовые блоки (paragraph) без формул → обычные чанки с ограничением MAX_CHUNK_SIZE.
      2. Блоки с формулами (paragraph с formulas или equation_block) → каждая формула в свой чанк,
         вместе с окружающим текстом (контекстом).
    """
    chunks = []
    current = _new_chunk()          # текущий текстовый чанк (без формул)
    current_section = None

    for block in blocks:
        btype = block.get("type")

        # ----- Обработка смены секции -----
        if btype == "section":
            current_section = block.get("content", "")
            if current["content"] or current["formulas"]:
                finalized = _finalize_chunk(current, paper_id, authors)
                if finalized:
                    chunks.append(finalized)
                current = _new_chunk()
            current["section"] = current_section
            continue

        # ----- Обычный параграф (без формул или с формулами) -----
        if btype == "paragraph":
            text = block.get("content", "").strip()
            formulas = block.get("formulas", [])

            # Если в блоке есть формулы → каждая формула в отдельный чанк
            if formulas:
                # Сначала закрываем текущий текстовый чанк, если он не пуст
                if current["content"]:
                    finalized = _finalize_chunk(current, paper_id, authors)
                    if finalized:
                        chunks.append(finalized)
                    current = _new_chunk(section=current_section)

                # Для каждой формулы создаём отдельный чанк
                for f in formulas:
                    latex = f.get("latex", "")
                    raw = f.get("raw", "")
                    label = f.get("label", "")
                    confidence = f.get("confidence", 0.0)
                    source = f.get("source", "heuristic")
                    ast = f.get("ast", False)

                    formula_text = f"<FORMULA>{latex}</FORMULA>"
                    if label:
                        formula_text = f"<FORMULA label='{label}'>{latex}</FORMULA>"

                    # Содержимое чанка: контекст (весь текст параграфа) + формула
                    chunk_content = f"{text}\n{formula_text}" if text else formula_text

                    formula_obj = {
                        "latex": latex,
                        "normalized": latex,
                        "raw": raw,
                        "source": source,
                        "confidence": confidence,
                        "ast": ast,
                        "label": label,
                        "semantic_description": text[:300] if text else "",
                        "context_inline": formula_text,
                    }

                    formula_chunk = _new_chunk(section=current_section)
                    formula_chunk["content"] = chunk_content
                    formula_chunk["formulas"] = [formula_obj]
                    formula_chunk["metadata"]["is_formula_only"] = True
                    formula_chunk["search_text"] = f"{text} {latex}"

                    finalized = _finalize_chunk(formula_chunk, paper_id, authors)
                    if finalized:
                        chunks.append(finalized)

                # После обработки формул текст параграфа уже использован, переходим к следующему блоку
                continue

            # --- Обычный текст без формул ---
            if not text:
                continue

            # Если добавление текста превысит лимит – закрываем текущий чанк и начинаем новый
            if len(current["content"]) + len(text) + 1 > MAX_CHUNK_SIZE:
                finalized = _finalize_chunk(current, paper_id, authors)
                if finalized:
                    chunks.append(finalized)
                current = _new_chunk(section=current_section)

            # Добавляем текст
            current["content"] += (" " + text) if current["content"] else text
            continue

        # ----- Готовый equation_block (из normalizer) -----
        if btype == "equation_block" and block.get("latex"):
            # Закрываем текущий текстовый чанк
            if current["content"]:
                finalized = _finalize_chunk(current, paper_id, authors)
                if finalized:
                    chunks.append(finalized)
                current = _new_chunk(section=current_section)

            latex = block.get("latex")
            context = block.get("context_before") or block.get("description") or ""
            formula_text = f"<FORMULA>{latex}</FORMULA>"
            chunk_content = f"{context}\n{formula_text}" if context else formula_text

            formula_obj = {
                "latex": latex,
                "normalized": latex,
                "raw": block.get("raw", ""),
                "source": block.get("source", "grobid"),
                "confidence": block.get("confidence", 0.8),
                "ast": block.get("ast", False),
                "label": block.get("label", ""),
                "semantic_description": context[:300],
                "context_inline": formula_text,
            }

            formula_chunk = _new_chunk(section=current_section)
            formula_chunk["content"] = chunk_content
            formula_chunk["formulas"] = [formula_obj]
            formula_chunk["metadata"]["is_formula_only"] = True
            formula_chunk["search_text"] = f"{context} {latex}"

            finalized = _finalize_chunk(formula_chunk, paper_id, authors)
            if finalized:
                chunks.append(finalized)
            continue

        # Другие типы блоков (таблицы, списки и т.д.) – пока пропускаем или можно добавить как текст
        # Для простоты игнорируем

    # Последний незавершённый чанк
    if current["content"] or current["formulas"]:
        finalized = _finalize_chunk(current, paper_id, authors)
        if finalized:
            chunks.append(finalized)

    # Добавляем контекстные окна
    chunks = _add_context_windows(chunks)
    return chunks