import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import ollama

sys.path.insert(0, str(Path(__file__).parent))

from src.retrieval.retriever import Retriever
from src.knowledge.formula_kb_loader import FormulaKnowledgeBase


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

INDEX_PATH = "master_index"
PDF_DIR = "data/raw"
FORMULA_KB_PATH = "data/knowledge/formulas_knowledge_base.json"

OLLAMA_MODEL = "vikhr-7b-instruct"
NUM_PREDICT = 512
TEMPERATURE = 0.0
RETRIEVAL_TOP_K = 70
CONTEXT_TOP_K = 3


SYSTEM_PROMPT = """
Ты являешься научным ассистентом в области математического моделирования электромембранных процессов.

Используй только информацию из переданного контекста.
Не добавляй сведения из внешних источников.
Не придумывай формулы, параметры, авторов и библиографические ссылки.
Если в контексте нет нужной информации, прямо сообщи об этом.
Сохраняй математические выражения без изменения обозначений и структуры записи.
Формируй ответ в научном стиле.
""".strip()


class OllamaLLM:
    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        num_predict: int = NUM_PREDICT,
        temperature: float = TEMPERATURE,
    ):
        self.model = model
        self.num_predict = num_predict
        self.temperature = temperature

    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        messages = []

        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": system_prompt,
                }
            )

        messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        loop = asyncio.get_running_loop()

        response = await loop.run_in_executor(
            None,
            lambda: ollama.chat(
                model=self.model,
                messages=messages,
                options={
                    "num_predict": self.num_predict,
                    "temperature": self.temperature,
                },
            ),
        )

        return response["message"]["content"].strip()


class ElectromembraneAgent:
    def __init__(
        self,
        retriever: Retriever,
        llm: OllamaLLM,
        formula_kb: Optional[FormulaKnowledgeBase] = None,
        retrieval_top_k: int = RETRIEVAL_TOP_K,
        context_top_k: int = CONTEXT_TOP_K,
    ):
        self.retriever = retriever
        self.llm = llm
        self.formula_kb = formula_kb
        self.retrieval_top_k = retrieval_top_k
        self.context_top_k = context_top_k

    async def answer(self, question: str) -> str:
        results = await self.retrieve(question)

        if not results:
            return (
                "В локальном корпусе документов не найдено фрагментов, "
                "достаточных для формирования ответа на данный запрос."
            )

        selected_results = results[:self.context_top_k]
        context = self.build_context(selected_results)
        prompt = self.build_prompt(question, context)

        logger.info(
            "Найдено фрагментов: %s, передано в контекст: %s",
            len(results),
            len(selected_results),
        )

        return await self.llm.complete(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
        )

    async def retrieve(self, question: str) -> list[Any]:
        method_names = (
            "hybrid_search",
            "search",
            "retrieve",
            "get_relevant_chunks",
        )

        for method_name in method_names:
            method = getattr(self.retriever, method_name, None)

            if method is None:
                continue

            result = await self.call_retriever_method(method, question)
            return list(result) if result else []

        raise AttributeError(
            "В Retriever не найден метод hybrid_search, search, retrieve или get_relevant_chunks."
        )

    async def call_retriever_method(self, method, question: str):
        try:
            result = method(question, top_k=self.retrieval_top_k)
        except TypeError:
            try:
                result = method(question, k=self.retrieval_top_k)
            except TypeError:
                result = method(question)

        if asyncio.iscoroutine(result):
            result = await result

        return result

    def build_prompt(self, question: str, context: str) -> str:
        return f"""
Вопрос пользователя:
{question}

Контекст из локального корпуса документов:
{context}

Сформируй ответ на основе приведённого контекста. Если в контексте есть формула, приведи её и поясни обозначения. В конце ответа укажи использованные источники.
""".strip()

    def build_context(self, results: list[Any]) -> str:
        context_parts = []

        for index, result in enumerate(results, start=1):
            text = self.extract_text(result)
            source = self.extract_source(result)

            context_parts.append(
                f"[{index}] {source}\n{text}"
            )

        return "\n\n---\n\n".join(context_parts)

    def extract_text(self, item: Any) -> str:
        for key in (
            "content",
            "text",
            "fragment",
            "chunk",
            "chunk_text",
            "search_text",
            "page_content",
        ):
            value = self.get_value(item, key)

            if value:
                return str(value).strip()

        return str(item).strip()

    def extract_source(self, item: Any) -> str:
        citation = self.get_value(item, "citation")
        metadata = self.get_value(item, "metadata")

        source_data = citation or metadata or item

        title = self.get_nested_value(source_data, "title") or self.get_nested_value(source_data, "source")
        authors = self.get_nested_value(source_data, "authors") or self.get_nested_value(source_data, "author")
        year = self.get_nested_value(source_data, "year")
        page = self.get_nested_value(source_data, "page") or self.get_nested_value(source_data, "pages")
        section = self.get_nested_value(source_data, "section")

        parts = []

        if title:
            parts.append(f"Источник: {title}")

        if authors:
            parts.append(f"Авторы: {authors}")

        if year:
            parts.append(f"Год: {year}")

        if page:
            parts.append(f"Страница: {page}")

        if section:
            parts.append(f"Раздел: {section}")

        return "\n".join(parts) if parts else "Источник не указан"

    def get_value(self, item: Any, key: str):
        if isinstance(item, dict):
            return item.get(key)

        return getattr(item, key, None)

    def get_nested_value(self, item: Any, key: str):
        if isinstance(item, dict):
            return item.get(key)

        return getattr(item, key, None)


async def main():
    if Path(f"{INDEX_PATH}_text.faiss").exists():
        logger.info("Загрузка индекса из %s", INDEX_PATH)
        retriever = Retriever(index_path=INDEX_PATH)
    else:
        logger.info("Индекс не найден, выполняется построение из %s", PDF_DIR)

        from src.ingestion.processor import build_index_from_directory

        retriever = build_index_from_directory(
            PDF_DIR,
            index_name=INDEX_PATH,
        )

    if not retriever:
        logger.error("Не удалось загрузить или построить индекс.")
        return

    formula_kb = None

    if Path(FORMULA_KB_PATH).exists():
        formula_kb = FormulaKnowledgeBase(FORMULA_KB_PATH)

        logger.info(
            "База формул загружена: %s записей",
            len(formula_kb.formulas),
        )
    else:
        logger.warning(
            "Файл базы формул не найден: %s. Работа продолжается без базы формул.",
            FORMULA_KB_PATH,
        )

    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        num_predict=NUM_PREDICT,
        temperature=TEMPERATURE,
    )

    agent = ElectromembraneAgent(
        retriever=retriever,
        llm=llm,
        formula_kb=formula_kb,
    )

    print("\n=== Агент для математического моделирования электромембранных процессов ===\n")
    print("Введите вопрос или 'exit' для выхода.")

    while True:
        question = input("\n> ").strip()

        if question.lower() in ("exit", "quit"):
            break

        if not question:
            continue

        print("\nОбработка запроса...")

        answer = await agent.answer(question)

        print("\n--- ОТВЕТ ---\n")
        print(answer)
        print("\n" + "-" * 50)


if __name__ == "__main__":
    asyncio.run(main())
