from typing import Any
from src.agent.state import AgentState
from src.retrieval.retriever import Retriever
from src.agent.llm_client import LLMClient
from src.knowledge.formula_kb_loader import FormulaKnowledgeBase


class SearchTool:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    async def execute(self, query: str, top_k: int, state: AgentState) -> str:
        results = await self.retriever.search(query, top_k)
        state.raw_results = results
        return f"Найдено {len(results)} релевантных фрагментов."


class GatherTool:
    async def execute(self, state: AgentState) -> str:
        if not state.raw_results:
            state.gathered_chunks = "Нет данных для сбора."
            return "Нет данных для сбора."

        chunks = []
        for res in state.raw_results:
            chunks.append(res.get("content", ""))
        state.gathered_chunks = "\n\n---\n\n".join(chunks)
        return f"Собрано {len(chunks)} фрагментов."


class GenerateTool:
    def __init__(self, llm_client: LLMClient, system_prompt: str, formula_kb: FormulaKnowledgeBase = None):
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.formula_kb = formula_kb

    async def execute(self, state: AgentState) -> str:
        # Получаем контекст формул
        formula_context = getattr(state, 'formula_context', '')
        if not formula_context and self.formula_kb:
            relevant = self.formula_kb.search_by_text(state.question, top_k=3)
            if relevant:
                formula_context = "\n\n".join(
                    self.formula_kb.format_for_prompt(f) for f in relevant
                )
        prompt = f"""{self.system_prompt}

Вопрос: {state.question}

Найденные фрагменты текста:
{state.gathered_chunks}

Найденные формулы (из базы знаний):
{formula_context or 'Нет релевантных формул.'}

Ответь, используя приведённые данные.
"""
        state.answer = self.llm.generate(prompt)
        return state.answer


class CompleteTool:
    async def execute(self, state: AgentState) -> str:
        return "Завершено."