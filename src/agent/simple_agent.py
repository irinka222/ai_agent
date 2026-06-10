import logging
from src.agent.env import Environment
from src.knowledge.formula_kb_loader import FormulaKnowledgeBase

logger = logging.getLogger(__name__)

class SimpleAgent:
    def __init__(self, env: Environment):
        self.env = env
        self.formula_kb = FormulaKnowledgeBase()   # загружает базу

    async def run(self, question: str, top_k: int = 5) -> str:
        await self.env.reset(question)
        # 1. Поиск текстовых чанков
        await self.env.step("search", query=question, top_k=top_k)

        # 2. Поиск формул по базе знаний (если ещё не добавлены в state)
        if not hasattr(self.env.state, 'formula_context'):
            relevant_formulas = self.formula_kb.search_by_text(question, top_k=3)
            if relevant_formulas:
                formula_context = "\n\n".join(
                    self.formula_kb.format_for_prompt(f) for f in relevant_formulas
                )
                self.env.state.formula_context = formula_context
                logger.info(f"Found {len(relevant_formulas)} relevant formulas")

        # 3. Сбор текстовых чанков
        await self.env.step("gather")
        # 4. Генерация ответа (учитывает формулы)
        answer = await self.env.step("generate")
        # 5. Завершение
        await self.env.step("complete")
        return self.env.state.answer