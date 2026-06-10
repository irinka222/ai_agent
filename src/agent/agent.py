import logging
from src.agent.env import Environment

logger = logging.getLogger(__name__)

class SimpleAgent:
    """Агент с фиксированной последовательностью действий."""
    def __init__(self, env: Environment):
        self.env = env

    async def run(self, question: str, top_k: int = 5) -> str:
        await self.env.reset(question)

        # 1. Поиск
        result = await self.env.step("search", query=question, top_k=top_k)
        logger.info(f"Search: {result}")

        # 2. Сбор (пока заглушка)
        result = await self.env.step("gather")
        logger.info(f"Gather: {result}")

        # 3. Генерация ответа
        answer = await self.env.step("generate")
        logger.info(f"Answer: {answer}")

        # 4. Завершение
        await self.env.step("complete")
        return self.env.state.answer