# src/retrieval/hyde.py
"""
Hypothetical Document Embeddings (HyDE) – улучшение запроса.
Генерирует гипотетический ответ на вопрос, затем использует его для поиска.
"""

import logging
from typing import List, Dict, Any, Optional

from src.agent.llm_client import LLMClient

logger = logging.getLogger(__name__)

HYDE_PROMPT = """Ты – эксперт в области математического моделирования электромембранных процессов.
Напиши подробный гипотетический ответ на следующий вопрос, как если бы ты отвечал на основе научной статьи.
Не используй фразы "я не знаю" или "недостаточно информации". Просто напиши правдоподобный ответ, содержащий ключевые термины, уравнения, названия моделей.

Вопрос: {question}

Гипотетический ответ:"""


class HyDEQueryTransformer:
    """
    Трансформер запроса с использованием HyDE.
    Генерирует гипотетический ответ и возвращает его для поиска.
    """
    def __init__(self, llm_client: LLMClient, prompt: str = HYDE_PROMPT):
        self.llm = llm_client
        self.prompt = prompt

    async def transform(self, query: str) -> str:
        """
        Генерирует гипотетический ответ на запрос.
        Возвращает текст, который будет использован для поиска.
        """
        prompt = self.prompt.format(question=query)
        hypothetical_answer = await self.llm.complete(prompt)
        logger.info(f"HyDE сгенерировал ответ длины {len(hypothetical_answer)} символов")
        return hypothetical_answer

    async def transform_with_original(self, query: str, alpha: float = 0.7) -> List[str]:
        """
        Возвращает список из двух строк: исходный запрос и гипотетический ответ.
        Затем можно объединить результаты поиска (например, через RRF).
        """
        hypothetical = await self.transform(query)
        return [query, hypothetical]