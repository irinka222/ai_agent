import litellm
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class LLMClient:
    """Асинхронный клиент для вызовов LLM через litellm."""
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    async def complete(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Простой вызов без инструментов."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = await litellm.acompletion(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM completion error: {e}")
            return "Ошибка при вызове LLM."

    async def select_tool(self, prompt: str, tools_desc: List[Dict], system_prompt: Optional[str] = None) -> str:
        """
        Выбирает название инструмента на основе описания.
        tools_desc: список словарей вида {"name": "...", "description": "..."}
        Возвращает имя выбранного инструмента (или пустую строку).
        """
        tools_text = "\n".join([f"- {t['name']}: {t['description']}" for t in tools_desc])
        selection_prompt = f"""
{prompt}

Доступные инструменты:
{tools_text}

Твой ответ должен содержать только название одного инструмента (одно слово). Если ни один не подходит, напиши "complete".
"""
        response = await self.complete(selection_prompt, system_prompt)
        # Очищаем ответ
        selected = response.strip().split()[0].lower()
        # Проверяем, что выбранный инструмент существует
        valid_names = [t['name'].lower() for t in tools_desc] + ["complete"]
        if selected not in valid_names:
            return "complete"
        return selected