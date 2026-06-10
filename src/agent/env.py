from typing import Dict, Any
from src.agent.state import AgentState
from src.agent.tools import SearchTool, GatherTool, GenerateTool, CompleteTool
from src.retrieval.retriever import Retriever
from src.agent.llm_client import LLMClient
from src.knowledge.formula_kb_loader import FormulaKnowledgeBase


class Environment:
    def __init__(self, retriever: Retriever, llm_client: LLMClient, system_prompt: str, formula_kb: FormulaKnowledgeBase = None):
        self.retriever = retriever
        self.llm = llm_client
        self.system_prompt = system_prompt
        self.formula_kb = formula_kb
        self.state: AgentState | None = None
        self.tools = self._create_tools()

    def _create_tools(self) -> Dict[str, Any]:
        return {
            "search": SearchTool(self.retriever),
            "gather": GatherTool(),
            "generate": GenerateTool(self.llm, self.system_prompt, self.formula_kb),
            "complete": CompleteTool(),
        }

    async def reset(self, question: str) -> AgentState:
        self.state = AgentState(question=question)
        return self.state

    async def step(self, action_name: str, **kwargs) -> str:
        tool = self.tools.get(action_name)
        if not tool:
            return f"Неизвестный инструмент: {action_name}"
        if action_name == "search":
            query = kwargs.get("query", self.state.question)
            top_k = kwargs.get("top_k", 5)
            return await tool.execute(query, top_k, self.state)
        elif action_name == "gather":
            return await tool.execute(self.state)
        elif action_name == "generate":
            return await tool.execute(self.state)
        elif action_name == "complete":
            return await tool.execute(self.state)
        else:
            return f"Инструмент {action_name} не реализован"