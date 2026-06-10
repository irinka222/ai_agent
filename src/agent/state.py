from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from uuid import uuid4


@dataclass
class Evidence:
    """
    Один найденный фрагмент-доказательство,
    который передаётся агенту и используется для цитирования.
    """

    chunk_id: str
    text: str

    formulas: List[Dict[str, Any]] = field(default_factory=list)
    score: float = 1.0

    source_paper: str = ""
    title: str = ""
    year: Optional[str] = None
    authors: List[str] = field(default_factory=list)

    section: Optional[str] = None
    page: Optional[int] = None
    pages: Optional[Any] = None


@dataclass
class AgentState:
    """
    Состояние одного запроса пользователя.
    Сейчас используется как вспомогательная структура,
    но может быть расширена для многошагового агента.
    """

    question: str

    evidences: List[Evidence] = field(default_factory=list)
    answer: str = ""

    is_successful: Optional[bool] = None
    step_history: List[str] = field(default_factory=list)

    total_cost: float = 0.0
    session_id: str = field(default_factory=lambda: str(uuid4()))

    def add_evidence(self, evidence: Evidence) -> None:
        """
        Добавляет evidence, если такого chunk_id ещё нет.
        """
        if not any(item.chunk_id == evidence.chunk_id for item in self.evidences):
            self.evidences.append(evidence)

    def get_top_evidences(self, limit: int = 5) -> List[Evidence]:
        """
        Возвращает top-N evidence по score.
        """
        sorted_evidences = sorted(
            self.evidences,
            key=lambda item: item.score,
            reverse=True,
        )
        return sorted_evidences[:limit]

    def format_context(self, limit: int = 5) -> str:
        """
        Формирует текстовый контекст для промпта.
        """
        evidences = self.get_top_evidences(limit)
        context_parts = []

        for i, evidence in enumerate(evidences, 1):
            authors_str = (
                ", ".join(evidence.authors)
                if evidence.authors
                else "автор неизвестен"
            )

            year_str = f" ({evidence.year})" if evidence.year else ""
            title_str = f" *{evidence.title}*" if evidence.title else ""

            context = f"[{i}] {authors_str}{year_str}.{title_str}\n"
            context += f"Источник: {evidence.source_paper}"

            if evidence.section:
                context += f", секция: {evidence.section}"

            if evidence.page:
                context += f", стр. {evidence.page}"

            if evidence.pages:
                context += f", страницы: {evidence.pages}"

            context += f"\n{evidence.text}\n"

            if evidence.formulas:
                formulas = []

                for formula in evidence.formulas:
                    if not isinstance(formula, dict):
                        continue

                    latex = formula.get("latex", "")
                    description = (
                        formula.get("semantic_description")
                        or formula.get("description")
                        or ""
                    )

                    if latex and description:
                        formulas.append(f"{latex} — {description}")
                    elif latex:
                        formulas.append(latex)

                if formulas:
                    context += "Формулы:\n"
                    for formula_text in formulas:
                        context += f"- {formula_text}\n"

            context_parts.append(context)

        return "\n\n".join(context_parts)

    def add_step(self, message: str) -> None:
        """
        Добавляет запись в историю шагов агента.
        """
        self.step_history.append(message)

    def mark_success(self, answer: str) -> None:
        """
        Отмечает успешное завершение запроса.
        """
        self.answer = answer
        self.is_successful = True

    def mark_failed(self, message: str = "") -> None:
        """
        Отмечает неуспешное завершение запроса.
        """
        if message:
            self.answer = message

        self.is_successful = False