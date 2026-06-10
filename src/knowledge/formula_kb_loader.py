import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional


class FormulaKnowledgeBase:
    def __init__(self, json_path: str = "data/knowledge/formulas_knowledge_base.json"):
        self.json_path = Path(json_path)
        self.formulas: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if not self.json_path.exists():
            raise FileNotFoundError(f"Formula KB not found: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.formulas = data.get("formulas", [])

    def search_by_text(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        scored = []
        for f in self.formulas:
            score = 0
            if f.get("number"):
                if f["number"].lower() in query_lower:
                    score += 10
            if f.get("description"):
                desc_words = set(f["description"].lower().split())
                if any(word in query_lower for word in desc_words):
                    score += 5
            if f.get("context"):
                if any(word in query_lower for word in f["context"].lower().split()):
                    score += 3
            if f.get("variables"):
                for var, desc in f["variables"].items():
                    if var.lower() in query_lower or desc.lower() in query_lower:
                        score += 2
            if f.get("latex"):
                latex_clean = re.sub(r"[^a-zA-Z0-9]", "", f["latex"].lower())
                query_clean = re.sub(r"[^a-zA-Z0-9]", "", query_lower)
                if query_clean in latex_clean:
                    score += 4
            if score > 0:
                scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:top_k]]

    def get_by_id(self, formula_id: str) -> Optional[Dict[str, Any]]:
        for f in self.formulas:
            if f.get("id") == formula_id:
                return f
        return None

    def get_by_number(self, number: str) -> Optional[Dict[str, Any]]:
        for f in self.formulas:
            if f.get("number") == number:
                return f
        return None

    def format_for_prompt(self, formula: Dict[str, Any]) -> str:
        parts = []
        if formula.get("number"):
            parts.append(f"Формула {formula['number']}:")
        if formula.get("latex"):
            parts.append(f"`{formula['latex']}`")
        if formula.get("description"):
            parts.append(f"Описание: {formula['description']}")
        if formula.get("variables"):
            vars_str = ", ".join([f"{k}: {v}" for k, v in formula["variables"].items()])
            parts.append(f"Переменные: {vars_str}")
        if formula.get("context"):
            parts.append(f"Источник: {formula['context']}")
        return "\n".join(parts)