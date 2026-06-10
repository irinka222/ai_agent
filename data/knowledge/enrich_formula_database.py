# src/knowledge/enrich_formula_database.py
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path
from typing import Dict, Any, List


def classify_formula(formula: Dict[str, Any]) -> str:
    text = " ".join([
        formula.get("latex", ""),
        formula.get("description", ""),
        formula.get("context", "")
    ]).lower()

    if "гранич" in text or "boundary" in text or "x=0" in text or "x=h" in text:
        return "boundary_conditions"

    if "2d" in text or "двумер" in text or "электроконвек" in text or "\\vec{v}" in text:
        return "2D_electroconvection"

    if "3d" in text or "трехмер" in text or "трёхмер" in text or "x, y, z" in text:
        return "3D_model"

    if "стационар" in text and "нестационар" not in text:
        return "1D_stationary"

    if "нестационар" in text or "\\partial" in text:
        return "1D_unsteady"

    if "pe" in text or "re" in text or "безразмер" in text or "критер" in text:
        return "dimensionless"

    return "general"


def detect_phenomena(formula: Dict[str, Any]) -> List[str]:
    text = " ".join([
        formula.get("latex", ""),
        formula.get("description", ""),
        formula.get("context", "")
    ]).lower()

    phenomena = []

    rules = {
        "diffusion": ["диффуз", "diffusion", "d_i", "d_1", "d_2", "\\nabla c", "\\partial c"],
        "migration": ["миграц", "migration", "электромиграц", "z_i", "electric field", "\\varphi", "\\vec{e}"],
        "water_dissociation": ["диссоциац", "рекомбинац", "h_2o", "h^+", "oh^-", "k_d", "k_r", "k_w"],
        "poisson_equation": ["пуассон", "poisson", "\\varepsilon", "\\rho", "\\sum"],
        "space_charge": ["пространственный заряд", "опз", "space charge", "\\rho"],
        "electroconvection": ["электроконвек", "electroconvection", "\\vec{v}", "navier", "stokes", "навье"],
        "current_density": ["плотность тока", "current density", "i_c", "\\vec{i}", "i_{av}", "i_{lim}"],
        "boundary_conditions": ["гранич", "boundary", "x=0", "x=h", "мембран"],
        "dimensionless_parameters": ["pe", "re", "k_{el}", "безразмер", "критерий"],
        "thermal_effects": ["температур", "тепло", "джоул", "q =", "t "],
    }

    for name, keys in rules.items():
        if any(k in text for k in keys):
            phenomena.append(name)

    return phenomena


def build_aliases_ru(formula: Dict[str, Any]) -> List[str]:
    text = formula.get("description", "").lower()
    aliases = []

    if "нернст" in text or "поток ионов" in text:
        aliases.extend(["уравнение Нернста-Планка", "поток ионов", "ионный поток"])

    if "пуассон" in text:
        aliases.extend(["уравнение Пуассона", "электрический потенциал", "пространственный заряд"])

    if "диссоциац" in text or "рекомбинац" in text:
        aliases.extend(["диссоциация воды", "рекомбинация воды", "ионное произведение воды"])

    if "навье" in text or "стокс" in text:
        aliases.extend(["уравнение Навье-Стокса", "гидродинамика", "электроконвекция"])

    if "предельный" in text:
        aliases.extend(["предельный ток", "предельный диффузионный ток"])

    if "гранич" in text:
        aliases.extend(["граничные условия", "условия на мембране"])

    return sorted(set(aliases))


def build_aliases_en(formula: Dict[str, Any]) -> List[str]:
    text = formula.get("description", "").lower()
    aliases = []

    if "нернст" in text or "поток ионов" in text:
        aliases.extend(["Nernst-Planck equation", "ionic flux", "ion flux"])

    if "пуассон" in text:
        aliases.extend(["Poisson equation", "electric potential", "space charge"])

    if "диссоциац" in text or "рекомбинац" in text:
        aliases.extend(["water dissociation", "water recombination", "water splitting"])

    if "навье" in text or "стокс" in text:
        aliases.extend(["Navier-Stokes equation", "hydrodynamics", "electroconvection"])

    if "предельный" in text:
        aliases.extend(["limiting current", "diffusion-limited current"])

    if "гранич" in text:
        aliases.extend(["boundary conditions", "membrane boundary"])

    return sorted(set(aliases))


def build_query_terms(formula: Dict[str, Any]) -> List[str]:
    parts = [
        formula.get("description", ""),
        formula.get("context", ""),
        formula.get("latex", ""),
        " ".join(build_aliases_ru(formula)),
        " ".join(build_aliases_en(formula)),
    ]

    text = " ".join(parts).lower()
    tokens = re.findall(r"[a-zа-яё0-9_\\^{}+-]+", text)

    stop = {
        "для", "при", "как", "или", "что", "это", "the", "and", "with",
        "formula", "equation", "уравнение", "формула"
    }

    return sorted(set(t for t in tokens if len(t) >= 3 and t not in stop))


def enrich_formula(formula: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(formula)

    enriched["source_type"] = enriched.get("source_type", "curated")
    enriched["confidence"] = enriched.get("confidence", 1.0)
    enriched["model_class"] = enriched.get("model_class") or classify_formula(enriched)
    enriched["phenomena"] = enriched.get("phenomena") or detect_phenomena(enriched)
    enriched["aliases_ru"] = enriched.get("aliases_ru") or build_aliases_ru(enriched)
    enriched["aliases_en"] = enriched.get("aliases_en") or build_aliases_en(enriched)
    enriched["query_terms"] = enriched.get("query_terms") or build_query_terms(enriched)
    enriched["is_core"] = enriched.get("is_core", enriched["model_class"] != "general")

    return enriched


def enrich_database(input_path: str, output_path: str) -> None:
    input_file = Path(input_path)
    output_file = Path(output_path)

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    formulas = data.get("formulas", [])
    enriched = [enrich_formula(formula) for formula in formulas]

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open("w", encoding="utf-8") as f:
        json.dump({"formulas": enriched}, f, ensure_ascii=False, indent=2)

    print(f"Saved enriched formula database: {output_file}")
    print(f"Total formulas: {len(enriched)}")


if __name__ == "__main__":
    enrich_database(
        "data/knowledge/formulas_knowledge_base.json",
        "data/knowledge/formulas_knowledge_base_enriched.json",
    )