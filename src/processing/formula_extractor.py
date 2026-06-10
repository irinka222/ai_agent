# src/processing/formula_extractor.py
# -*- coding: utf-8 -*-

import re
import logging
from typing import Optional, Dict, Any, List, Set, Tuple

logger = logging.getLogger(__name__)


GARBAGE_PATTERNS = [
    r"\(cid:\d+\)",
    r"�",
    r"[©®™]",
    r"http[s]?://",
    r"www\.",
    r"[‡†]",
    r"[¦§]",
]


DOMAIN_KEYWORDS = {
    "membrane", "ion", "flux", "diffusion", "current",
    "voltage", "potential", "concentration", "transport",
    "electro", "charge", "field", "gradient", "poisson",
    "nernst", "planck", "navier", "stokes",
    "диссоциац", "рекомбинац", "электродиализ", "мембран",
    "концентрац", "поток", "потенциал", "пуассона",
    "нернста", "планка", "навье", "стокса",
    "электроконвекц", "пространственный заряд",
    "область пространственного заряда", "диффузионный слой",
}


MATH_SYMBOLS_PATTERN = re.compile(
    r"(=|\\frac|\\partial|\\nabla|\\Delta|\\sum|\\int|"
    r"\^|_|[+\-*/]|≤|≥|≠|∂|∇|∆|Δ|∑|∫|√|∞|≈)"
)


DOMAIN_SYMBOLS_PATTERN = re.compile(
    r"\b(C_i|C_1|C_2|C_3|C_4|j_i|J_i|D_i|z_i|"
    r"k_d|k_r|k_w|pH|Pe|Re|F|R|T|H|L|V_0|I_c|I|E|"
    r"Na|Cl|OH|H_2O|H\+|OH\-)\b"
)


LATEX_DELIMITERS = [
    (r"\$\$(.*?)\$\$", re.DOTALL),
    (r"\$(.*?)\$", 0),
    (r"\\\[(.*?)\\\]", re.DOTALL),
    (r"\\begin\{equation\}(.*?)\\end\{equation\}", re.DOTALL),
    (r"\\begin\{align\}(.*?)\\end\{align\}", re.DOTALL),
    (r"\\begin\{multline\}(.*?)\\end\{multline\}", re.DOTALL),
]


KNOWN_REPAIRS = [
    {
        "pattern": r"O\s+H\s+d\s+C\s+k\s+v\s+2\s+1.*OH\s+H\s+r\s+C\s+C\s+k",
        "replacement": r"v_1 = k_d C_{H_2O}, \quad v_2 = k_r C_{H^+} C_{OH^-}",
        "confidence": 0.9,
    },
    {
        "pattern": r"k\s*d.*C.*H.*2.*O.*k\s*r.*C.*H.*C.*OH",
        "replacement": r"k_d C_{H_2O} = k_r C_{H^+} C_{OH^-}",
        "confidence": 0.85,
    },
    {
        "pattern": r"lg.*H.*C.*pH|pH.*lg.*H",
        "replacement": r"\mathrm{pH} = -\lg C_{H^+}",
        "confidence": 0.85,
    },
]


FORMULA_SEMANTIC_PATTERNS = [
    (
        r"\\partial.?2.*\\varphi|\\partial.?2.*\\phi|\\Delta.*\\varphi|\\Delta.*\\phi|\\nabla\^?2.*\\varphi|\\nabla\^?2.*\\phi|\\sum.*z_?i.*C_?i",
        "уравнение Пуассона потенциал пространственный заряд"
    ),
    (
        r"j_?\{?i\}?\s*=|J_?\{?i\}?\s*=|\\vec\{j\}_?\{?i\}?\s*=|\\mathbf\{j\}_?\{?i\}?\s*=",
        "поток ионов уравнение Нернста-Планка миграция диффузия"
    ),
    (
        r"D_?\{?i\}?.*\\partial.*C|D_?\{?i\}?.*\\nabla.*C|D_?\{?i\}?.*dC|\\nabla C|\\partial C",
        "диффузионный поток закон Фика градиент концентрации"
    ),
    (
        r"z_?\{?i\}?.*F.*D|z_?\{?i\}?.*\\varphi|z_?\{?i\}?.*\\vec\{E\}|C_?\{?i\}?.*\\nabla\\varphi|C_?\{?i\}?.*\\partial.*\\varphi",
        "миграционный поток электрическое поле электромиграция"
    ),
    (
        r"\\partial\s*C|\\frac\{\\partial\s*C|\\frac\{\\partial C|\\partial C_?\{?i\}?.*\\partial t|Pe.*\\partial C",
        "уравнение материального баланса нестационарный перенос концентрация"
    ),
    (
        r"\\operatorname\{div\}.*\\vec\{j\}|\\operatorname\{div\}.*j|\\nabla.*\\cdot.*\\vec\{j\}|\\nabla.*\\cdot.*j|div.*j",
        "дивергенция потока уравнение непрерывности сохранение массы"
    ),
    (
        r"R_?\{?3\}?|R_?\{?4\}?|k_d|k_r|k_w|H_2O|H\^\+|OH\^-|H_3O\^\+|C_\{H\^\+\}|C_\{OH\^-\}",
        "диссоциация рекомбинация воды источниковые члены ионы водорода гидроксила"
    ),
    (
        r"I_?\{?c\}?|\\vec\{I\}|\\mathbf\{I\}|F\\s*\\sum.*z_?i.*j_?i|F.*z_?1.*j_?1|current",
        "плотность электрического тока ток проводимости"
    ),
    (
        r"I_?\{?av\}?|i_?\{?av\}?|I_?\{?lim\}?|i_?\{?lim\}?|limiting current|предельн.*ток",
        "вольт-амперная характеристика средний ток предельный диффузионный ток"
    ),
    (
        r"\\rho_?\{?e\}?|\\rho\s*=|\\rho\\vec\{E\}|F\(z_?1 C_?1.*z_?2 C_?2\)|C_1-C_2|C_1\s*-\s*C_2\s*\+\s*C_3\s*-\s*C_4",
        "плотность пространственного заряда область пространственного заряда"
    ),
    (
        r"\\partial.*\\vec\{V\}.*\\partial t|\\vec\{V\}.*\\nabla.*\\vec\{V\}|\\nabla P|\\Delta\\vec\{V\}|\\nu\\Delta|Navier|Stokes|Навье|Стокс",
        "уравнение Навье-Стокса гидродинамика скорость давление вязкость"
    ),
    (
        r"\\nabla\\cdot\\vec\{V\}|\\operatorname\{div\}\\vec\{V\}|div\\s*V\s*=\s*0",
        "условие несжимаемости уравнение неразрывности жидкости"
    ),
    (
        r"\\vec\{f\}|\\rho\\vec\{E\}|F\(z_?1 C_?1.*z_?2 C_?2\).*\\vec\{E\}|K_\{?el\}?.*\\varepsilon.*\\vec\{E\}",
        "объемная электрическая сила кулоновская сила электроконвекция"
    ),
    (
        r"x\s*=\s*0|x\s*=\s*H|C_?i\(t,\s*0\)|C_?i\(t,\s*H\)|\\varphi\(t,\s*0\)|\\varphi\(t,\s*H\)",
        "граничное условие ионообменная мембрана АОМ КОМ"
    ),
    (
        r"C_1\(0\)|C_2\(0\)|C_i\(0,\s*x\)|\\varphi\(0,\s*x\)|\\vec\{V\}\(0",
        "начальное условие начальное распределение"
    ),
    (
        r"\\varepsilon\s*=|\\varepsilon\^\{\(u\)\}|l_?D|l_?d|\\left\(\s*\\frac\{l_?D\}\{H\}\s*\\right\)\^2",
        "малый параметр дебаевская длина толщина канала"
    ),
    (
        r"\bPe\b|\bRe\b|K_\{?el\}?|Le|Arn|\\mu|\\gamma|\\lambda",
        "безразмерные параметры критерии подобия"
    ),
    (
        r"\\tilde\{S\}|\\eta|\\Phi|u\^\{\(0\)\}|\\nabla\\eta|\\Delta\\eta",
        "общая упрощенная модель функция тока обобщенная концентрация"
    ),
    (
        r"k\\s*\\frac\{d\^2T\}|\\frac\{d\^2T\}|G\s*\+\\s*Q|qk_r|T\)|температур|heat",
        "тепловые эффекты температура джоулев нагрев теплота реакции"
    ),
    (
        r"\\delta\s*=|\\delta_0|1\.47|Leveque|Левек",
        "толщина диффузионного слоя оценка Левека"
    ),
    (
        r"\\bar\{C\}|\\bar\{E\}|\\Pi C|\\Pi\\varphi|O\(\\sqrt\{\\varepsilon\}\)",
        "асимптотическое разложение пограничный слой электронейтральная область"
    ),
    (
        r"\\vec\{E\}\s*=|\\mathbf\{E\}\s*=|E\s*=\s*-?\\nabla\\varphi|\\vec\{E\}.*\\vec\{I\}",
        "напряженность электрического поля обобщенный закон Ома"
    ),
    (
        r"C_1\s*-\s*C_2\s*=\s*0|электронейтральн|electroneutral",
        "условие электронейтральности"
    ),
]



class FormulaExtractor:
    def __init__(self):
        self.seen: Set[str] = set()

    def extract(
        self,
        text: str,
        page: Optional[int] = None,
        source: str = "heuristic",
    ) -> List[Dict[str, Any]]:
        if not text:
            return []

        text = preclean_formula_text(text)

        formulas: List[Dict[str, Any]] = []
        formulas.extend(self._extract_latex_blocks(text, page, source))
        formulas.extend(self._extract_line_candidates(text, page, source))

        result = []
        for formula in formulas:
            if not formula:
                continue

            latex = formula.get("latex", "")
            if self._is_duplicate(latex):
                continue

            result.append(formula)

        logger.debug("Extracted %d formula candidates from page=%s", len(result), page)
        return result

    def _extract_latex_blocks(
        self,
        text: str,
        page: Optional[int],
        source: str,
    ) -> List[Dict[str, Any]]:
        results = []

        for pattern, flags in LATEX_DELIMITERS:
            for match in re.findall(pattern, text, flags):
                candidate = build_formula_candidate(
                    latex=match,
                    source=source if source != "heuristic" else "latex",
                    page=page,
                    context_before="",
                    context_after="",
                )
                if candidate:
                    results.append(candidate)

        return results

    def _extract_line_candidates(
        self,
        text: str,
        page: Optional[int],
        source: str,
    ) -> List[Dict[str, Any]]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        results: List[Dict[str, Any]] = []

        buffer: List[str] = []
        buffer_start_index = 0

        for i, line in enumerate(lines):
            score = formula_score(line)

            if score >= 0.6:
                if not buffer:
                    buffer_start_index = i
                buffer.append(line)
                continue

            if buffer:
                finalized = self._finalize_buffer(
                    buffer=buffer,
                    lines=lines,
                    start_index=buffer_start_index,
                    end_index=i,
                    page=page,
                    source=source,
                )
                if finalized:
                    results.append(finalized)
                buffer = []

        if buffer:
            finalized = self._finalize_buffer(
                buffer=buffer,
                lines=lines,
                start_index=buffer_start_index,
                end_index=len(lines),
                page=page,
                source=source,
            )
            if finalized:
                results.append(finalized)

        return results

    def _finalize_buffer(
        self,
        buffer: List[str],
        lines: List[str],
        start_index: int,
        end_index: int,
        page: Optional[int],
        source: str,
    ) -> Optional[Dict[str, Any]]:
        if not buffer:
            return None

        raw = " ".join(buffer)
        context_before = " ".join(lines[max(0, start_index - 2):start_index])
        context_after = " ".join(lines[end_index:min(len(lines), end_index + 2)])

        return build_formula_candidate(
            latex=raw,
            source=source,
            page=page,
            context_before=context_before,
            context_after=context_after,
        )

    def _is_duplicate(self, formula: str) -> bool:
        if not formula:
            return True

        key = re.sub(r"\s+", "", formula)

        if key in self.seen:
            return True

        self.seen.add(key)
        return False


def preclean_formula_text(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"[]", "", text)
    text = re.sub(r"\(cid:\d+\)", "", text)
    text = text.replace("\u00a0", " ")
    text = text.replace("\u200b", "")
    text = text.replace("\ufeff", "")
    text = text.replace("", " ")

    return text


def apply_known_repairs(text: str) -> Tuple[str, Optional[float]]:
    if not text:
        return "", None

    normalized = re.sub(r"\s+", " ", text).strip()

    for rule in KNOWN_REPAIRS:
        if re.search(rule["pattern"], normalized, flags=re.IGNORECASE):
            return rule["replacement"], float(rule["confidence"])

    return text, None


def repair_ocr_formula(text: str) -> str:
    if not text:
        return ""

    text = re.sub(r"\bH\s*2\s*O\b", "H_2O", text)
    text = re.sub(r"\bH\s*3\s*O\s*\+\b", "H_3O^+", text)
    text = re.sub(r"\bH\s*\+\b", "H^+", text)
    text = re.sub(r"\bOH\s*-\b", "OH^-", text)

    text = re.sub(r"\bk\s*d\b", "k_d", text)
    text = re.sub(r"\bk\s*r\b", "k_r", text)
    text = re.sub(r"\bk\s*w\b", "k_w", text)

    text = re.sub(r"\bC\s+([1-4i])\b", r"C_\1", text)
    text = re.sub(r"\bD\s+([1-4i])\b", r"D_\1", text)
    text = re.sub(r"\bj\s+([1-4i])\b", r"j_\1", text)
    text = re.sub(r"\bJ\s+([1-4i])\b", r"J_\1", text)
    text = re.sub(r"\bz\s+([1-4i])\b", r"z_\1", text)

    text = re.sub(r"\bV\s+0\b", "V_0", text)
    text = re.sub(r"\bI\s+c\b", "I_c", text)

    return text


def normalize_formula_text(text: str) -> str:
    if not text:
        return ""

    text, _ = apply_known_repairs(text)
    text = repair_ocr_formula(text)

    replacements = {
        "−": "-",
        "–": "-",
        "—": "-",
        "-": "-",
        "‐": "-",
        "×": r"\times",
        "⋅": r"\cdot",
        "∙": r"\cdot",
        "·": r"\cdot",
        "∗": r"\cdot",
        "∂": r"\partial",
        "∇": r"\nabla",
        "∆": r"\Delta",
        "Δ": r"\Delta",
        "Σ": r"\sum",
        "∑": r"\sum",
        "Π": r"\prod",
        "∏": r"\prod",
        "∫": r"\int",
        "√": r"\sqrt",
        "∞": r"\infty",
        "≤": r"\le",
        "≥": r"\ge",
        "≠": r"\ne",
        "≈": r"\approx",
        "≃": r"\simeq",
        "≅": r"\cong",
        "∼": r"\sim",
        "∝": r"\propto",
        "±": r"\pm",
        "∓": r"\mp",
        "→": r"\to",
        "←": r"\leftarrow",
        "↔": r"\leftrightarrow",
        "⇒": r"\Rightarrow",
        "⇔": r"\Leftrightarrow",
        "∈": r"\in",
        "∉": r"\notin",
        "⊂": r"\subset",
        "⊆": r"\subseteq",
        "∅": r"\varnothing",
        "∀": r"\forall",
        "∃": r"\exists",
        "∧": r"\wedge",
        "∨": r"\vee",
        "¬": r"\neg",
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "δ": r"\delta",
        "ε": r"\varepsilon",
        "ϵ": r"\epsilon",
        "ζ": r"\zeta",
        "η": r"\eta",
        "θ": r"\theta",
        "ϑ": r"\vartheta",
        "ι": r"\iota",
        "κ": r"\kappa",
        "λ": r"\lambda",
        "μ": r"\mu",
        "µ": r"\mu",
        "ν": r"\nu",
        "ξ": r"\xi",
        "π": r"\pi",
        "ρ": r"\rho",
        "ϱ": r"\varrho",
        "σ": r"\sigma",
        "τ": r"\tau",
        "υ": r"\upsilon",
        "φ": r"\varphi",
        "ϕ": r"\varphi",
        "χ": r"\chi",
        "ψ": r"\psi",
        "ω": r"\omega",
        "Γ": r"\Gamma",
        "Λ": r"\Lambda",
        "Ω": r"\Omega",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"\bgrad\b", r"\\nabla", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdiv\b", r"\\operatorname{div}", text, flags=re.IGNORECASE)
    text = re.sub(r"\brot\b", r"\\operatorname{rot}", text, flags=re.IGNORECASE)
    text = re.sub(r"\bcurl\b", r"\\operatorname{rot}", text, flags=re.IGNORECASE)
    text = re.sub(r"\bln\b", r"\\ln", text)
    text = re.sub(r"\blg\b", r"\\lg", text)
    text = re.sub(r"\bexp\b", r"\\exp", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\\partial\s*([A-Za-zА-Яа-я])", r"\\partial \1", text)
    text = re.sub(r"\\nabla\s*(?:\\cdot|·|\.)", r"\\nabla\\cdot", text)
    text = re.sub(r"\\nabla\s*(?:\\times|x|×)", r"\\nabla\\times", text)

    return text.strip()


def formula_score(text: str) -> float:
    if not text:
        return 0.0

    s = text.strip()

    if len(s) < 3:
        return 0.0

    score = 0.0

    if "=" in s:
        score += 0.35

    if any(c in s for c in "=≈≠≤≥+−-*/^_∂∇∑∏∫√∞"):
        score += 0.25

    if any(g in s.lower() for g in "αβγδεζηθικλμνξοπρστυφχψωφϕεερν"):
        score += 0.15

    if re.search(r"\d+", s):
        score += 0.1

    if "_" in s or "^" in s:
        score += 0.1

    if "\\" in s:
        score += 0.2

    lower = s.lower()

    if any(kw in lower for kw in DOMAIN_KEYWORDS):
        score += 0.2

    if DOMAIN_SYMBOLS_PATTERN.search(s):
        score += 0.2

    words = re.findall(r"[A-Za-zА-Яа-яЁё]{3,}", s)

    if len(words) > 8:
        score -= 0.35

    if len(s) > 250:
        score -= 0.2

    if re.search(r"^(рис\.|рисунок|figure|таблица|table)", lower):
        score -= 0.5

    return max(0.0, min(1.0, score))


def is_valid_formula_candidate(text: str) -> bool:
    if not text:
        return False

    s = text.strip()

    if len(s) < 3 or len(s) > 700:
        return False

    for pat in GARBAGE_PATTERNS:
        if re.search(pat, s):
            return False

    if re.match(r"^(NaCl|KCl|CEM|AEM|EDL|SCR|DL|PD|PS)$", s):
        return False

    russian_words = re.findall(r"[А-Яа-яЁё]{4,}", s)
    english_words = re.findall(r"[A-Za-z]{5,}", s)

    has_math_symbols = bool(MATH_SYMBOLS_PATTERN.search(s))
    has_domain_symbols = bool(DOMAIN_SYMBOLS_PATTERN.search(s))
    has_domain_keyword = any(k in s.lower() for k in DOMAIN_KEYWORDS)

    if len(russian_words) + len(english_words) > 12 and not has_math_symbols:
        return False

    if not has_math_symbols and not has_domain_symbols and not has_domain_keyword:
        return False

    return True


def parse_ast_safe(latex: str):
    try:
        from sympy.parsing.latex import parse_latex
        return parse_latex(latex)
    except ImportError:
        logger.debug("sympy или antlr runtime не установлен, AST не используется")
        return None
    except Exception as e:
        logger.debug("AST parse failed: %s | %s", latex[:80], e)
        return None


def compute_confidence(
    latex: str,
    ast,
    source: str = "heuristic",
    repair_confidence: Optional[float] = None,
) -> float:
    if repair_confidence is not None:
        return repair_confidence

    source_weights = {
        "grobid": 0.45,
        "mathml": 0.4,
        "latex": 0.5,
        "heuristic": 0.25,
        "ocr": 0.1,
        "pymupdf": 0.2,
    }

    score = source_weights.get(source, 0.15)

    if "=" in latex:
        score += 0.15

    if any(token in latex for token in [
        r"\frac", r"\partial", r"\nabla", r"\Delta",
        r"\sum", r"\int", "^", "_"
    ]):
        score += 0.25

    if DOMAIN_SYMBOLS_PATTERN.search(latex):
        score += 0.15

    if any(k in latex.lower() for k in DOMAIN_KEYWORDS):
        score += 0.1

    if ast is not None:
        score += 0.15

    if len(latex) < 180:
        score += 0.05

    words = re.findall(r"[А-Яа-яЁёA-Za-z]{5,}", latex)

    if len(words) > 10:
        score -= 0.2

    return max(0.0, min(score, 1.0))


def extract_variables_from_formula(latex: str) -> List[str]:
    if not latex:
        return []

    candidates = re.findall(
        r"\\?[A-Za-zА-Яа-я]+(?:_\{?[A-Za-z0-9+\-]+\}?)?(?:\^\{?[+\-0-9]+\}?)?",
        latex,
    )

    stop = {
        "frac", "partial", "nabla", "Delta", "sum", "int", "left",
        "right", "cdot", "times", "text", "mathrm", "operatorname",
        "quad", "qquad", "begin", "end", "equation", "align",
        "to", "leftrightarrow", "rightarrow", "leftarrow",
        "approx", "le", "ge", "ne", "varphi", "varepsilon",
    }

    variables = []
    seen = set()

    for item in candidates:
        clean = item.replace("\\", "").strip()

        if not clean or clean in stop:
            continue

        if len(clean) > 25:
            continue

        if clean not in seen:
            seen.add(clean)
            variables.append(clean)

    return variables


def get_formula_semantic_description(latex: str, context_before: str = "") -> str:
    if not latex:
        return ""

    matched = []

    for pattern, desc in FORMULA_SEMANTIC_PATTERNS:
        if re.search(pattern, latex, re.IGNORECASE):
            matched.append(desc)

    if matched:
        return " | ".join(list(dict.fromkeys(matched))[:3])

    if context_before:
        text = f"{latex} {context_before}".strip()
        for pattern, desc in FORMULA_SEMANTIC_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                matched.append(desc)

    if matched:
        return " | ".join(list(dict.fromkeys(matched))[:3])

    symbols = re.findall(r"[A-Za-zА-Яа-я]{2,}", latex)

    if symbols:
        return "математическое выражение " + " ".join(symbols[:8])

    return "математическое выражение формула уравнение"

def build_formula_candidate(
    latex: str,
    source: str = "heuristic",
    page: Optional[int] = None,
    context_before: str = "",
    context_after: str = "",
    label: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if not latex:
        return None

    raw = latex
    repaired, repair_confidence = apply_known_repairs(latex)
    normalized = normalize_formula_text(repaired)

    if not is_valid_formula_candidate(normalized):
        return None

    ast = parse_ast_safe(normalized)
    confidence = compute_confidence(
        normalized,
        ast,
        source=source,
        repair_confidence=repair_confidence,
    )

    if confidence < 0.25:
        return None

    return {
        "type": "formula_candidate",
        "latex": normalized,
        "normalized": normalized,
        "raw": raw,
        "source": source,
        "page": page,
        "label": label,
        "confidence": confidence,
        "score": formula_score(normalized),
        "ast": ast is not None,
        "variables": extract_variables_from_formula(normalized),
        "context_before": context_before[:500] if context_before else "",
        "context_after": context_after[:500] if context_after else "",
        "semantic_description": get_formula_semantic_description(
            normalized,
            context_before=context_before,
        ),
        "use_for_index": confidence >= 0.65,
        "extraction_method": source,
    }


def extract_full_formula(elem, source: str = "grobid") -> Optional[Dict[str, Any]]:
    latex = None
    label = None

    if hasattr(elem, "attrib"):
        latex = (
            elem.attrib.get("tex")
            or elem.attrib.get("latex")
            or elem.attrib.get("formula")
        )
        label = elem.attrib.get("label") or elem.attrib.get("n")

    if not latex and hasattr(elem, "text"):
        latex = elem.text

    return build_formula_candidate(
        latex=latex or "",
        source=source,
        label=label,
    )


def extract_formula_candidates_from_text(
    text: str,
    page: Optional[int] = None,
    source: str = "heuristic",
) -> List[Dict[str, Any]]:
    extractor = FormulaExtractor()
    return extractor.extract(text=text, page=page, source=source)


def is_valid_formula(latex: str) -> bool:
    return is_valid_formula_candidate(latex)


def clean_formula(formula: str) -> str:
    return normalize_formula_text(formula)


def clean_formula_candidate(formula: str) -> Dict[str, Any]:
    candidate = build_formula_candidate(formula, source="heuristic")

    if candidate is None:
        return {
            "raw": formula,
            "latex": "",
            "confidence": 0.0,
            "type": "formula_candidate",
            "use_for_index": False,
        }

    return candidate


def format_formula_for_rag(formula: str, description: Optional[str] = None) -> str:
    formula = clean_formula(formula)

    if description:
        return f"[FORMULA] {description}: {formula}"

    return f"[FORMULA] {formula}"