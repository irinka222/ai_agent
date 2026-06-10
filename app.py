import asyncio
import re
import time

import streamlit as st

from run_agent import ElectromembraneAgent, OllamaLLM
from src.retrieval.retriever import Retriever
from src.knowledge.formula_kb_loader import FormulaKnowledgeBase


INDEX_PATH = "master_index"
FORMULA_KB_PATH = "data/knowledge/formulas_knowledge_base.json"
LLM_MODEL = "vikhr-7b-instruct"

st.set_page_config(
    page_title="Научный ассистент",
    layout="wide"
)


CUSTOM_CSS = """
<style>
    .stApp {
        background: #f6f7f9;
    }

    .main .block-container {
        max-width: 1180px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
    }

    .hero {
        background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
        border-radius: 22px;
        padding: 34px 40px;
        margin-bottom: 26px;
        color: white;
        box-shadow: 0 18px 45px rgba(17,24,39,0.18);
    }

    .hero h1 {
        margin: 0;
        font-size: 34px;
        font-weight: 700;
        letter-spacing: -0.03em;
    }

    .hero p {
        margin-top: 12px;
        margin-bottom: 0;
        color: #d1d5db;
        font-size: 16px;
        line-height: 1.6;
    }

    .panel {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 26px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.06);
        margin-top: 0;
    }

    .section-title {
        font-size: 22px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 16px;
    }

    .muted {
        color: #6b7280;
        font-size: 14px;
        line-height: 1.55;
    }

    div[data-testid="stSidebar"] {
        background: #fff;
        border-right: 1px solid #e5e7eb;
    }

    .stTextArea textarea {
        border-radius: 16px;
        border: 1px solid #d1d5db;
        padding: 16px;
        font-size: 15px;
        background: #fff;
    }

    .stTextArea textarea:focus {
        border-color: #111827;
        box-shadow: 0 0 0 1px #111827;
    }

    div.stButton > button {
        border-radius: 40px;
        width: 52px;
        height: 52px;
        padding: 0;
        font-size: 24px;
        font-weight: normal;
        background: #111827;
        color: white;
        border: none;
        transition: 0.2s;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    div.stButton > button:hover {
        background: #2d3a4f;
    }

    .answer-box {
        background: #fff;
        border: 1px solid #e5e7eb;
        border-radius: 20px;
        padding: 28px;
        box-shadow: 0 12px 30px rgba(15,23,42,0.06);
        margin-top: 22px;
    }

    .answer-title {
        font-size: 20px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 14px;
        border-bottom: 1px solid #e5e7eb;
        padding-bottom: 12px;
    }

    .small-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 16px;
        margin-bottom: 10px;
        color: #374151;
        font-size: 14px;
    }

    .footer {
        color: #6b7280;
        font-size: 13px;
        margin-top: 28px;
        text-align: center;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_agent():
    retriever = Retriever(index_path=INDEX_PATH)
    formula_kb = FormulaKnowledgeBase(FORMULA_KB_PATH)

    llm = OllamaLLM(
        model=LLM_MODEL,
        num_predict=512,
        temperature=0.0
    )

    return ElectromembraneAgent(
        retriever=retriever,
        llm=llm,
        formula_kb=formula_kb
    )


def run_async(coro):
    return asyncio.run(coro)


def format_latex(text: str) -> str:
    text = re.sub(r"\\\[(.*?)\\\]", r"$$\1$$", text, flags=re.DOTALL)
    text = re.sub(r"\\\((.*?)\\\)", r"$\1$", text)
    return text


st.markdown(
    """
    <div class="hero">
        <h1>Научный ассистент по электромембранным процессам</h1>
        <p>
            Локальный RAG-агент для поиска, анализа и генерации проверяемых ответов
            на основе корпуса научных публикаций по математическому моделированию
            электромембранных систем.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


with st.sidebar:
    st.markdown("### Демонстрационные вопросы")
    st.markdown(
        """
        <div class="muted">
            Выберите один из подготовленных вопросов для демонстрации работы ассистента.
        </div>
        """,
        unsafe_allow_html=True
    )

    demo_question = st.radio(
        label="",
        options=[
            "Какое уравнение описывает перенос ионов?",
            "Что такое условие электронейтральности?",
            "Что такое электроконвекция?",
            "Что такое потенциал Доннана?",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("### Тематика")
    st.markdown(
        """
        <div class="small-card">Перенос ионов</div>
        <div class="small-card">Электродиализ</div>
        <div class="small-card">Диссоциация и рекомбинация воды</div>
        <div class="small-card">Электроконвекция</div>
        """,
        unsafe_allow_html=True
    )


left, right = st.columns([1.25, 0.75], gap="large")


with left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Запрос к ассистенту</div>', unsafe_allow_html=True)

    col_text, col_button = st.columns([10, 1])

    with col_text:
        question = st.text_area(
            label="Что бы вы хотели узнать?",
            value=demo_question,
            height=125,
            placeholder="Например: Что такое условие электронейтральности?",
            label_visibility="collapsed"
        )

    with col_button:
        st.markdown('<div style="margin-top: 32px;"></div>', unsafe_allow_html=True)
        ask_button = st.button("→", type="primary", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


with right:
    st.markdown(
        """
        <div class="panel">
            <div class="section-title">Описание системы</div>
            <div class="muted">
                Ассистент выполняет гибридный поиск по локальному корпусу научных документов,
                формирует контекст из релевантных фрагментов и генерирует ответ с указанием
                использованных источников. Система поддерживает работу с текстовыми данными,
                формулами и библиографическими метаданными.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


if ask_button:
    if not question.strip():
        st.warning("Введите вопрос для ассистента.")
    else:
        with st.spinner("Выполняется поиск по корпусу и формируется ответ..."):
            try:
                start_time = time.perf_counter()

                agent = load_agent()
                answer = run_async(agent.answer(question.strip()))
                answer = format_latex(answer)

                elapsed_time = time.perf_counter() - start_time

                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown('<div class="answer-title">Ответ ассистента</div>', unsafe_allow_html=True)
                st.markdown(answer)
                st.caption(f"Время обработки запроса: {elapsed_time:.2f} с")
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error("Произошла ошибка при работе ассистента.")
                st.exception(e)


st.markdown(
    """
    <div class="footer">
        Демонстрационный интерфейс локального RAG-агента для математического моделирования электромембранных процессов
    </div>
    """,
    unsafe_allow_html=True
)