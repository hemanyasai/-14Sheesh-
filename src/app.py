"""
app.py — Streamlit demo UI for the Cricketers Self-Query RAG application.

Features:
  - Text input for user questions
  - One-click example questions covering all four routing categories
  - Side-by-side comparison: Self-Query RAG vs Naive Semantic-Only RAG
  - Expandable "details" panel: extracted filters JSON, routing decision, matched records
"""

from __future__ import annotations

import sys
import os

# Ensure src/ is on the path when running from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from src.pipeline import answer_question, answer_question_naive
import json

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="🏏 Cricketers Self-Query RAG",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark cricket-themed design
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark background */
    .stApp {
        background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
        color: #e2e8f0;
    }

    /* Hide default streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Hero header */
    .hero-header {
        background: linear-gradient(135deg, rgba(20, 184, 166, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
        border: 1px solid rgba(20, 184, 166, 0.25);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
    }

    .hero-header h1 {
        font-family: 'Outfit', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(90deg, #14b8a6, #3b82f6, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    .hero-header p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.4rem;
    }

    /* Section cards */
    .answer-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(20, 184, 166, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(8px);
    }

    .naive-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(8px);
    }

    /* Routing badge */
    .routing-badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .badge-filter   { background: rgba(20,184,166,0.2); color: #14b8a6; border: 1px solid rgba(20,184,166,0.4); }
    .badge-hybrid   { background: rgba(168,85,247,0.2); color: #a855f7; border: 1px solid rgba(168,85,247,0.4); }
    .badge-semantic { background: rgba(59,130,246,0.2); color: #60a5fa; border: 1px solid rgba(59,130,246,0.4); }
    .badge-oos      { background: rgba(239,68,68,0.2);  color: #f87171; border: 1px solid rgba(239,68,68,0.4); }

    /* Example question pills */
    .example-container {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0;
    }

    /* Main input styling */
    .stTextInput > div > div > input {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1.5px solid rgba(20, 184, 166, 0.35) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
        font-size: 1rem !important;
        padding: 0.8rem 1rem !important;
        transition: border-color 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: rgba(20, 184, 166, 0.7) !important;
        box-shadow: 0 0 0 3px rgba(20, 184, 166, 0.1) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(20,184,166,0.2), rgba(59,130,246,0.2)) !important;
        border: 1px solid rgba(20,184,166,0.4) !important;
        color: #14b8a6 !important;
        border-radius: 8px !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
        font-size: 0.85rem !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(20,184,166,0.35), rgba(59,130,246,0.3)) !important;
        border-color: rgba(20,184,166,0.7) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(20,184,166,0.2) !important;
    }

    /* Toggle */
    .stToggle > label { color: #94a3b8 !important; }

    /* Expander */
    .stExpander {
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        background: rgba(15,23,42,0.5) !important;
    }

    /* Spinner */
    .stSpinner > div { border-top-color: #14b8a6 !important; }

    /* Dividers */
    hr { border-color: rgba(255,255,255,0.06) !important; }

    /* Code blocks */
    code, pre { background: rgba(0,0,0,0.3) !important; border-radius: 8px !important; }

    /* Success / info boxes */
    .stSuccess { background: rgba(20,184,166,0.1) !important; border-left: 3px solid #14b8a6 !important; }
    .stInfo    { background: rgba(59,130,246,0.1) !important; border-left: 3px solid #3b82f6 !important; }
    .stWarning { background: rgba(245,158,11,0.1) !important; border-left: 3px solid #f59e0b !important; }
    .stError   { background: rgba(239,68,68,0.1) !important;  border-left: 3px solid #ef4444 !important; }

    /* Metric labels */
    [data-testid="metric-container"] {
        background: rgba(15,23,42,0.7) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        padding: 0.8rem !important;
    }

    .section-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        color: #14b8a6;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-header">
        <h1>🏏 Cricketers Self-Query RAG</h1>
        <p>Intelligent cricket knowledge base powered by Self-Query retrieval · Groq LLaMA-3 70B · Qdrant · LangChain</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Example questions
# ---------------------------------------------------------------------------

EXAMPLES = {
    "📋 Filter/List": [
        "List all left-arm fast bowlers",
        "Show me all Australian batsmen",
        "List all wicket-keeper batsmen from India",
    ],
    "🔢 Count": [
        "How many Pakistani all-rounders are there?",
        "How many Sri Lankan players are in the database?",
        "How many bowlers from Australia are in the database?",
    ],
    "⏳ Era": [
        "Which players were active in 2024?",
        "Who played cricket in the 2000s?",
        "Players active in the 1990s",
    ],
    "🔍 Semantic": [
        "Who is known for aggressive, attacking batting?",
        "Which players revolutionised their position?",
        "Who is considered the greatest all-rounder ever?",
    ],
    "🎯 Hybrid": [
        "Pakistani all-rounders known for aggressive batting",
        "Indian spinners who dominated in the 2000s",
        "Australian batsmen known for elegant technique",
    ],
    "🚫 Out-of-Scope": [
        "Who won the 2023 IPL final?",
        "What is the current ICC cricket ranking?",
    ],
}

st.markdown('<p class="section-title">💡 Try an example question</p>', unsafe_allow_html=True)

# Render examples as clickable buttons in rows
if "question_input" not in st.session_state:
    st.session_state.question_input = ""

cols_per_row = 3
all_examples = [(cat, q) for cat, qs in EXAMPLES.items() for q in qs]

# Flatten into rows
for i in range(0, len(all_examples), cols_per_row):
    chunk = all_examples[i : i + cols_per_row]
    btn_cols = st.columns(len(chunk))
    for col, (cat, q) in zip(btn_cols, chunk):
        label = f"{cat.split()[0]} {q[:42]}{'…' if len(q) > 42 else ''}"
        if col.button(label, key=f"ex_{i}_{q[:15]}"):
            st.session_state.question_input = q

st.divider()

# ---------------------------------------------------------------------------
# Main input
# ---------------------------------------------------------------------------

st.markdown('<p class="section-title">🎙️ Ask a question</p>', unsafe_allow_html=True)

question = st.text_input(
    label="question",
    value=st.session_state.question_input,
    placeholder="e.g. List all left-arm fast bowlers from Pakistan",
    label_visibility="collapsed",
    key="main_input",
)

col_submit, col_compare, _ = st.columns([1, 2, 4])
with col_submit:
    run_btn = st.button("🔍 Ask", use_container_width=True)
with col_compare:
    compare_mode = st.toggle(
        "⚡ Compare vs Naive Semantic-Only RAG",
        value=False,
        help="Run both Self-Query and plain top-5 semantic search side by side to see why self-query matters.",
    )

# ---------------------------------------------------------------------------
# Badge helper
# ---------------------------------------------------------------------------

def routing_badge(routing: str) -> str:
    mapping = {
        "filter_only": ("badge-filter", "🎯 Filter Only"),
        "hybrid": ("badge-hybrid", "🔀 Hybrid"),
        "semantic_only": ("badge-semantic", "🔍 Semantic"),
        "out_of_scope": ("badge-oos", "🚫 Out of Scope"),
    }
    cls, label = mapping.get(routing, ("badge-semantic", routing))
    return f'<span class="routing-badge {cls}">{label}</span>'


def render_result_count_metric(result: dict) -> None:
    exact_count = result.get("exact_count")
    routing = result.get("routing")
    if exact_count is None:
        return

    if routing == "filter_only":
        st.metric("Exact count (code-computed)", exact_count)
        return

    if routing == "hybrid":
        extracted = result.get("extracted") or {}
        filters = extracted.get("filters") or {}
        era_filter = extracted.get("era_filter") or {}
        has_filter_narrowing = bool(
            filters.get("country")
            or filters.get("role")
            or filters.get("style_keyword")
            or era_filter.get("type")
        )
        if has_filter_narrowing:
            st.metric("Exact count (code-computed)", exact_count)
        else:
            st.metric("Candidates retrieved", len(result.get("results", [])))
        return


# ---------------------------------------------------------------------------
# Execute on button click or Enter
# ---------------------------------------------------------------------------

if run_btn and question.strip():
    st.divider()

    if compare_mode:
        col_self, col_naive = st.columns(2)

        with col_self:
            st.markdown(
                """
                <div style="background:rgba(20,184,166,0.08);border:1px solid rgba(20,184,166,0.2);
                border-radius:12px;padding:0.7rem 1rem;margin-bottom:1rem;">
                <strong style="color:#14b8a6;">✅ Self-Query RAG</strong>
                <span style="color:#64748b;font-size:0.8rem;margin-left:0.5rem;">(filter-aware · exact counts · complete lists)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("Running self-query pipeline…"):
                self_result = answer_question(question)

            st.markdown(routing_badge(self_result["routing"]), unsafe_allow_html=True)
            st.markdown(f"**Answer:**\n\n{self_result['answer']}")
            render_result_count_metric(self_result)

        with col_naive:
            st.markdown(
                """
                <div style="background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.2);
                border-radius:12px;padding:0.7rem 1rem;margin-bottom:1rem;">
                <strong style="color:#f87171;">⚠️ Naive Semantic RAG</strong>
                <span style="color:#64748b;font-size:0.8rem;margin-left:0.5rem;">(top-5 only · no filters · counts may be wrong)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.spinner("Running naive pipeline…"):
                naive_result = answer_question_naive(question, top_k=5)

            st.markdown(
                '<span class="routing-badge badge-semantic">🔍 Semantic Only</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Answer:**\n\n{naive_result['answer']}")
            st.caption(
                f"⚠️ Retrieved only top-5 results — lists may be incomplete and "
                f"counts may be incorrect."
            )

        # Details (self-query only)
        with st.expander("🔎 Self-Query Details — filters, routing & matched records"):
            st.subheader("Extracted Filters (from LLM)")
            st.json(self_result["extracted"])
            st.subheader(f"Routing Decision: `{self_result['routing']}`")
            st.write(
                f"**{len(self_result['results'])}** player(s) matched the filters."
            )
            if self_result["results"]:
                st.subheader("Matched Records")
                for p in self_result["results"]:
                    st.markdown(
                        f"- **{p.get('name')}** ({p.get('country')}) — "
                        f"{p.get('role')} | {p.get('style')} | Era: {p.get('era')}"
                    )
            if self_result.get("prompt_used"):
                st.subheader("Generation Prompt (truncated)")
                st.text(self_result["prompt_used"][:1200] + ("…" if len(self_result["prompt_used"]) > 1200 else ""))

    else:
        # Single column self-query mode
        with st.spinner("Running self-query pipeline…"):
            result = answer_question(question)

        st.markdown(routing_badge(result["routing"]), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="answer-card">
            {result['answer'].replace(chr(10), '<br>')}
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_result_count_metric(result)

        with st.expander("🔎 Details — filters, routing & matched records"):
            tab1, tab2, tab3 = st.tabs(["Extracted Filters", "Matched Records", "Generation Prompt"])

            with tab1:
                st.subheader("LLM-extracted filters")
                st.json(result["extracted"])
                st.subheader(f"Routing: `{result['routing']}`")

            with tab2:
                count = len(result["results"])
                st.write(f"**{count}** player(s) matched.")
                if result["results"]:
                    for p in result["results"]:
                        st.markdown(
                            f"- **{p.get('name')}** ({p.get('country')}) — "
                            f"{p.get('role')} | {p.get('style')} | Era: {p.get('era')} | "
                            f"{p.get('achievements', '')[:80]}"
                        )

            with tab3:
                if result.get("prompt_used"):
                    st.text(result["prompt_used"][:2000] + ("…" if len(result["prompt_used"]) > 2000 else ""))
                else:
                    st.write("No prompt (out-of-scope or zero-result short-circuit).")

elif run_btn and not question.strip():
    st.warning("Please enter a question first.")

# ---------------------------------------------------------------------------
# Sidebar — architecture info
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🏗️ Architecture")
    st.markdown(
        """
        **Stack**
        - Vector DB: Qdrant (local embedded)
        - Embeddings: all-MiniLM-L6-v2 (384-dim)
        - LLM: Groq LLaMA-3 70B (8192 ctx)
        - Orchestration: LangChain
        - Observability: Langfuse
        - UI: Streamlit

        **Self-Query Pipeline**
        1. Extract filters (LLM, temp=0)
        2. Route: filter_only / hybrid / semantic / OOS
        3. Retrieve (scroll for lists, query_points for semantic)
        4. Zero-result guardrail
        5. Generate (LLM, temp=0.3)
        6. Log to Langfuse

        **Data**
        - 129 world cricketers
        - 1 Qdrant point per player (no chunking)
        - Era stored as integer range for overlap filters
        - All metadata normalised to lowercase at ingest
        """
    )
    st.markdown("---")
    st.markdown("### 📁 Quick Setup")
    st.code(
        "pip install -r requirements.txt\n"
        "# copy .env.example → .env and set GROQ_API_KEY\n"
        "python -m src.ingestion\n"
        "streamlit run src/app.py",
        language="bash",
    )
