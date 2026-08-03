"""
pipeline.py - Orchestrates the Self-Query RAG flow
"""
from src.retrieval import extract_filters, should_use_filters, retrieve, retrieve_with_filters
from src.generation import generate, _get_groq
from src.ingestion import DATA_FILE, norm
import pandas as pd
from src.monitoring import observe

# Standard fallback messages
_OUT_OF_SCOPE_MSG = "This question appears to be outside the scope of the World Cricketers dataset. I can only answer questions about the players in this database — such as their roles, batting/bowling styles, countries, eras, and achievements. Questions about match results, tournament outcomes, or current news are out of scope."
_NO_MATCH_MSG = "No players were found in this dataset that match those filters. Please try different criteria (e.g. check the country, role, or era)."

_distinct_cache = None

def _get_distinct_values() -> dict:
    """Helper to dynamically generate the allowed schema values for the LLM."""
    global _distinct_cache
    if _distinct_cache:
        return _distinct_cache

    try:
        df = pd.read_excel(DATA_FILE)
        # Assuming ingestion.py's normalizations apply
        df['Country_norm'] = df['Country'].astype(str).apply(norm)
        df['Role_norm'] = df['Role'].astype(str).apply(norm)
        df['Style_norm'] = df['Batting/Bowling Style'].astype(str).apply(norm)
        
        _distinct_cache = {
            "country": sorted([c for c in df['Country_norm'].unique() if c != 'nan']),
            "role": sorted([r for r in df['Role_norm'].unique() if r != 'nan']),
            "batting_style": sorted([s for s in df['Style_norm'].unique() if s != 'nan']),
        }
        return _distinct_cache
    except Exception:
        # Fallback if file isn't found
        return {"country": [], "role": [], "batting_style": []}

@observe(name="answer_question")
def answer_question(question: str) -> dict:
    """
    Main orchestration logic:
      1. Extract filters via LLM
      2. Route & Retrieve
      3. Generate exact count if needed
      4. Answer via LLM
    """
    
    # Step 1: Extract filters
    distinct = _get_distinct_values()
    extracted = extract_filters(question, distinct)

    # Step 2: Route
    routing = should_use_filters(extracted)
    
    # Check out of scope early
    if routing == "out_of_scope":
        return {
            "answer": _OUT_OF_SCOPE_MSG,
            "routing": "out_of_scope",
            "extracted": extracted,
            "results": [],
            "exact_count": None,
            "prompt_used": "",
        }
        
    # Step 3: Retrieve
    results = []
    if routing == "semantic_only":
        semantic_q = extracted.get("semantic_query") or question
        results = retrieve(semantic_q, top_k=10)
    else:
        results = retrieve_with_filters(extracted, top_k=1000)

    # Step 4: Zero-result guardrail
    if not results and routing != "semantic_only":
        exact_count = len(results)
        if extracted.get("is_list_or_count"):
            answer, prompt_used = generate(question, results, exact_count)
            return {
                "answer": answer,
                "routing": routing,
                "extracted": extracted,
                "results": results,
                "exact_count": exact_count,
                "prompt_used": prompt_used,
            }
        return {
            "answer": _NO_MATCH_MSG,
            "routing": routing,
            "extracted": extracted,
            "results": [],
            "exact_count": None,
            "prompt_used": "",
        }

    # Step 5: Compute exact_count for list/count questions (code-side, not LLM-side)
    exact_count = None
    if extracted.get("is_list_or_count") and results:
        exact_count = len(results)

    # Step 6: Generate
    answer, prompt_used = generate(question, results, exact_count)

    return {
        "answer": answer,
        "routing": routing,
        "extracted": extracted,
        "results": results,
        "exact_count": exact_count,
        "prompt_used": prompt_used,
    }


# ---------------------------------------------------------------------------
# Naive baseline (deliberately limited — used for comparison demo only)
# ---------------------------------------------------------------------------

@observe(name="answer_question_naive")
def answer_question_naive(question: str, top_k: int = 5) -> dict:
    """
    Naive semantic-only RAG baseline — no filters, no exact counting.
    """
    results = retrieve(question, top_k=top_k)
    
    # Just format them as a blob without enforcing constraints
    context_text = "\n\n".join([
        f"Name: {r['name']}\nCountry: {r.get('country','')}\nRole: {r.get('role','')}\nStyle: {r.get('style','')}\nEra: {r.get('era','')}\nAchievements: {r.get('achievements','')}\nBackground: {r.get('background','')}"
        for r in results
    ])
    
    prompt = f"Answer the user's question using ONLY the provided context.\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer:"
    
    llm = _get_groq()
    try:
        from langchain_core.messages import HumanMessage
        resp = llm.invoke([HumanMessage(content=prompt)])
        answer = str(resp.content)
    except Exception as e:
        answer = f"[LLM Error] {e}"
        
    return {
        "answer": answer,
        "routing": "naive",
        "extracted": None,
        "results": results,
        "exact_count": None,
        "prompt_used": prompt,
    }
