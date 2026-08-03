"""
retrieval.py — Self-query retrieval implementation.

The self-query mechanism is implemented *explicitly* here (not by importing a
pre-built SelfQueryRetriever) so the filter-extraction, routing, and retrieval
logic are visible, testable, and demoable.

Key design decisions:
  - extract_filters() uses temperature=0 for deterministic routing.
  - Malformed LLM output falls back to pure semantic search (never crashes).
  - filter_only (list/count) questions use scroll(), not query_points(),
    to guarantee complete/exact results — this is the correctness-critical detail.
  - Era filtering is an OVERLAP check (not one-sided bound):
      active in year Y  → era_start <= Y AND era_end >= Y
      decade D0–D9      → era_start <= D9 AND era_end >= D0
  - Style is stored as a single composite field (e.g. "Left-arm fast-medium").
    The LLM extracts a keyword phrase; we use substring matching + exact match fallback.
  - Embeddings model is cached at module level to avoid re-loading on every call.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter, FieldCondition, MatchValue, MatchAny, Range,
)

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
QDRANT_PATH = str(BASE_DIR / "qdrant_data")
COLLECTION_NAME = "world_cricketers"

# ---------------------------------------------------------------------------
# Shared clients — module-level singletons to avoid reinitialisation overhead
# ---------------------------------------------------------------------------

_qdrant_client: QdrantClient | None = None
_groq_llm: ChatGroq | None = None
_embeddings_model = None


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
    return _qdrant_client


def _get_groq() -> ChatGroq:
    global _groq_llm
    if _groq_llm is None:
        _groq_llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,  # deterministic — same question routes the same way every run
            api_key=os.getenv("GROQ_API_KEY"),
            max_retries=1,
        )
    return _groq_llm


def _get_embeddings():
    global _embeddings_model
    if _embeddings_model is None:
        from src.embeddings import get_embedding_model
        _embeddings_model = get_embedding_model("all-MiniLM-L6-v2")
    return _embeddings_model


# ---------------------------------------------------------------------------
# Filter extraction — the self-query LLM call
# ---------------------------------------------------------------------------

EXTRACT_FILTERS_SYSTEM_PROMPT = """You are a structured data extraction assistant for a cricket player database.

Your job: parse the user's question and return a strict JSON object with filter fields.

ALLOWED VALUES (use ONLY these — never invent new ones):
{vocab_block}

JSON schema (return ONLY valid JSON, no markdown fences, no extra text):
{{
  "filters": {{
    "country": <string | null>,
    "role": <string | null>,
    "style_keyword": <string | null>
  }},
  "era_filter": {{
    "type": "active_year" | "year_range" | "before_year" | "after_year" | null,
    "year": <int | null>,
    "start": <int | null>,
    "end": <int | null>
  }},
  "semantic_query": "<string>",
  "is_list_or_count": <bool>,
  "out_of_scope": <bool>
}}

RULES:
1. Only set a filter field if the user clearly specifies that dimension. Leave others null.
2. Normalise country and role values to lowercase to match the ALLOWED VALUES above exactly.
3. For country and role: only use values from the ALLOWED VALUES lists above.
4. For style_keyword: extract a lowercase keyword phrase (e.g. "left-arm fast", "leg spin",
   "off spin", "right-arm fast-medium", "left-hand bat"). This will be used for substring
   matching against the style field — you don't need an exact match from the allowed list.
5. For era:
   - "active in 2024" → {{"type": "active_year", "year": 2024}}
   - "played in 1990" → {{"type": "active_year", "year": 1990}}
   - "active in 1990" → {{"type": "active_year", "year": 1990}}
   - "active in the 2000s" → {{"type": "year_range", "start": 2000, "end": 2009}}
   - "1990s players" → {{"type": "year_range", "start": 1990, "end": 1999}}
   - "after 1990" → {{"type": "after_year", "year": 1990}}
   - "since 1990" → {{"type": "after_year", "year": 1990}}
   - "active before 1950" → {{"type": "before_year", "year": 1950}}
   - "active after 1950" → {{"type": "after_year", "year": 1950}}
   - No era mention → {{"type": null}}
6. Set is_list_or_count=true for questions asking for a list or count
   ("list all", "how many", "who are all", "show me all").
7. Set out_of_scope=true if the question is completely outside cricket player data
   (e.g. match results, IPL finals, tournament scores, current news, politics).
8. Set semantic_query to the user's question for fuzzy/descriptive queries
   (personality, playing style description, legacy, impact, etc.).
   Leave it EMPTY ("") for pure filter/count questions with no descriptive element.
9. For hybrid queries (filter + description), set both filters AND semantic_query.
"""


def _build_vocab_block(distinct_values: dict) -> str:
    lines = []
    for field, values in distinct_values.items():
        if field != "batting_style":  # style is handled via substring
            lines.append(f"  {field}: {json.dumps(values)}")
    return "\n".join(lines)


def _infer_era_filter_from_text(query: str) -> dict | None:
    """Infer era filters directly from the user's wording before the LLM response is used."""
    q = (query or "").strip().lower()

    m_before = re.search(r"\b(before|prior to|earlier than)\s+(\d{4})\b", q)
    if m_before:
        year = int(m_before.group(2))
        return {"type": "before_year", "year": year, "start": None, "end": None}

    m_after = re.search(r"\b(after|since)\s+(\d{4})\b", q)
    if m_after:
        year = int(m_after.group(2))
        return {"type": "after_year", "year": year, "start": None, "end": None}

    m_overlap = re.search(r"\b(?:played|playing|active|was active|were active)\s+(?:in|during)\s+(\d{4})\b", q)
    if m_overlap:
        year = int(m_overlap.group(1))
        return {"type": "active_year", "year": year, "start": None, "end": None}

    return None


def extract_filters(query: str, distinct_values: dict) -> dict:
    """
    Send the query to Groq at temperature=0 and parse structured filter JSON.

    Falls back to a safe default on any parse failure — a slightly degraded
    semantic-only answer beats a crash mid-demo.
    """
    _FALLBACK = {
        "filters": {"country": None, "role": None, "style_keyword": None},
        "era_filter": {"type": None, "year": None, "start": None, "end": None},
        "semantic_query": query,
        "is_list_or_count": False,
        "out_of_scope": False,
    }

    try:
        llm = _get_groq()
        vocab_block = _build_vocab_block(distinct_values)
        system_prompt = EXTRACT_FILTERS_SYSTEM_PROMPT.format(vocab_block=vocab_block)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        response = llm.invoke(messages)
        raw_text = response.content.strip()

        # Strip markdown fences if present (```json ... ```)
        raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
        raw_text = re.sub(r"\s*```$", "", raw_text)
        raw_text = raw_text.strip()

        # Handle extra text before/after JSON
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            raw_text = json_match.group(0)

        parsed = json.loads(raw_text)

        # Ensure required keys exist (merge with fallback for any missing ones)
        result = {**_FALLBACK, **parsed}
        result["filters"] = {**_FALLBACK["filters"], **parsed.get("filters", {})}
        result["era_filter"] = {**_FALLBACK["era_filter"], **parsed.get("era_filter", {})}

        # Override the LLM when the query explicitly uses before/after year phrasing.
        inferred_era_filter = _infer_era_filter_from_text(query)
        if inferred_era_filter is not None:
            result["era_filter"] = {**_FALLBACK["era_filter"], **inferred_era_filter}

        # Normalise filter string values to lowercase
        for k, v in result["filters"].items():
            if isinstance(v, str):
                result["filters"][k] = v.strip().lower()

        return result

    except Exception as e:
        print(f"[retrieval] extract_filters parse failure: {e!r}. Using fallback.")
        return _FALLBACK


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def should_use_filters(extracted: dict) -> str:
    """
    Route to one of: 'out_of_scope' | 'filter_only' | 'hybrid' | 'semantic_only'
    """
    if extracted.get("out_of_scope"):
        return "out_of_scope"

    filters = extracted.get("filters", {})
    era = extracted.get("era_filter", {})
    semantic = extracted.get("semantic_query", "").strip()
    is_list_or_count = extracted.get("is_list_or_count", False)

    has_hard_filter = (
        bool(filters.get("country")) or
        bool(filters.get("role"))
    )
    has_style_keyword = bool(filters.get("style_keyword"))
    has_era = era.get("type") is not None
    has_filter = has_hard_filter or has_style_keyword or has_era

    if has_filter and (not semantic or is_list_or_count):
        return "filter_only"
    if has_filter and semantic:
        return "hybrid"
    return "semantic_only"


# ---------------------------------------------------------------------------
# Qdrant filter builder (for exact metadata fields only)
# ---------------------------------------------------------------------------

def _build_qdrant_filter(extracted: dict, skip_style: bool = False) -> Filter | None:
    """
    Build a Qdrant Filter from extracted filter fields.

    Era overlap semantics:
      - active_year Y : era_start <= Y AND era_end >= Y
      - year_range S–E: era_start <= E AND era_end >= S  (any overlap counts)

    Note: style_keyword is handled via post-filter substring matching (not here),
    because Qdrant MatchValue requires exact matches, and style values in the data
    are composite (e.g. "Left-arm fast-medium", "RH bat / Right-arm fast").
    """
    conditions = []

    filters = extracted.get("filters", {})

    if filters.get("country"):
        conditions.append(
            FieldCondition(key="country_norm", match=MatchValue(value=filters["country"]))
        )

    if filters.get("role"):
        conditions.append(
            FieldCondition(key="role_norm", match=MatchValue(value=filters["role"]))
        )

    era = extracted.get("era_filter", {})
    era_type = era.get("type")

    if era_type == "active_year":
        year = era.get("year")
        if year:
            # era_start <= year AND era_end >= year
            conditions.append(FieldCondition(key="era_start", range=Range(lte=year)))
            conditions.append(FieldCondition(key="era_end", range=Range(gte=year)))

    elif era_type == "year_range":
        s = era.get("start")
        e = era.get("end")
        if s and e:
            # era_start <= range_end AND era_end >= range_start (overlap, not containment)
            conditions.append(FieldCondition(key="era_start", range=Range(lte=e)))
            conditions.append(FieldCondition(key="era_end", range=Range(gte=s)))

    elif era_type == "before_year":
        year = era.get("year")
        if year is not None:
            conditions.append(FieldCondition(key="era_end", range=Range(lt=year)))

    elif era_type == "after_year":
        year = era.get("year")
        if year is not None:
            conditions.append(FieldCondition(key="era_start", range=Range(gt=year)))

    if not conditions:
        return None

    return Filter(must=conditions)


# ---------------------------------------------------------------------------
# Style post-filter (substring matching)
# ---------------------------------------------------------------------------

def _apply_style_filter(records: list[dict], style_keyword: str | None) -> list[dict]:
    """
    Post-filter records by style_keyword using case-insensitive substring match.
    Returns all records if no style_keyword is specified.
    """
    if not style_keyword:
        return records
    kw = style_keyword.lower()
    return [r for r in records if kw in (r.get("style_norm") or "").lower()]


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = 10) -> list[dict]:
    """
    Plain semantic search — no filters.  Used for semantic_only routing.
    """
    emb = _get_embeddings()
    vector = emb.embed_query(query)

    client = _get_qdrant()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=top_k,
        with_payload=True,
    )
    return [r.payload for r in results.points]


def retrieve_with_filters(extracted: dict, top_k: int = 10) -> list[dict]:
    """
    Filter-aware retrieval.

    - filter_only (list/count): uses scroll() to get ALL matching points, no top-k cap.
      Then applies style substring filter as a post-processing step.
    - hybrid: uses query_points() with query_filter for Qdrant-side filtering,
      then applies style substring filter on the ranked results.
    """
    routing = should_use_filters(extracted)
    qdrant_filter = _build_qdrant_filter(extracted)
    style_keyword = (extracted.get("filters") or {}).get("style_keyword")
    client = _get_qdrant()

    if routing == "filter_only":
        # scroll() returns ALL matching points — critical for exact counts
        records, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=qdrant_filter,
            limit=1000,          # effectively unlimited for 129-row dataset
            with_payload=True,
            with_vectors=False,
        )
        results = [r.payload for r in records]
        # Apply style substring post-filter
        results = _apply_style_filter(results, style_keyword)
        return results

    elif routing == "hybrid":
        emb = _get_embeddings()
        semantic_q = extracted.get("semantic_query", "") or ""
        vector = emb.embed_query(semantic_q)

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=qdrant_filter,
            limit=top_k * 3,    # over-fetch to account for style post-filter
            with_payload=True,
        )
        records = [r.payload for r in results.points]
        records = _apply_style_filter(records, style_keyword)
        return records[:top_k]

    else:
        # Fallback to plain semantic
        semantic_q = extracted.get("semantic_query", "") or ""
        return retrieve(semantic_q, top_k)


# ---------------------------------------------------------------------------
# Public entry point with style-only fallback for robustness
# ---------------------------------------------------------------------------

def retrieve_with_style_substring(extracted: dict, top_k: int = 10) -> list[dict]:
    """
    Primary retrieval entry point.

    If normal filter retrieval returns zero results AND we have a style_keyword,
    attempt a broader semantic search with substring post-filter as fallback.
    This handles cases like "left-arm fast bowlers" where the style_keyword
    "left-arm fast" matches "left-arm fast-medium" via substring but wouldn't
    match via exact filter.
    """
    results = retrieve_with_filters(extracted, top_k=top_k)

    if not results:
        style_keyword = (extracted.get("filters") or {}).get("style_keyword")
        if style_keyword:
            # Broader fallback: search entire collection, then post-filter by style
            all_records = retrieve(style_keyword, top_k=200)
            results = _apply_style_filter(all_records, style_keyword)

    return results
