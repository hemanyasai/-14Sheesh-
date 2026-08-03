"""
ingestion.py — Load World_Cricketers.xlsx, normalise, embed, and upsert to Qdrant.

Design decisions (documented here so they aren't mistaken for oversights):
  - NO CHUNKING: Each player is a single Qdrant point.  A player's entire record
    is well within the effective context window of the embedding model
    (all-MiniLM-L6-v2, 256-token window).  Chunking would break the natural
    unit of retrieval (the player) and make filters unreliable.
  - LOWERCASE NORMALISATION at ingest: every string metadata field is stored both
    in original case (for display) and normalised lowercase (for exact-match
    filtering).  Without this, "Pakistan" vs "pakistan" causes silent filter misses
    — the single most common failure mode in production self-query RAG.
  - ERA OVERLAP LOGIC: era_start/era_end are stored as integers so Qdrant Range
    filters can do proper numeric comparisons (overlap, not one-sided bounds).
"""

from __future__ import annotations

import os
import re
import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.embeddings import get_embedding_model

load_dotenv()

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent          # BuildwithRAG/
DATA_FILE = BASE_DIR / "data" / "World_Cricketers.xlsx"
QDRANT_PATH = str(BASE_DIR / "qdrant_data")
COLLECTION_NAME = "world_cricketers"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_SIZE = 384
CURRENT_YEAR = datetime.datetime.now().year

# ---------------------------------------------------------------------------
# Column mapping (COLUMN_ALIASES allows quick adjustment if column names change)
# ---------------------------------------------------------------------------

COLUMN_ALIASES: dict[str, list[str]] = {
    # canonical name  : [possible actual column names, in priority order]
    "name":            ["Name", "Player Name", "Cricketer"],
    "country":         ["Country", "Nation", "Nationality"],
    "role":            ["Role", "Player Role", "Position"],
    "style":           ["Batting/Bowling Style", "Style", "Batting Style"],
    "era":             ["Era", "Career", "Years Active"],
    "achievements":    ["Notable Achievements", "Achievements", "Notable"],
    "background":      ["Background", "Bio", "Description"],
}


def _resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the first alias that exists in df.columns, or None."""
    for alias in COLUMN_ALIASES.get(canonical, []):
        if alias in df.columns:
            return alias
    return None


# ---------------------------------------------------------------------------
# Era parsing
# ---------------------------------------------------------------------------

def parse_era(era_str: str) -> tuple[int, int]:
    """
    Parse a free-text era string into (start_year, end_year) integers.

    Handles:
      "1928-1948"       -> (1928, 1948)
      "2010-present"    -> (2010, CURRENT_YEAR)
      "2010–present"    -> same (en-dash)
      "1990s"           -> (1990, 1999)
      "late 1980s"      -> (1980, 1989)
      "1980"            -> (1980, 1980)
      Any fallback      -> (0, CURRENT_YEAR)
    """
    if not isinstance(era_str, str):
        return (0, CURRENT_YEAR)

    s = era_str.strip()

    # Normalise en/em dashes to hyphen
    s = s.replace("–", "-").replace("—", "-")

    # Explicit year range: "YYYY-YYYY" or "YYYY-present/current"
    m = re.match(r"(\d{4})\s*-\s*(\d{4}|\bpresent\b|\bcurrent\b)", s, re.IGNORECASE)
    if m:
        start = int(m.group(1))
        end_raw = m.group(2).lower()
        end = CURRENT_YEAR if end_raw in ("present", "current") else int(end_raw)
        return (start, end)

    # Decade shorthand: "1990s", "late 1980s", "early 2000s"
    m = re.search(r"(\d{4})s", s)
    if m:
        decade = int(m.group(1))
        return (decade, decade + 9)

    # Single year
    m = re.fullmatch(r"\d{4}", s)
    if m:
        y = int(s)
        return (y, y)

    # Fallback
    return (0, CURRENT_YEAR)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def norm(value) -> str:
    """Strip and lowercase a value for use in filter comparisons."""
    try:
        import math
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower()


def safe_str(value) -> str:
    """Strip a value for display (original casing preserved)."""
    try:
        import math
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


# ---------------------------------------------------------------------------
# Distinct values (fed into the self-query LLM prompt as allowed vocabulary)
# ---------------------------------------------------------------------------

def get_distinct_values(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Return sorted unique normalised values for each filterable field.
    These are injected into the LLM prompt so it can never invent filter
    values that don't exist in the data.
    """
    col_country = _resolve_column(df, "country")
    col_role    = _resolve_column(df, "role")
    col_style   = _resolve_column(df, "style")

    def _distinct(col: str | None) -> list[str]:
        if col is None:
            return []
        return sorted({norm(v) for v in df[col].dropna() if norm(v)})

    return {
        "country":       _distinct(col_country),
        "role":          _distinct(col_role),
        "batting_style": sorted({
            norm(v) for raw in (df[col_style].dropna() if col_style else [])
            for v in [safe_str(raw)]
            if norm(v)
        }),
    }


# ---------------------------------------------------------------------------
# Embedding text builder
# ---------------------------------------------------------------------------

def build_embedding_text(row: dict) -> str:
    """
    Concatenate key fields into natural English sentences.
    This is the semantic search surface — richness here improves fuzzy queries.
    """
    parts = []
    if row.get("name"):
        parts.append(f"{row['name']} is a cricketer from {row.get('country', 'unknown')}.")
    if row.get("role"):
        parts.append(f"They play as a {row['role']}.")
    if row.get("style"):
        parts.append(f"Their batting/bowling style is {row['style']}.")
    if row.get("achievements"):
        parts.append(f"Notable achievements: {row['achievements']}.")
    if row.get("background"):
        parts.append(row["background"])
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def run_ingestion() -> None:
    print("=" * 60)
    print("STEP 0 — Load and inspect the raw data")
    print("=" * 60)

    df = pd.read_excel(DATA_FILE)
    print("df.columns.tolist():", df.columns.tolist())
    print()
    print(df.head(5).to_string())
    print()

    # Resolve actual column names
    col_name    = _resolve_column(df, "name")
    col_country = _resolve_column(df, "country")
    col_role    = _resolve_column(df, "role")
    col_style   = _resolve_column(df, "style")
    col_era     = _resolve_column(df, "era")
    col_ach     = _resolve_column(df, "achievements")
    col_bg      = _resolve_column(df, "background")

    missing = [k for k, v in {
        "name": col_name, "country": col_country, "role": col_role,
        "era": col_era, "achievements": col_ach, "background": col_bg,
    }.items() if v is None]
    if missing:
        print(f"[WARNING] Could not resolve columns: {missing}. "
              "Update COLUMN_ALIASES in ingestion.py.")

    print(f"\nResolved columns -> name='{col_name}', country='{col_country}', "
          f"role='{col_role}', style='{col_style}', era='{col_era}', "
          f"achievements='{col_ach}', background='{col_bg}'")

    distinct = get_distinct_values(df)
    print("\nDistinct values (for self-query vocabulary):")
    for k, v in distinct.items():
        print(f"  {k}: {v}")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1 — Initialise Qdrant collection")
    print("=" * 60)

    client = QdrantClient(path=QDRANT_PATH)

    # Recreate collection for a clean ingest run
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'.")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Created collection '{COLLECTION_NAME}' (size={VECTOR_SIZE}, COSINE).")

    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2 — Embed & upsert")
    print("=" * 60)

    embeddings_model = get_embedding_model(EMBEDDING_MODEL)

    points: list[PointStruct] = []
    for idx, row in df.iterrows():
        raw: dict = {
            "name":         safe_str(row.get(col_name, "")),
            "country":      safe_str(row.get(col_country, "")) if col_country else "",
            "role":         safe_str(row.get(col_role, "")) if col_role else "",
            "style":        safe_str(row.get(col_style, "")) if col_style else "",
            "era":          safe_str(row.get(col_era, "")) if col_era else "",
            "achievements": safe_str(row.get(col_ach, "")) if col_ach else "",
            "background":   safe_str(row.get(col_bg, "")) if col_bg else "",
        }

        era_start, era_end = parse_era(raw["era"])

        # Build normalised versions for exact-match filtering
        payload = {
            # Display (original casing)
            "name":         raw["name"],
            "country":      raw["country"],
            "role":         raw["role"],
            "style":        raw["style"],
            "era":          raw["era"],
            "achievements": raw["achievements"],
            "background":   raw["background"],
            # Normalised (lowercase, stripped) — used by self-query filters
            "country_norm": norm(raw["country"]),
            "role_norm":    norm(raw["role"]),
            "style_norm":   norm(raw["style"]),
            # Era integers — used for Range filters
            "era_start":    era_start,
            "era_end":      era_end,
        }

        embedding_text = build_embedding_text(raw)
        vector = embeddings_model.embed_query(embedding_text)

        points.append(PointStruct(id=int(idx), vector=vector, payload=payload))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Upserted {len(points)} points into '{COLLECTION_NAME}'.")
    print("\nIngestion complete.")

    # Quick verification
    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"Verified: {count} points in collection.")


if __name__ == "__main__":
    run_ingestion()
