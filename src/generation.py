"""
generation.py — LLM answer generation using Groq (llama3-70b-8192).

Key design points:
  - build_system_prompt() instructs the LLM to:
      * answer ONLY from the provided context (no hallucination).
      * list EVERY name given — never truncate or summarise a list itself.
      * use the exact_count value supplied by code, not recount from bullet text.
  - temperature=0.3 for answers (slightly creative phrasing, stable facts).
  - Returns both the answer and the full prompt used (for Langfuse tracing).
"""

from __future__ import annotations

import os
import re
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

_groq_llm: ChatGroq | None = None


def _get_groq():
    """Returns a ChatGroq instance for standard generation, keeping temperature low."""
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    _groq_llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    return _groq_llm


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    return (
        "You are a knowledgeable cricket analyst assistant. "
        "Answer questions ONLY using the context provided below — do not use "
        "any outside knowledge or make up information not in the context.\n\n"
        "IMPORTANT RULES:\n"
        "1. When listing players, list EVERY name present in the context. "
        "   Never truncate, summarise, or say 'and others' — if the context has 30 names, list all 30.\n"
        "2. When an 'Exact count computed from the database' line is provided, "
        "   use that number as your answer for count questions — do NOT recount "
        "   the bullet points yourself, as the code-computed number is authoritative.\n"
        "3. Do not add commentary, caveats, or notes about players beyond what the question asks. "
        "   If the context contains 28 players, list all 28 and stop — do not append editorial notes "
        "   speculating about which players' eras 'ended before' the query year or whether their retirement "
        "   status is unclear. The retrieved context has already been filtered correctly; trust it and do not "
        "   second-guess or annotate it.\n"
        "4. For list or count questions, answer with only the requested result. "
        "   If the user asks for a list, output only the names (no introduction, no conclusion, no caveats). "
        "   If the user asks for a count, output a full sentence such as 'There are 0 matching players in this dataset.' "
        "   or 'There is 1 matching player: Mushfiqur Rahim.'\n"
        "5. If the context is empty or says no matches found, say so clearly. "
        "   Do not guess or invent players.\n"
        "6. Be concise but complete."
    )


def _is_list_or_count_question(question: str) -> bool:
    q = (question or "").lower()
    return any(token in q for token in [
        "which players",
        "who are",
        "list",
        "how many",
        "show me all",
        "show all",
        "count",
    ])


def build_prompt(
    question: str,
    retrieved: list[dict],
    exact_count: int | None = None,
) -> str:
    """
    Format retrieved player records into a prompt for the LLM.

    If exact_count is provided (for list/count questions), a clearly marked
    authoritative count line is injected — the LLM is told to use this number,
    not recount from the bullets.
    """
    if not retrieved:
        context_block = "(No matching players found in the database.)"
    else:
        lines = []
        for p in retrieved:
            line = (
                f"• **{p.get('name', 'Unknown')}** ({p.get('country', '?')}) — "
                f"Role: {p.get('role', '?')} | "
                f"Style: {p.get('style', '?')} | "
                f"Era: {p.get('era', '?')} | "
                f"Achievements: {p.get('achievements', '?')}"
            )
            lines.append(line)
        context_block = "\n".join(lines)

    count_line = ""
    if exact_count is not None:
        count_line = f"\nExact count computed from the database: {exact_count}\n"

    is_list_or_count = _is_list_or_count_question(question)
    output_instruction = ""
    if is_list_or_count:
        output_instruction = (
            "\nOutput format requirement: answer the question directly and concisely. "
            "For list questions, output only the requested player names with no introductory or concluding commentary. "
            "For count questions, output a full sentence such as 'There are 0 matching players in this dataset.' "
            "or 'There is 1 matching player: Mushfiqur Rahim.'\n"
        )

    prompt = (
        f"Context — players retrieved from the database:\n"
        f"{context_block}"
        f"{count_line}"
        f"{output_instruction}"
        f"Question: {question}\n"
        f"Answer:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _format_count_answer(question: str, retrieved: list[dict], exact_count: int | None) -> str:
    count = len(retrieved) if exact_count is None else exact_count
    if count == 0:
        return "There are 0 matching players in this dataset."
    if count == 1:
        names = [p.get("name") for p in retrieved if p.get("name")]
        if names:
            return f"There is 1 matching player: {names[0]}"
        return "There is 1 matching player."
    return f"There are {count} matching players in this dataset."


def _extract_count_from_answer(answer: str) -> int | None:
    if not answer:
        return None
    match = re.search(r"\b(\d+)\b", answer)
    if not match:
        return None
    return int(match.group(1))


def _validate_count_answer(answer: str, exact_count: int | None, retrieved: list[dict], question: str) -> None:
    if exact_count is None:
        return
    expected = exact_count if exact_count is not None else len(retrieved)
    parsed = _extract_count_from_answer(answer)
    if parsed is None:
        raise AssertionError(f"Count-answer generation failed for {question!r}: no numeric count found in answer {answer!r}")
    if parsed != expected:
        raise AssertionError(
            f"Count-answer mismatch for {question!r}: expected {expected} but generated answer {answer!r} states {parsed}"
        )


def generate(
    question: str,
    retrieved: list[dict],
    exact_count: int | None = None,
) -> tuple[str, str]:
    """
    Generate an answer from the retrieved context.

    Returns (answer_text, full_prompt_used) — the prompt is returned for
    Langfuse tracing and for the "details" panel in the Streamlit UI.
    """
    system_prompt = build_system_prompt()
    user_prompt = build_prompt(question, retrieved, exact_count)
    full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

    if _is_list_or_count_question(question):
        if any(token in (question or "").lower() for token in ["how many", "count", "number of"]):
            answer = _format_count_answer(question, retrieved, exact_count)
            _validate_count_answer(answer, exact_count, retrieved, question)
        else:
            names = [p.get("name") for p in retrieved if p.get("name")]
            if names:
                answer = "\n".join(f"- {name}" for name in names)
            else:
                answer = "No matching players found in the database."
        return answer, full_prompt

    llm = _get_groq()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    response = llm.invoke(messages)
    answer = response.content.strip()
    return answer, full_prompt
