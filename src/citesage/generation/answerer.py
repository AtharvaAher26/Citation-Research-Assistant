"""Grounded answering: evidence blocks -> LLM -> validated, citation-resolved Answer."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from citesage.generation.citations import (
    Answer,
    has_citations,
    looks_like_refusal,
    resolve_citations,
    strip_invalid_citations,
)
from citesage.generation.llm import LLMClient, LLMError, generate_with_fallback

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are CitationSage, a research assistant that answers strictly from the numbered evidence blocks provided by the user.

Rules:
1. Use ONLY the evidence blocks. Never use outside knowledge, even if you are confident.
2. Every factual claim must end with a citation marker for the block(s) that support it, like [1] or [2][4]. Place each marker immediately after the claim it supports.
3. Only cite block numbers that actually exist in the evidence. Never invent sources.
4. If the evidence is insufficient, unrelated, or contradictory, reply with NOT_ENOUGH_EVIDENCE on the first line, then one short sentence explaining what is missing.
5. Write 1-3 short paragraphs. No headers, no bullet lists, no bibliography at the end - inline markers only.
"""


def build_user_prompt(question: str, evidence: Sequence[Any]) -> str:
    if not evidence:
        raise ValueError("cannot build a prompt with empty evidence")
    blocks: list[str] = []
    for position, chunk in enumerate(evidence, start=1):
        title = getattr(chunk, "title", None) or getattr(chunk, "paper_id", "unknown")
        arxiv_id = getattr(chunk, "paper_id", "unknown")
        page = getattr(chunk, "page_start", None)
        section = getattr(chunk, "section", None) or "unknown"
        provenance = f"[{position}] ({title}, arXiv:{arxiv_id}, p.{page}, {section})"
        text = " ".join(str(getattr(chunk, "text", "")).split())
        blocks.append(f"{provenance}\n{text}")
    joined = "\n\n".join(blocks)
    return f"Question: {question}\n\nEvidence:\n{joined}\n\nAnswer:"


def ask(question: str, evidence: Sequence[Any], clients: Sequence[LLMClient]) -> Answer:
    """Generate a grounded answer. Refusals are reported in the Answer, never raised."""
    user_prompt = build_user_prompt(question, evidence)
    text, provider = generate_with_fallback(clients, SYSTEM_PROMPT, user_prompt)

    if looks_like_refusal(text):
        lines = text.strip().splitlines()[1:]
        explanation = " ".join(line.strip() for line in lines).strip()
        return Answer(
            question=question,
            text=explanation or "The indexed papers do not cover this question.",
            citations=[],
            refused=True,
            provider=provider,
        )

    # one retry if the model answered substantively but cited nothing
    if not has_citations(text):
        logger.warning("Answer had no citations - retrying once with a stricter instruction")
        stricter = (
            user_prompt
            + "\n\nIMPORTANT: cite at least one evidence block number like [1] for every claim, "
            "or reply NOT_ENOUGH_EVIDENCE."
        )
        text, provider = generate_with_fallback(clients, SYSTEM_PROMPT, stricter)

    warning: str | None = None
    if not has_citations(text):
        warning = "model returned no citations - answer is unverified"
        text = text + "\n\n[unverified: no citations were returned]"

    cleaned, removed = strip_invalid_citations(text, len(evidence))
    if removed:
        logger.warning("Removed out-of-range citation markers: %s", removed)
        warning = (warning + "; " if warning else "") + f"removed invalid citation markers: {removed}"

    cleaned = cleaned.strip() or text.strip()
    return Answer(
        question=question,
        text=cleaned,
        citations=resolve_citations(cleaned, evidence),
        refused=False,
        provider=provider,
        warning=warning,
    )