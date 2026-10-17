"""Citation contract: parse, validate, and resolve [N] markers against evidence.

The rule that makes this product trustworthy: every claim in an answer must
point at a numbered evidence block, and every number must resolve to a real
retrieved chunk with paper + page + section provenance.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

_CITATION_RE = re.compile(r"\[(\d{1,2})\]")

REFUSAL_TOKEN = "NOT_ENOUGH_EVIDENCE"


@dataclass
class Citation:
    number: int
    paper_title: str
    arxiv_id: str
    page_start: int | None
    section: str
    chunk_id: str

    @property
    def reference(self) -> str:
        page = f", p. {self.page_start}" if self.page_start else ""
        return f"[{self.number}] {self.paper_title} (arXiv:{self.arxiv_id}){page} [{self.section}]"


@dataclass
class Answer:
    question: str
    text: str
    citations: list[Citation]
    refused: bool
    provider: str
    warning: str | None = None


def parse_citation_numbers(text: str) -> list[int]:
    """All citation numbers in order of appearance (duplicates included)."""
    return [int(match) for match in _CITATION_RE.findall(text)]


def has_citations(text: str) -> bool:
    return bool(_CITATION_RE.search(text))


def strip_invalid_citations(text: str, evidence_count: int) -> tuple[str, list[int]]:
    """Remove markers pointing outside 1..evidence_count. Returns (clean_text, removed)."""
    removed: list[int] = []

    def _replace(match: re.Match[str]) -> str:
        number = int(match.group(1))
        if 1 <= number <= evidence_count:
            return match.group(0)
        removed.append(number)
        return ""

    cleaned = _CITATION_RE.sub(_replace, text)
    return cleaned, removed


def looks_like_refusal(text: str) -> bool:
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return REFUSAL_TOKEN in first_line.upper()


def resolve_citations(text: str, evidence: Sequence[Any]) -> list[Citation]:
    """Map citation markers to real chunks, deduplicated and sorted by number."""
    seen: dict[int, Citation] = {}
    for number in parse_citation_numbers(text):
        if number in seen or not (1 <= number <= len(evidence)):
            continue
        chunk = evidence[number - 1]
        seen[number] = Citation(
            number=number,
            paper_title=str(getattr(chunk, "title", "") or getattr(chunk, "paper_id", "unknown")),
            arxiv_id=str(getattr(chunk, "paper_id", "unknown")),
            page_start=getattr(chunk, "page_start", None),
            section=str(getattr(chunk, "section", "unknown")),
            chunk_id=str(getattr(chunk, "chunk_id", "")),
        )
    return sorted(seen.values(), key=lambda citation: citation.number)