from __future__ import annotations

import logging
import math
import re
from collections import Counter
from typing import Any

from citesage.config import settings
from citesage.models import Chunk

log = logging.getLogger(__name__)

# Token heuristic (~1.35 words/token). Deliberately model-agnostic so the pipeline
# runs offline; revisit against the embedder's real tokenizer in Phase 1B evals.
_WORDS_PER_TOKEN = 1.35

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])")
_ABBREV = re.compile(
    r"(?:et al|e\.g|i\.e|vs|cf|Fig|fig|Eq|eq|Sec|sec|Tab|approx|no|Dr|Prof|Mr|Ms|Jr|etc)\.$"
)
_SECTION_HEADING = re.compile(
    r"^\s*(?:\d+\.?\d*\.?\s+)?(abstract|introduction|background|related work|"
    r"method(?:s|ology)?|experiment(?:s|al setup)?|results|discussion|conclusion|"
    r"references|acknowledg\w*|appendix)\b.*$",
    re.IGNORECASE,
)


def _split_sentences(paragraph: str) -> list[str]:
    parts = _SENT_SPLIT.split(paragraph)
    merged: list[str] = []
    for part in parts:
        # re-join fragments wrongly split after abbreviations ("et al." / "Fig.")
        if merged and _ABBREV.search(merged[-1]):
            merged[-1] += " " + part
        else:
            merged.append(part)
    return [s.strip() for s in merged if s.strip()]


def _sentence_stream(pages: list[dict[str, Any]]):
    """Yields (page, section, sentence) — heading lines set the current section."""
    for page in pages:
        current_section = None
        for line in page["text"].split("\n"):
            line = line.strip()
            if not line:
                continue
            heading = _SECTION_HEADING.match(line)
            if heading and len(line) < 70:
                current_section = heading.group(1).title()
                continue
            for sent in _split_sentences(line):
                yield page["page"], current_section, sent


def chunk_paper(arxiv_id: str, title: str, pages: list[dict[str, Any]]) -> list[Chunk]:
    """Sentence-aware packing with overlap. Oversized single sentences are hard-split."""

    def est(s: str) -> int:
        return math.ceil(len(s.split()) * _WORDS_PER_TOKEN)

    chunks: list[Chunk] = []
    buf: list[tuple[int, str | None, str]] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        text = " ".join(s for _, _, s in buf)
        pages_in = [p for p, _, _ in buf]
        sections = [sec for _, sec, _ in buf if sec]
        position = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=Chunk.make_id(arxiv_id, position, text),
                paper_id=arxiv_id,
                paper_title=title,
                position=position,
                section=Counter(sections).most_common(1)[0][0] if sections else None,
                page_start=min(pages_in),
                page_end=max(pages_in),
                text=text,
                token_count=est(text),
            )
        )
        # carry the tail into the next chunk (overlap)
        tail: list[tuple[int, str | None, str]] = []
        carried = 0
        for item in reversed(buf):
            if carried + est(item[2]) > settings.chunk_overlap:
                break
            tail.insert(0, item)
            carried += est(item[2])
        buf = tail

    stream = list(_sentence_stream(pages))
    max_words = int(settings.chunk_tokens / _WORDS_PER_TOKEN)

    i = 0
    while i < len(stream):
        page, section, sent = stream[i]
        tokens = est(sent)

        if tokens > settings.chunk_tokens:
            # pathological single sentence -> hard-split by word budget
            flush()
            buf = []
            words = sent.split()
            for j in range(0, len(words), max_words):
                piece = " ".join(words[j: j + max_words])
                buf.append((page, section, piece))
                flush()
            i += 1
            continue

        if sum(est(s) for _, _, s in buf) + tokens > settings.chunk_tokens:
            flush()
        buf.append((page, section, sent))
        i += 1
    flush()

    log.info("chunked %s: %d chunks", arxiv_id, len(chunks))
    return chunks

