from __future__ import annotations

import json
import logging
from pathlib import Path

import pymupdf          # was: import fitz  # PyMuPDF
from citesage.config import settings
from citesage.models import Paper

log = logging.getLogger(__name__)


class ExtractionError(Exception):
    """PDF text extraction produced garbage (e.g. scanned paper)."""


def extract_pdf(paper: Paper, out_dir: Path) -> Path:
    """PDF -> interim JSON with per-page text (page numbers kept for citations).

    Idempotent: skips if output exists. Raises ExtractionError if the paper looks
    scanned/unreadable — this is the validation gate that protects index quality.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{paper.arxiv_id.replace('/', '_')}.json"
    if out_path.exists():
        return out_path

    doc = pymupdf.open(paper.pdf_path)   # was: fitz.open(...)
    try:
        pages = []
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if len(text) < settings.min_page_chars:
                text = ""  # near-empty page (figure-only) — drop it
            pages.append({"page": i, "text": text})
    finally:
        doc.close()

    n_chars = sum(len(p["text"]) for p in pages)
    if n_chars < settings.min_paper_chars:
        raise ExtractionError(
            f"only {n_chars} chars extracted — likely a scanned PDF; rejecting"
        )

    payload = {
        "arxiv_id": paper.arxiv_id,
        "title": paper.title,
        "published": paper.published.isoformat() if paper.published else None,
        "n_chars": n_chars,
        "pages": pages,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")
    log.info("extracted %s: %d pages, %d chars", paper.arxiv_id, len(pages), n_chars)
    return out_path

