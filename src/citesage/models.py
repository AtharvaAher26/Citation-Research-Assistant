from __future__ import annotations

from datetime import datetime
from hashlib import sha1

from pydantic import BaseModel, Field


def normalize_arxiv_id(raw: str) -> str:
    """'http://arxiv.org/abs/2401.12345v2' -> '2401.12345v2'"""
    return raw.rsplit("/abs/", 1)[-1]


class Paper(BaseModel):
    arxiv_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str = ""
    published: datetime | None = None
    categories: list[str] = Field(default_factory=list)
    pdf_path: str | None = None


class Chunk(BaseModel):
    chunk_id: str          # deterministic: paper_id + position + content hash
    paper_id: str
    paper_title: str
    position: int          # chunk index within the paper
    section: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    text: str
    token_count: int

    @staticmethod
    def make_id(paper_id: str, position: int, text: str) -> str:
        digest = sha1(text.encode("utf-8")).hexdigest()[:12]
        return f"{paper_id}::chunk_{position:03d}::{digest}"