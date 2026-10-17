"""Sparse keyword retrieval (BM25) with a dependency-free tokenizer.

We persist only the corpus (ids + texts + metadata) as JSON and rebuild the
BM25 structure in memory on load - deterministic, version-proof, and cheap
for corpora from thousands up to hundreds of thousands of chunks.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """
    a an and are as at be been by for from has have in into is it its of on or
    that the to was were will with this these those we our you your they their
    et al fig figure table section eq equation
    """.split()
)


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, minus stopwords and single characters."""
    return [
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in _STOPWORDS
    ]


@dataclass
class BM25Document:
    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BM25Index:
    def __init__(self) -> None:
        self._doc_ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._engine: BM25Okapi | None = None

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    def build(self, documents: list[BM25Document]) -> None:
        self._doc_ids = [doc.chunk_id for doc in documents]
        self._texts = [doc.text for doc in documents]
        self._metadatas = [dict(doc.metadata) for doc in documents]
        corpus = [tokenize(text) for text in self._texts]
        if self._doc_ids and any(corpus):
            self._engine = BM25Okapi(corpus)
        else:
            self._engine = None
            logger.warning("BM25 index is empty (no usable tokens in %d documents)", len(documents))

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._engine is None or top_k <= 0:
            return []
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._engine.get_scores(query_tokens)
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], self._doc_ids[i]))
        hits: list[dict[str, Any]] = []
        for i in order[:top_k]:
            if scores[i] <= 0:
                break
            hits.append(
                {
                    "chunk_id": self._doc_ids[i],
                    "text": self._texts[i],
                    "metadata": dict(self._metadatas[i]),
                    "score": float(scores[i]),
                }
            )
        return hits

    def save(self, path: str | Path) -> None:
        """Atomic JSON write (tmp file + os.replace), same pattern as the stores."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "doc_ids": self._doc_ids,
            "texts": self._texts,
            "metadatas": self._metadatas,
        }
        handle, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.remove(tmp_name)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | Path) -> "BM25Index":
        source = Path(path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        index = cls()
        index.build(
            [
                BM25Document(chunk_id=str(cid), text=str(text), metadata=dict(meta or {}))
                for cid, text, meta in zip(payload["doc_ids"], payload["texts"], payload["metadatas"])
            ]
        )
        return index