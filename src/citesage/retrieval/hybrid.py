"""Hybrid retrieval: dense (semantic) + sparse (BM25), fused by Reciprocal Rank Fusion.

RRF is score-free: it only uses ranks, so we never need to calibrate a BM25
score against a cosine similarity. It is the industry-standard first choice
for hybrid search (Elasticsearch, Azure AI Search, Weaviate all ship it).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, Sequence

import numpy as np

from citesage.retrieval.keyword_index import BM25Index

if TYPE_CHECKING:  # pragma: no cover
    from citesage.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

RRF_K = 60  # standard RRF constant (from the original Cormack et al. paper)

VALID_MODES = ("hybrid", "dense", "sparse")


class EmbedsText(Protocol):
    """Anything with Embedder's .encode() - lets tests plug in fake embedders."""

    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


@dataclass
class RetrievedChunk:
    """A retrieved chunk with everything needed to build a citation later."""

    chunk_id: str
    text: str
    score: float
    paper_id: str
    title: str
    section: str
    page_start: int | None = None
    page_end: int | None = None
    dense_rank: int | None = None
    sparse_rank: int | None = None

    @property
    def citation(self) -> str:
        location = f", p. {self.page_start}" if self.page_start else ""
        return f"{self.title} (arXiv:{self.paper_id}){location} [{self.section}]"


def _reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked_ids in rank_lists:
        for rank, chunk_id in enumerate(ranked_ids, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    """Duck-typed collaborators make this unit-testable without heavy deps:

    - vector_store: anything with .query(embedding, top_k) -> list[dict]  (VectorStore)
    - bm25_index:   anything with .search(query, top_k) -> list[dict]     (BM25Index)
    - embedder:     anything with .encode(texts) -> np.ndarray            (Embedder)
    """

    def __init__(
        self,
        vector_store: "VectorStore",
        bm25_index: BM25Index,
        embedder: EmbedsText,
    ) -> None:
        self._vector_store = vector_store
        self._bm25_index = bm25_index
        self._embedder = embedder

    def search(
        self,
        query: str,
        top_k: int = 8,
        candidate_k: int = 30,
        mode: str = "hybrid",
    ) -> list[RetrievedChunk]:
        """Retrieve chunks in ``mode``: hybrid (RRF), dense-only, or sparse-only.

        Score semantics: hybrid -> RRF score; dense -> cosine similarity;
        sparse -> BM25 score. Deterministic tie-breaking by chunk_id.
        """
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
        if top_k <= 0:
            return []

        dense_hits: list[dict[str, Any]] = []
        sparse_hits: list[dict[str, Any]] = []
        if mode in ("hybrid", "dense"):
            query_vector = self._embedder.encode([query])[0]
            dense_hits = self._vector_store.query(query_vector, top_k=candidate_k)
        if mode in ("hybrid", "sparse"):
            sparse_hits = self._bm25_index.search(query, top_k=candidate_k)

        dense_ranks = {hit["chunk_id"]: rank for rank, hit in enumerate(dense_hits, start=1)}
        sparse_ranks = {hit["chunk_id"]: rank for rank, hit in enumerate(sparse_hits, start=1)}

        rank_lists: list[list[str]] = []
        if mode in ("hybrid", "dense"):
            rank_lists.append(list(dense_ranks))
        if mode in ("hybrid", "sparse"):
            rank_lists.append(list(sparse_ranks))
        fused = _reciprocal_rank_fusion(rank_lists)

        hits_by_id: dict[str, dict[str, Any]] = {}
        for hit in dense_hits + sparse_hits:
            hits_by_id.setdefault(hit["chunk_id"], hit)

        results: list[RetrievedChunk] = []
        for chunk_id, fused_score in sorted(fused.items(), key=lambda item: (-item[1], item[0])):
            if len(results) >= top_k:
                break
            hit = hits_by_id[chunk_id]
            raw_score = fused_score if mode == "hybrid" else float(hit.get("score") or 0.0)
            results.append(
                self._to_retrieved(hit, raw_score, dense_ranks.get(chunk_id), sparse_ranks.get(chunk_id))
            )
        logger.info(
            "Hybrid search (%s): %d dense + %d sparse -> %d fused results",
            mode, len(dense_hits), len(sparse_hits), len(results),
        )
        return results

    @staticmethod
    def _to_retrieved(
        hit: dict[str, Any],
        score: float,
        dense_rank: int | None,
        sparse_rank: int | None,
    ) -> RetrievedChunk:
        metadata = hit.get("metadata") or {}
        paper_id = str(metadata.get("paper_id") or "unknown")
        return RetrievedChunk(
            chunk_id=str(hit["chunk_id"]),
            text=str(hit.get("text") or ""),
            score=float(score),
            paper_id=paper_id,
            title=str(metadata.get("title") or paper_id),
            section=str(metadata.get("section") or "unknown"),
            page_start=_optional_int(metadata.get("page_start")) or None,
            page_end=_optional_int(metadata.get("page_end")) or None,
            dense_rank=dense_rank,
            sparse_rank=sparse_rank,
        )