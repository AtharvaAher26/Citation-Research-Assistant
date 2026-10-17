"""One-shot indexer: chunks.jsonl -> dense (Chroma) + sparse (BM25 JSON).

Reads chunk records defensively: ``chunk_id`` and ``text`` are required
(missing lines are skipped with a warning), every citation field is
best-effort so schema additions upstream never break indexing.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from citesage.retrieval.embedder import Embedder
from citesage.retrieval.keyword_index import BM25Document, BM25Index

logger = logging.getLogger(__name__)


@dataclass
class IndexRecord:
    chunk_id: str
    paper_id: str
    text: str
    metadata: dict[str, Any]


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_chunk_records(chunks_path: Path) -> list[IndexRecord]:
    records: list[IndexRecord] = []
    skipped = 0
    with Path(chunks_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{chunks_path} line {line_number}: invalid JSON ({exc})") from exc
            chunk_id = raw.get("chunk_id")
            text = raw.get("text")
            if not chunk_id or not text:
                skipped += 1
                logger.warning("%s line %d: missing chunk_id/text - skipped", chunks_path, line_number)
                continue
            paper_id = str(raw.get("paper_id") or "unknown")
            metadata = {
                "paper_id": paper_id,
                "title": str(raw.get("title") or paper_id),
                "section": str(raw.get("section") or "unknown"),
                "page_start": _safe_int(raw.get("page_start"), 0),
                "page_end": _safe_int(raw.get("page_end"), 0),
            }
            records.append(
                IndexRecord(chunk_id=str(chunk_id), paper_id=paper_id, text=str(text), metadata=metadata)
            )
    if skipped:
        logger.warning("Skipped %d malformed chunk line(s)", skipped)
    if not records:
        raise ValueError(f"No usable chunks in {chunks_path} - run 'citesage chunk' first")
    return records


def build_indexes(
    chunks_path: Path,
    index_dir: Path,
    embedder: Embedder | None = None,
    rebuild: bool = True,
) -> dict[str, int]:
    """Build both indexes from chunks.jsonl. Deterministic and idempotent."""
    from citesage.retrieval.vector_store import VectorStore  # deferred heavy import (chromadb)

    records = load_chunk_records(Path(chunks_path))
    embedder = embedder or Embedder()
    logger.info("Encoding %d chunks with '%s'...", len(records), embedder.model_name)
    vectors = embedder.encode([record.text for record in records])

    vector_store = VectorStore(index_dir)
    if rebuild:
        vector_store.reset()
    vector_store.upsert(
        ids=[record.chunk_id for record in records],
        embeddings=vectors,
        documents=[record.text for record in records],
        metadatas=[record.metadata for record in records],
    )

    bm25_index = BM25Index()
    bm25_index.build(
        [BM25Document(chunk_id=r.chunk_id, text=r.text, metadata=r.metadata) for r in records]
    )
    bm25_index.save(Path(index_dir) / "bm25.json")

    logger.info("Index build complete: %d dense, %d sparse", vector_store.count(), bm25_index.size)
    return {
        "chunks": len(records),
        "dense_count": vector_store.count(),
        "sparse_count": bm25_index.size,
        "dim": int(vectors.shape[1]),
    }