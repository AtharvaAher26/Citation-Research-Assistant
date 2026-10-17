"""Persistent dense vector store backed by Chroma.

- cosine distance space (matches our L2-normalised MiniLM vectors)
- one row per chunk: document = chunk text, metadata = citation fields
- rebuild semantics: reset() then upsert() keeps indexing deterministic
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

logger = logging.getLogger(__name__)

_UPSERT_BATCH = 256


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool metadata values - coerce or drop the rest."""
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class VectorStore:
    def __init__(self, path: str | Path, collection_name: str = "citesage_chunks") -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return int(self._collection.count())

    def reset(self) -> None:
        """Drop and recreate the collection (used for deterministic rebuilds)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ) -> int:
        if len(ids) == 0:
            return 0
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            raise ValueError("ids/embeddings/documents/metadatas length mismatch")
        clean_ids = [str(i) for i in ids]
        clean_meta = [_clean_metadata(m) for m in metadatas]
        for start in range(0, len(clean_ids), _UPSERT_BATCH):
            end = start + _UPSERT_BATCH
            self._collection.upsert(
                ids=clean_ids[start:end],
                embeddings=[[float(x) for x in vec] for vec in embeddings[start:end]],
                documents=list(documents[start:end]),
                metadatas=clean_meta[start:end],
            )
        logger.info("Upserted %d vectors into '%s'", len(clean_ids), self.collection_name)
        return len(clean_ids)

    def query(self, embedding: Sequence[float], top_k: int) -> list[dict[str, Any]]:
        """Return the top_k chunks closest to ``embedding`` (cosine similarity)."""
        total = self.count()
        if total == 0 or top_k <= 0:
            return []
        result = self._collection.query(
            query_embeddings=[[float(x) for x in embedding]],
            n_results=min(top_k, total),
            include=["documents", "metadatas", "distances"],
        )
        ids = result["ids"][0]
        docs = result["documents"][0] or [""] * len(ids)
        metas = result["metadatas"][0] or [{}] * len(ids)
        dists = result["distances"][0] or [1.0] * len(ids)
        hits: list[dict[str, Any]] = []
        for position, chunk_id in enumerate(ids):
            distance = float(dists[position])
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": docs[position],
                    "metadata": dict(metas[position] or {}),
                    "distance": distance,
                    "score": 1.0 - distance,  # cosine similarity
                }
            )
        return hits