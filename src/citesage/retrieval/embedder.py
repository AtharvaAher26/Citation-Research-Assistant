"""Dense sentence embeddings with lazy model loading.

sentence-transformers (and its PyTorch dependency) is imported only when we
actually encode text, so importing this module - and running the offline
test-suite - stays fast and dependency-light.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384


class Embedder:
    """Encode text into unit-normalised vectors (cosine-ready).

    The model loads lazily on the first ``encode`` call: constructing an
    ``Embedder`` is free, so CLI commands that don't need embeddings stay fast.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: "SentenceTransformer | None" = None

    @property
    def dimension(self) -> int:
        if self._model is None:
            return EXPECTED_DIMENSION
        return int(self._model.get_sentence_embedding_dimension())

    def _ensure_model(self) -> "SentenceTransformer":
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # deferred heavy import

            logger.info("Loading embedding model '%s' (first run downloads ~90 MB)...", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 matrix of shape (len(texts), dimension), L2-normalised."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # unit norm -> cosine similarity == dot product
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)