"""Retrieval layer: dense (Chroma) + sparse (BM25) + hybrid RRF fusion.

This package intentionally has no import-time side effects: heavy libraries
(chromadb, sentence-transformers) are imported lazily by the modules that
need them, so the CLI and test-suite stay fast.
"""