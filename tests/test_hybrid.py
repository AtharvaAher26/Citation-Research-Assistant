import numpy as np
import pytest

from citesage.retrieval.hybrid import HybridRetriever


class FakeEmbedder:
    def encode(self, texts):
        return np.zeros((len(texts), 4), dtype=np.float32)


class FakeVectorStore:
    def __init__(self, hits):
        self._hits = hits

    def query(self, embedding, top_k):
        return self._hits[:top_k]


class FakeBM25:
    def __init__(self, hits):
        self._hits = hits

    def search(self, query, top_k):
        return self._hits[:top_k]


def _hit(chunk_id):
    return {
        "chunk_id": chunk_id,
        "text": f"text of {chunk_id}",
        "metadata": {
            "paper_id": "9999.00001v1",
            "title": "Fixture Paper",
            "section": "Methods",
            "page_start": 1,
            "page_end": 1,
        },
        "score": 0.0,
    }


def test_rrf_prefers_docs_present_in_both_lists():
    dense = [_hit("A"), _hit("B"), _hit("C")]
    sparse = [_hit("B"), _hit("D"), _hit("A")]
    retriever = HybridRetriever(
        vector_store=FakeVectorStore(dense), bm25_index=FakeBM25(sparse), embedder=FakeEmbedder()
    )
    results = retriever.search("anything", top_k=4, candidate_k=3)
    # RRF(k=60): B=1/62+1/61 > A=1/61+1/63 > D=1/62 > C=1/63
    assert [r.chunk_id for r in results] == ["B", "A", "D", "C"]


def test_dense_mode_uses_cosine_scores_and_sets_dense_rank():
    hits = [_hit("A"), _hit("B")]
    hits[0]["score"] = 0.91
    hits[1]["score"] = 0.72
    retriever = HybridRetriever(
        vector_store=FakeVectorStore(hits), bm25_index=FakeBM25([]), embedder=FakeEmbedder()
    )
    out = retriever.search("q", top_k=5, mode="dense")
    assert [r.chunk_id for r in out] == ["A", "B"]
    assert out[0].score == pytest.approx(0.91)
    assert out[0].dense_rank == 1 and out[0].sparse_rank is None


def test_citation_string_includes_title_id_and_page():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore([_hit("A")]), bm25_index=FakeBM25([]), embedder=FakeEmbedder()
    )
    result = retriever.search("q", top_k=1, mode="dense")[0]
    assert result.citation == "Fixture Paper (arXiv:9999.00001v1), p. 1 [Methods]"


def test_invalid_mode_raises():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore([]), bm25_index=FakeBM25([]), embedder=FakeEmbedder()
    )
    with pytest.raises(ValueError):
        retriever.search("q", mode="bogus")


def test_empty_indexes_return_no_results():
    retriever = HybridRetriever(
        vector_store=FakeVectorStore([]), bm25_index=FakeBM25([]), embedder=FakeEmbedder()
    )
    assert retriever.search("anything") == []