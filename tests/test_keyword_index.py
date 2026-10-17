from citesage.retrieval.keyword_index import BM25Document, BM25Index, tokenize

DOCS = [
    ("c1", "Retrieval augmented generation grounds answers in retrieved evidence from a corpus."),
    ("c2", "The cat sat on the mat and ignored the corpus entirely."),
    ("c3", "Evaluating retrieval quality with recall and precision at k matters for RAG pipelines."),
]


def _index() -> BM25Index:
    index = BM25Index()
    meta = {"paper_id": "p1", "title": "T", "section": "s", "page_start": 1, "page_end": 1}
    index.build([BM25Document(cid, text, dict(meta)) for cid, text in DOCS])
    return index


def test_tokenize_lowercases_and_removes_stopwords():
    tokens = tokenize("The Cat and the RAG!")
    assert "the" not in tokens and "and" not in tokens
    assert "cat" in tokens and "rag" in tokens


def test_bm25_ranks_relevant_doc_first():
    hits = _index().search("retrieval augmented generation", top_k=2)
    assert hits[0]["chunk_id"] == "c1"
    assert hits[0]["metadata"]["paper_id"] == "p1"


def test_empty_or_stopword_query_returns_nothing():
    index = _index()
    assert index.search("", top_k=3) == []
    assert index.search("the and of", top_k=3) == []


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "bm25.json"
    _index().save(path)
    loaded = BM25Index.load(path)
    assert loaded.search("retrieval augmented generation", top_k=1)[0]["chunk_id"] == "c1"