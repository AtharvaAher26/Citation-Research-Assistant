from dataclasses import dataclass

from citesage.generation.citations import (
    has_citations,
    looks_like_refusal,
    parse_citation_numbers,
    resolve_citations,
    strip_invalid_citations,
)


@dataclass
class FakeChunk:
    chunk_id: str = "c1"
    paper_id: str = "9999.00001v1"
    title: str = "Fixture Paper"
    section: str = "Methods"
    page_start: int = 2


def test_parse_citation_numbers_in_order():
    assert parse_citation_numbers("A [1] and [3] plus [2]") == [1, 3, 2]


def test_has_citations():
    assert has_citations("grounded claim [1]")
    assert not has_citations("no markers here")


def test_strip_invalid_citations_keeps_valid_only():
    cleaned, removed = strip_invalid_citations("x [1] y [5] z [99]", evidence_count=3)
    assert cleaned == "x [1] y  z "
    assert removed == [5, 99]


def test_refusal_detection():
    assert looks_like_refusal("NOT_ENOUGH_EVIDENCE\nthe papers do not cover X")
    assert not looks_like_refusal("Grounded answer [1]")


def test_resolve_citations_maps_numbers_to_chunks():
    evidence = [
        FakeChunk(),
        FakeChunk(chunk_id="c2", title="Second Paper", paper_id="9999.00002v1", page_start=3),
    ]
    text = "claim one [2] then claim two [1] and again [2]"
    citations = resolve_citations(text, evidence)
    assert [c.number for c in citations] == [1, 2]  # deduped and sorted
    assert citations[0].paper_title == "Fixture Paper"
    assert citations[1].arxiv_id == "9999.00002v1"
    assert citations[1].reference == "[2] Second Paper (arXiv:9999.00002v1), p. 3 [Methods]"


def test_resolve_ignores_out_of_range():
    assert resolve_citations("claim [7]", evidence=[FakeChunk()]) == []