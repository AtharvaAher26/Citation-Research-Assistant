from citesage.processing.chunker import _split_sentences, chunk_paper

PAGES = [{"page": 1, "text": "Introduction\n" + "We study grounding of models. " * 60
          + "\nResults\n" + "The pipeline improved recall. " * 60}]


def test_chunks_within_size_bounds():
    chunks = chunk_paper("2401.00001v1", "Test paper", PAGES)
    assert len(chunks) > 1
    assert all(c.token_count <= 300 for c in chunks)  # target 220 + tolerance


def test_ids_deterministic_and_unique():
    a = chunk_paper("2401.00001v1", "Test paper", PAGES)
    b = chunk_paper("2401.00001v1", "Test paper", PAGES)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_sections_and_pages_tracked():
    chunks = chunk_paper("2401.00001v1", "Test paper", PAGES)
    assert any(c.section == "Introduction" for c in chunks)
    assert all(c.page_start == 1 for c in chunks)


def test_abbreviation_not_split():
    sentences = _split_sentences("Smith et al. 2020 showed this. It worked well.")
    assert any("et al. 2020" in s for s in sentences)

