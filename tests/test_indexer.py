import json

import pytest

from citesage.retrieval.indexer import load_chunk_records


def _write(tmp_path, lines):
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return path


def test_loads_valid_records_with_defaults(tmp_path):
    path = _write(
        tmp_path,
        [
            {"chunk_id": "a", "paper_id": "p1", "text": "hello world", "section": "Intro",
             "page_start": 2, "page_end": 2, "title": "My Paper"},
            {"chunk_id": "b", "paper_id": "p2", "text": "second chunk"},
        ],
    )
    records = load_chunk_records(path)
    assert [r.chunk_id for r in records] == ["a", "b"]
    assert records[0].metadata["title"] == "My Paper"
    assert records[1].metadata["title"] == "p2"    # falls back to paper_id
    assert records[1].metadata["page_start"] == 0  # safe default


def test_skips_lines_missing_required_fields(tmp_path):
    path = _write(
        tmp_path,
        [
            {"chunk_id": "a", "text": "no paper_id is ok"},
            {"paper_id": "p", "text": "missing chunk_id"},
            {"chunk_id": "c", "paper_id": "p3", "text": "fine"},
        ],
    )
    records = load_chunk_records(path)
    assert [r.chunk_id for r in records] == ["a", "c"]


def test_invalid_json_raises_with_line_number(tmp_path):
    bad = tmp_path / "chunks.jsonl"
    bad.write_text('{"chunk_id": "a"}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2"):
        load_chunk_records(bad)


def test_empty_file_raises(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="No usable chunks"):
        load_chunk_records(path)