from dataclasses import dataclass

from citesage.generation.answerer import ask, build_user_prompt


@dataclass
class FakeChunk:
    chunk_id: str = "c1"
    paper_id: str = "9999.00001v1"
    title: str = "Fixture Paper"
    section: str = "Methods"
    page_start: int = 2
    text: str = "Grounding matters for trustworthy answers."


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.name = "fake/model"

    def generate(self, system, user):
        return self.text


def test_build_user_prompt_numbers_blocks_with_provenance():
    prompt = build_user_prompt("q?", [FakeChunk(), FakeChunk(chunk_id="c2", title="P2")])
    assert prompt.startswith("Question: q?")
    assert "[1] (Fixture Paper, arXiv:9999.00001v1, p.2, Methods)" in prompt
    assert "[2] (P2, arXiv:9999.00001v1, p.2, Methods)" in prompt
    assert prompt.endswith("Answer:")


def test_ask_returns_grounded_answer_with_resolved_citations():
    evidence = [FakeChunk()]
    answer = ask("q?", evidence, [FakeClient("Trust needs grounding [1].")])
    assert answer.refused is False
    assert answer.text == "Trust needs grounding [1]."
    assert answer.citations[0].paper_title == "Fixture Paper"
    assert answer.provider == "fake/model"


def test_ask_reports_refusal():
    answer = ask("q?", [FakeChunk()], [FakeClient("NOT_ENOUGH_EVIDENCE\nnothing about quantum fields")])
    assert answer.refused is True
    assert answer.citations == []
    assert "quantum" in answer.text


def test_ask_strips_out_of_range_markers():
    answer = ask("q?", [FakeChunk()], [FakeClient("claim [1] and [9]")])
    assert "[9]" not in answer.text
    assert answer.citations[0].number == 1
    assert answer.warning and "invalid citation" in answer.warning


def test_ask_annotates_uncited_answers():
    answer = ask("q?", [FakeChunk()], [FakeClient("confident answer without markers")])
    assert "unverified" in answer.text
    assert answer.citations == []