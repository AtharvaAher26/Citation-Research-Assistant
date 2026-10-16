from pathlib import Path

from citesage.fetch.storage import PaperStore
from citesage.models import Paper


def make_paper(pid="2401.00001v1"):
    return Paper(arxiv_id=pid, title="T", authors=["A"], abstract="x",
                 categories=["cs.LG"])


def test_roundtrip_and_atomicity(tmp_path: Path):
    path = tmp_path / "papers.json"
    store = PaperStore(path)
    store.upsert(make_paper())
    store.save()
    assert not list(tmp_path.glob("*.tmp"))          # no temp files left behind

    reloaded = PaperStore(path)
    assert reloaded.has("2401.00001v1")
    assert reloaded.papers["2401.00001v1"].title == "T"

