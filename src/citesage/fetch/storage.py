from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from citesage.models import Paper

log = logging.getLogger(__name__)


class PaperStore:
    """JSON-file registry of fetched papers. All writes are atomic (tmp + rename),
    so a crash mid-write can never corrupt the manifest."""

    def __init__(self, path: Path):
        self.path = path
        self.papers: dict[str, Paper] = {}
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.papers = {pid: Paper.model_validate(p) for pid, p in raw.items()}

    def has(self, arxiv_id: str) -> bool:
        return arxiv_id in self.papers

    def upsert(self, paper: Paper) -> None:
        self.papers[paper.arxiv_id] = paper

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(
                    {pid: p.model_dump(mode="json") for pid, p in self.papers.items()},
                    f,
                    indent=2,
                )
            os.replace(tmp, self.path)  # atomic on POSIX and Windows
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

