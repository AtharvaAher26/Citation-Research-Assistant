from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import typer
from rich.console import Console

from citesage.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, settings
from citesage.fetch.arxiv_client import ArxivClient, ArxivFetchError, ArxivRateLimitError
from citesage.fetch.storage import PaperStore
from citesage.logging_conf import setup_logging
from citesage.processing.chunker import chunk_paper
from citesage.processing.extractor import ExtractionError, extract_pdf

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="CiteSage — citation-grounded research assistant",
)
console = Console()
log = logging.getLogger(__name__)


@app.command()
def fetch(topic: str, max_papers: int = typer.Option(10, min=1, max=50)):
    """Search arXiv and download PDFs into data/raw (idempotent)."""
    setup_logging(settings.log_level)
    store = PaperStore(RAW_DIR / "papers.json")
    client = ArxivClient()

    try:
        papers = client.search(topic, max_papers)
    except ArxivRateLimitError:
        console.print("[red]arXiv is rate-limiting this IP (HTTP 429).[/]")
        console.print("This is temporary — wait 60–90 seconds, then run the same command again.")
        raise typer.Exit(1)
    except ArxivFetchError as exc:
        console.print(f"[red]Search failed:[/] {exc}")
        raise typer.Exit(1)

    new = 0
    for i, paper in enumerate(papers, start=1):
        if store.has(paper.arxiv_id):
            console.print(f"[dim]{i}. skip {paper.arxiv_id} (already fetched)[/]")
            continue
        console.print(f"{i}. downloading [cyan]{paper.arxiv_id}[/] — {paper.title[:60]}…")
        try:
            pdf_path = client.download_pdf(paper, RAW_DIR)
        except Exception as exc:  # keep going: one bad paper must not kill the batch
            log.warning("download failed %s: %s", paper.arxiv_id, exc)
            console.print(f"   [yellow]failed: {exc}[/]")
            continue
        paper.pdf_path = str(pdf_path)
        store.upsert(paper)
        new += 1
        time.sleep(1.0)  # politeness between downloads

    store.save()
    console.print(f"[green]✔ fetched {new} new papers[/] (store total: {len(store.papers)})")


@app.command()
def extract():
    """Extract text from all downloaded PDFs -> data/interim/*.json."""
    setup_logging(settings.log_level)
    store = PaperStore(RAW_DIR / "papers.json")
    ok = skipped = 0
    for pid, paper in store.papers.items():
        if not paper.pdf_path or not Path(paper.pdf_path).exists():
            log.warning("missing pdf for %s", pid)
            continue
        try:
            extract_pdf(paper, INTERIM_DIR)
            ok += 1
        except ExtractionError as exc:
            skipped += 1
            console.print(f"[yellow]rejected {pid}: {exc}[/]")
    console.print(f"[green]✔ extracted {ok}[/], rejected {skipped}")


@app.command()
def chunk():
    """Chunk all extracted papers -> data/processed/chunks.jsonl."""
    setup_logging(settings.log_level)
    files = sorted(INTERIM_DIR.glob("*.json"))
    if not files:
        console.print("[red]Nothing to chunk — run `extract` first.[/]")
        raise typer.Exit(1)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "chunks.jsonl"
    total = 0
    with out_path.open("w", encoding="utf-8") as f:
        for path in files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            chunks = chunk_paper(payload["arxiv_id"], payload["title"], payload["pages"])
            for c in chunks:
                f.write(c.model_dump_json() + "\n")
            total += len(chunks)
            console.print(f"  {payload['arxiv_id']}: {len(chunks)} chunks")
    console.print(f"[green]✔ wrote {total} chunks -> {out_path}[/]")


@app.command()
def pipeline(topic: str, max_papers: int = typer.Option(5)):
    """Full Phase-1A pipeline: fetch -> extract -> chunk."""
    fetch(topic=topic, max_papers=max_papers)
    extract()
    chunk()


if __name__ == "__main__":
    app()

