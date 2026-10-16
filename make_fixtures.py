"""Create synthetic local papers so the pipeline can be tested offline.

Usage:
    python make_fixtures.py
    citesage extract
    citesage chunk
"""
from pathlib import Path

import pymupdf

from citesage.config import RAW_DIR
from citesage.fetch.storage import PaperStore
from citesage.models import Paper

# 9999.* IDs = obviously synthetic, easy to purge before real indexing
PAPERS = [
    {
        "arxiv_id": "9999.00001v1",
        "title": "Synthetic Study of Retrieval Augmented Generation",
        "authors": ["Ada Lovelace", "Alan Turing"],
        "abstract": "We study retrieval augmented generation evaluation with synthetic data.",
        "pages": [
            "Introduction\n" + "Grounding matters for trustworthy answers. " * 40,
            "Methods\n" + "We chunk documents into passages with overlap. " * 40,
            "Results\n" + "Hybrid retrieval improved citation recall notably. " * 40,
        ],
    },
    {
        "arxiv_id": "9999.00002v1",
        "title": "Evaluating Citation Faithfulness in Language Models",
        "authors": ["Grace Hopper"],
        "abstract": "We propose metrics for citation precision and recall in RAG systems.",
        "pages": [
            "Abstract\n" + "Citations must be verifiable. " * 40,
            "Experiments\n" + "We benchmark vector and hybrid retrieval approaches. " * 40,
            "Conclusion\n" + "Evaluation drives better retrieval systems. " * 40,
        ],
    },
]


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    store = PaperStore(RAW_DIR / "papers.json")

    for spec in PAPERS:
        pdf_path = RAW_DIR / f"{spec['arxiv_id']}.pdf"

        doc = pymupdf.open()
        for page_text in spec["pages"]:
            page = doc.new_page()
            page.insert_textbox(page.rect, page_text, fontsize=11)
        doc.save(pdf_path)
        doc.close()

        paper = Paper(
            arxiv_id=spec["arxiv_id"],
            title=spec["title"],
            authors=spec["authors"],
            abstract=spec["abstract"],
        )
        paper.pdf_path = str(pdf_path)
        store.upsert(paper)
        print(f"created {pdf_path}")

    store.save()
    print("\nfixtures ready — now run:\n  citesage extract\n  citesage chunk")


if __name__ == "__main__":
    main()