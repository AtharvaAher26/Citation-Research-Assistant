from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """All knobs in one place. Override any of them with CITESAGE_* env vars."""

    data_dir: Path = Path("data")

    # fetching — arXiv asks for <= 1 request / 3 seconds. We comply.
    arxiv_request_delay: float = 3.0

    # chunking — tuned for all-MiniLM-L6-v2 (max sequence = 256 tokens)
    chunk_tokens: int = 220
    chunk_overlap: int = 30

    # validation gates — garbage PDFs must never reach the index
    min_page_chars: int = 200
    min_paper_chars: int = 2000

    log_level: str = "INFO"

    model_config = {"env_prefix": "CITESAGE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()
RAW_DIR = settings.data_dir / "raw"
INTERIM_DIR = settings.data_dir / "interim"
PROCESSED_DIR = settings.data_dir / "processed"