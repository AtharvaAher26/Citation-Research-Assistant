"""Central configuration for CitationSage."""

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

    # --- Phase 1C: generation layer ---
    groq_api_key: str = ""
    gemini_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    gemini_model: str = "gemini-2.5-flash"
    llm_timeout_seconds: float = 30.0
    llm_max_output_tokens: int = 1024

    model_config = {"env_prefix": "CITESAGE_", "env_file": ".env", "extra": "ignore"}


settings = Settings()

RAW_DIR = settings.data_dir / "raw"
INTERIM_DIR = settings.data_dir / "interim"
PROCESSED_DIR = settings.data_dir / "processed"