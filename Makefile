.PHONY: setup test lint pipeline
setup:
    python -m venv .venv && . .venv/bin/activate && pip install -e ".[dev]"
test:
    pytest -q
lint:
    ruff check src tests
pipeline:
    citesage pipeline "$(TOPIC)" --max-papers 5