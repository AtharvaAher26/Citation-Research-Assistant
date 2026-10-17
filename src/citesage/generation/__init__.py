"""Generation layer: grounded answers with mandatory, resolved citations.

No import-time side effects: network clients are constructed lazily by the
CLI, so tests and --help stay fast and fully offline.
"""