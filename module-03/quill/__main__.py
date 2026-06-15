"""CLI entry point so ``uv run python -m quill <csv> <question>`` works too.

Delegates to :func:`quill.agent.main` (the lab's canonical command is
``python -m quill.agent data/sales.csv "<question>"``).
"""
from .agent import main

if __name__ == "__main__":
    main()
