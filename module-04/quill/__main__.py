"""CLI entry point so ``uv run python -m quill <csv> <question>`` works too.

As of Module 4 the canonical command is ``python -m quill.run data/sales.csv "<question>"``
(it prints the backend/model and the run's token cost). ``python -m quill`` delegates to that
same cost-aware entry point. The Module 3 trajectory printer still lives at
``python -m quill.agent`` if you want the full ReAct replay instead.
"""
from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
