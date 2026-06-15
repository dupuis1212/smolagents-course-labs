"""CLI: ``uv run python -m quill_intro "<task>"``

Prints the agent's trajectory (the Python it writes) and the final answer.
"""
from __future__ import annotations

import sys

from .first_agent import build_first_agent

DEFAULT_TASK = "Calculate the sum of all integers from 1 to 100"


def main() -> None:
    task = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TASK
    agent = build_first_agent()
    result = agent.run(task)
    print(result)


if __name__ == "__main__":
    main()
