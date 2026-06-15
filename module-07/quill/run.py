"""Quill's cost-aware CLI (Module 4).

Run Quill on a CSV + question, then print which backend/model powered the run and the token
cost of the whole run — read from ``agent.monitor.get_total_token_counts()`` (a ``TokenUsage``
aggregated across every step). This is the "cost in plain sight" half of the M4 contract; the
recipe is reused by the eval harness in Module 14.

    uv run python -m quill.run data/sales.csv "Which category grew fastest last quarter?"

Swap the backend without touching any agent code:

    QUILL_MODEL_BACKEND=litellm QUILL_MODEL_ID="gpt-4o" \\
        uv run python -m quill.run data/sales.csv "..."

We read tokens via ``Monitor`` (NOT the removed ``agent.logs`` or legacy token attributes,
gone since 1.21.0). No silent try/except: if a run errors, you see it.
"""
from __future__ import annotations

import sys

from .agent import build_quill, build_task
from .config import Settings


def _format_cost(usage) -> str:
    """One readable line for the run's token cost, with thousands separators."""
    return (
        f"[Quill] Run cost — input tokens: {usage.input_tokens:,} | "
        f"output tokens: {usage.output_tokens:,} | total: {usage.total_tokens:,}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run Quill once and print the backend, model, and token cost. Returns an exit code."""
    args = sys.argv[1:] if argv is None else argv
    csv_path = args[0] if len(args) > 0 else "data/sales.csv"
    question = (
        args[1]
        if len(args) > 1
        else "Which category grew fastest from the first to the last quarter of 2025?"
    )

    # Build Quill with the env-selected model (make_model() under the hood). The two lines
    # below are the ONLY place the run announces what powers it.
    agent = build_quill()
    print(f"[Quill] Backend: {Settings.MODEL_BACKEND} | Model: {Settings.MODEL_ID}")

    result = agent.run(build_task(csv_path, question))
    print("\n===== ANSWER =====")
    print(result)

    # Cost in plain sight: a Monitor accumulates per-step token counts and aggregates them
    # into a TokenUsage for the whole run. A multi-step CodeAgent burns tokens fast, so this
    # line is your guard-rail (it becomes a real budget cap in production — see the article).
    usage = agent.monitor.get_total_token_counts()
    print(_format_cost(usage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
