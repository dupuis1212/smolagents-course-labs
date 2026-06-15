"""Quill's cost-aware CLI (Module 4; Module 8: returns a validated QuillReport).

Run Quill on a CSV + question, then print which backend/model powered the run, the answer, and
the token cost of the whole run — read from ``agent.monitor.get_total_token_counts()`` (a
``TokenUsage`` aggregated across every step). This is the "cost in plain sight" half of the M4
contract; the recipe is reused by the eval harness in Module 14.

    uv run python -m quill "Which category grew fastest, and is that consistent with the public trend?" --data data/sales.csv

As of Module 8, Quill is asked to return a ``QuillReport`` (``build_report_task``) and the
default ``final_answer_checks`` refuse a report with no saved chart, or with web claims but no
source — so you may see a rejection line in the trajectory followed by a corrective step. The
final answer is rendered as Markdown via ``QuillReport.to_markdown()`` (numbered ``[n]``
citations into ``sources``); a plain answer is printed as-is.

Swap the backend without touching any agent code:

    QUILL_MODEL_BACKEND=litellm QUILL_MODEL_ID="gpt-4o" \\
        uv run python -m quill "..." --data data/sales.csv

We read tokens via ``Monitor`` (NOT the removed ``agent.logs`` or legacy token attributes,
gone since 1.21.0). No silent try/except: if a run errors, you see it.
"""
from __future__ import annotations

import sys

from .agent import build_quill, build_report_task
from .config import Settings
from .report import QuillReport


def _format_cost(usage) -> str:
    """One readable line for the run's token cost, with thousands separators."""
    return (
        f"[Quill] Run cost — input tokens: {usage.input_tokens:,} | "
        f"output tokens: {usage.output_tokens:,} | total: {usage.total_tokens:,}"
    )


def _parse_args(args: list[str]) -> tuple[str, str]:
    """Parse ``<question> [--data <csv>]`` (06 §6 observable result). Defaults are sensible.

    Accepts the question as the first positional and an optional ``--data <path>`` flag (default
    ``data/sales.csv``). No silent failure: an unknown flag is left in the positionals and ignored
    by position, which keeps the CLI forgiving without hiding a bad question.
    """
    csv_path = "data/sales.csv"
    positional: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--data" and i + 1 < len(args):
            csv_path = args[i + 1]
            i += 2
            continue
        positional.append(args[i])
        i += 1
    question = (
        positional[0]
        if positional
        else "Which category grew fastest, and is that consistent with the public trend?"
    )
    return csv_path, question


def main(argv: list[str] | None = None) -> int:
    """Run Quill once and print the backend, model, the report, and token cost. Returns a code."""
    args = sys.argv[1:] if argv is None else argv
    csv_path, question = _parse_args(args)

    # Build Quill with the env-selected model (make_model() under the hood) and Quill's default
    # final_answer_checks (M8). The line below is the ONLY place the run announces what powers it.
    agent = build_quill()
    print(f"[Quill] Backend: {Settings.MODEL_BACKEND} | Model: {Settings.MODEL_ID}")

    # build_report_task tells Quill to package its answer as a QuillReport; the default checks then
    # reject a report with no saved chart / no source for a web claim, so the trajectory may show a
    # "Final answer rejected: ..." line followed by a corrective step before the accepted answer.
    result = agent.run(build_report_task(csv_path, question))

    print("\n===== REPORT =====")
    # M8: a validated QuillReport renders to Markdown with numbered [n] citations; anything else
    # (e.g. an opted-out plain answer) prints as-is.
    print(result.to_markdown() if isinstance(result, QuillReport) else result)

    # Cost in plain sight: a Monitor accumulates per-step token counts and aggregates them
    # into a TokenUsage for the whole run. A multi-step CodeAgent burns tokens fast, so this
    # line is your guard-rail (it becomes a real budget cap in production — see the article).
    usage = agent.monitor.get_total_token_counts()
    print(_format_cost(usage))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
