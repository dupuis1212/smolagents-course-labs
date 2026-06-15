"""Quill's cost-aware CLI (M4; M8: validated QuillReport; M11: vision; M12: agentic RAG).

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

**Module 11 — Quill checks its own chart.** Add ``--review`` (a VLM re-reads each saved chart via
``run(images=...)`` and appends a verdict to ``QuillReport.caveats`` — needs a VLM, no extra) and
``--browse`` (add the OPTIONAL ``vision_browser`` sub-agent for JS/chart-heavy pages — needs a
local Chrome + the ``[vision]`` extra + a VLM). Both are OFF by default so the cost-aware default
path is unchanged:

    QUILL_MODEL_ID="Qwen/Qwen2-VL-72B-Instruct" \\
        uv run python -m quill "Analyze data/sales.csv and chart monthly revenue, then check the chart yourself." --review

**Module 12 — Quill stops guessing what a column means.** The manager carries a ``RetrieverTool``
(BM25 over ``data/corpus/*.md`` — a data dictionary + business rules). Add ``--retrieve`` to phrase
the task so Quill looks up an ambiguous column (``net_rev``, ``churn_flag``) in the corpus and cites
the doc as a ``Source`` in ``QuillReport.sources`` (``[n]``) instead of inventing the meaning. It is
AGENTIC RAG — the agent decides when to call the retriever and may reformulate; no LLM is needed for
the retrieval itself (BM25 is lexical):

    uv run python -m quill "What was net_rev growth, and define net_rev precisely?" --retrieve

Swap the backend without touching any agent code:

    QUILL_MODEL_BACKEND=litellm QUILL_MODEL_ID="gpt-4o" \\
        uv run python -m quill "..." --data data/sales.csv

We read tokens via ``Monitor`` (NOT the removed ``agent.logs`` or legacy token attributes,
gone since 1.21.0). No silent try/except: if a run errors, you see it.
"""
from __future__ import annotations

import sys

from .agent import build_quill, build_report_task, build_retrieval_task, review_charts
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
    ``data/sales.csv``). The M11 boolean flags ``--review``/``--browse`` are recognised by
    ``main`` (see ``_flags``) and not treated as the question. No silent failure: an unknown flag
    is left in the positionals and ignored by position, which keeps the CLI forgiving.
    """
    csv_path = "data/sales.csv"
    positional: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--data" and i + 1 < len(args):
            csv_path = args[i + 1]
            i += 2
            continue
        if args[i].startswith("-"):  # a boolean flag (e.g. --review/--browse), not the question
            i += 1
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
    review = "--review" in args   # M11: VLM re-reads the saved charts (needs a VLM)
    browse = "--browse" in args   # M11: add the vision_browser (needs local Chrome + [vision])
    retrieve = "--retrieve" in args  # M12: phrase the task to look up + cite ambiguous columns

    # Build Quill with the env-selected model (make_model() under the hood) and Quill's default
    # final_answer_checks (M8). browse=True (M11) adds the vision_browser to the team — stays local.
    # The RetrieverTool is on by default (M12): Quill can look up an ambiguous column's meaning in
    # data/corpus and cite it. The line below is the ONLY place the run announces what powers it.
    agent = build_quill(browse=browse) if browse else build_quill()
    print(f"[Quill] Backend: {Settings.MODEL_BACKEND} | Model: {Settings.MODEL_ID}")

    # M12: --retrieve phrases the task so Quill looks up an ambiguous column (net_rev, churn_flag)
    # in the data dictionary and cites the corpus doc as a Source — grounding, not guessing. The
    # default build_report_task otherwise tells Quill to package its answer as a QuillReport; the
    # default checks reject a report with no saved chart / no source for a web claim, so the
    # trajectory may show a "Final answer rejected: ..." line followed by a corrective step.
    task = build_retrieval_task(csv_path, question) if retrieve else build_report_task(csv_path, question)
    result = agent.run(task)

    # M11: Quill RE-READS its own charts with a VLM (run(images=...)) and the verdict lands in
    # QuillReport.caveats. A SEPARATE end-of-run call (a VLM call is expensive — not every step).
    # Off by default (the cost-aware path stays text-only); --review turns it on. Needs a VLM.
    if review and isinstance(result, QuillReport) and result.chart_paths:
        print("\n===== CHART SELF-REVIEW (VLM re-reads the charts via run(images=...)) =====")
        result = review_charts(result)
        for caveat in result.caveats:
            if caveat.startswith("Chart review "):
                print(f"  {caveat}")

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
