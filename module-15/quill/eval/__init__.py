"""Quill's evaluation harness (NEW, Module 14) — golden set + LLM-as-judge + regression gate.

"It looks like it works" is not a metric. This package turns that into defensible numbers: a
frozen **golden set** of data questions over ``data/sales.csv`` (inherited M2), an **LLM-as-judge**
that scores each ``QuillReport`` (the FROZEN M8 schema — reused, never modified), and a **regression
gate** that fails the build when Task Success Rate drops or cost/run jumps.

Frozen format (06-FIL-ROUGE-SPEC §2 — M15 reuses these keys verbatim):

- ``golden_set.json`` = a list of ``{id, question, dataset, expected_points[], min_sources}``.
- ``run_evals.py --out eval/results/run-<name>.json`` produces
  ``{run_name, model, scores: [{id, task_success, report_quality, citations, steps, cost}],
  aggregate}``.

The public API:

- :func:`load_golden_set` — read the frozen golden set.
- :func:`judge_report` — the LLM-as-judge (rubric + evidence-before-score + structured output).
- :func:`run_evals` — run Quill over the golden set, score, aggregate, apply the gate.
- :func:`evaluate_item` — score ONE golden-set item (the per-item unit the harness loops).

Cost/run is read via ``Monitor.get_total_token_counts()`` (M4) — never the token attributes removed
in 1.21.0. ``citations`` is computed deterministically (no LLM) against ``QuillReport.sources``.
"""
from __future__ import annotations

from .judge import (
    JUDGE_RUBRIC_MAX,
    build_judge_prompt,
    judge_report,
    parse_judge_response,
)
from .run_evals import (
    DEFAULT_GOLDEN_SET,
    DEFAULT_MAX_COST,
    DEFAULT_MIN_TSR,
    SCORE_KEYS,
    aggregate_scores,
    apply_regression_gate,
    count_action_steps,
    evaluate_item,
    format_summary,
    load_golden_set,
    run_evals,
    run_cost_tokens,
)

__all__ = [
    "JUDGE_RUBRIC_MAX",
    "build_judge_prompt",
    "judge_report",
    "parse_judge_response",
    "DEFAULT_GOLDEN_SET",
    "DEFAULT_MAX_COST",
    "DEFAULT_MIN_TSR",
    "SCORE_KEYS",
    "aggregate_scores",
    "apply_regression_gate",
    "count_action_steps",
    "evaluate_item",
    "format_summary",
    "load_golden_set",
    "run_evals",
    "run_cost_tokens",
]
