"""The eval harness — golden set, cost/run, and a regression gate (NEW, Module 14).

The assembly of the module: a reproducible script that runs Quill over a frozen **golden set**,
scores each run, aggregates **Task Success Rate (TSR)** + **cost/run**, and applies a **regression
gate** that exits non-zero when either crosses a threshold. "A small prompt tweak" can no longer
silently make Quill worse.

    uv run python -m quill.eval.run_evals --out eval/results/run-baseline.json

Frozen format (06-FIL-ROUGE-SPEC §2 — M15 reuses these keys VERBATIM; a new key = STOP, update the
spec first):

    golden_set.json = [{id, question, dataset, expected_points[], min_sources}]
    run-<name>.json = {
        run_name, model,
        scores: [{id, task_success, report_quality, citations, steps, cost}],
        aggregate: {TSR, avg_report_quality, avg_steps, cost_per_run},
    }

The five score columns, and how each is computed (outcome vs trajectory):
- ``task_success``    OUTCOME   : does the report cover the expected_points? (the JUDGE, M14).
- ``report_quality`` OUTCOME    : the judge's rubric total out of 6 (the JUDGE, M14).
- ``citations``       OUTCOME    : ``len(report.sources) >= min_sources`` — DETERMINISTIC, no LLM.
- ``steps``           TRAJECTORY : ``len([s for s in agent.memory.steps if ActionStep])`` (M6).
- ``cost``            TRAJECTORY : ``agent.monitor.get_total_token_counts().total_tokens`` (M4 —
                                  the Monitor accessor, never the token attrs removed in 1.21.0).

The gate (06 §6): compare the aggregate to ``QUILL_EVAL_MIN_TSR`` (default 0.70) and
``QUILL_EVAL_MAX_COST`` (default off / +inf — a token budget per run). ``sys.exit(1)`` if either
fails, so the harness is usable straight in CI. Compare ``run-baseline.json`` vs
``run-candidate.json`` to catch a regression.

Scope (06 §2): the eval runs on the CURRENT ``QUILL_EXECUTOR``. The harness itself does not harden
Quill — ``quill/runtime.py`` (M15) owns Approach 2 + the bounded-retry/idempotence wiring.

**Module 15 (the capstone) makes this gate the "green or no ship" release gate.** Nothing in the
scoring changes (the FROZEN keys + the ``apply_regression_gate`` logic are reused VERBATIM); the
capstone simply PROMOTES the gate's non-zero exit to a release blocker: ``python -m
quill.eval.run_evals --out eval/results/run-v1.0.json`` must exit 0 (TSR >= the floor, cost/run <=
the budget) before you tag ``v1.0``. A regression fails the build — you do not ship a Quill you
cannot defend with numbers (T12.11).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys

from ..agent import build_quill, build_report_task
from ..config import Settings, DEFAULT_MODEL_ID
from ..report import QuillReport
from .judge import JUDGE_RUBRIC_MAX, TASK_SUCCESS_COVERAGE_THRESHOLD, judge_report

# Resolve paths relative to THIS module-NN/ dir so the harness works from any cwd (the cumulative
# suite runs from the repo root). quill/eval/run_evals.py -> module-NN/.
MODULE_DIR = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_GOLDEN_SET = str(pathlib.Path(__file__).resolve().parent / "golden_set.json")
DEFAULT_RESULTS_DIR = MODULE_DIR / "quill" / "eval" / "results"

# The FROZEN per-item score keys (06 §2) — do NOT add/rename a key without updating the spec first.
SCORE_KEYS = ("id", "task_success", "report_quality", "citations", "steps", "cost")

# Regression-gate defaults (06 §6). MIN_TSR is the floor; MAX_COST is a per-run token budget that is
# OFF by default (+inf) so the gate only checks cost when you opt in via QUILL_EVAL_MAX_COST.
DEFAULT_MIN_TSR = 0.70
DEFAULT_MAX_COST = math.inf


def load_golden_set(path: str | None = None) -> list[dict]:
    """Load the frozen golden set (a list of ``{id, question, dataset, expected_points[],
    min_sources}``).

    Args:
        path: path to ``golden_set.json``; defaults to the one shipped next to this module.

    Returns:
        The parsed list of golden-set items.

    Raises:
        ValueError: if an item is missing a required frozen key — fail loud, the format is a
            contract M15 depends on (06 §2).
    """
    path = path or DEFAULT_GOLDEN_SET
    with open(path, encoding="utf-8") as handle:
        items = json.load(handle)
    required = {"id", "question", "dataset", "expected_points", "min_sources"}
    for item in items:
        missing = required - set(item)
        if missing:
            raise ValueError(
                f"golden_set item {item.get('id', '?')!r} is missing frozen keys {sorted(missing)} "
                f"(06 §2: {sorted(required)})."
            )
    return items


def _resolve_dataset(dataset: str) -> str:
    """Resolve a golden-set ``dataset`` path: as-is if it exists, else relative to module-NN/."""
    if os.path.exists(dataset):
        return dataset
    candidate = MODULE_DIR / dataset
    return str(candidate) if candidate.exists() else dataset


def count_action_steps(agent) -> int:
    """Count ActionSteps in the agent's memory (the trajectory length, M6).

    Reads ``agent.memory.steps`` (the supported way — NEVER the removed ``agent.logs``) and counts
    only ``ActionStep``s, so a ``PlanningStep`` / ``TaskStep`` does not inflate the step count.
    """
    from smolagents import ActionStep

    steps = getattr(getattr(agent, "memory", None), "steps", None) or []
    return len([s for s in steps if isinstance(s, ActionStep)])


def run_cost_tokens(agent) -> int:
    """Read the run's total token cost via ``Monitor`` (M4 — the frozen cost accessor).

    Uses ``agent.monitor.get_total_token_counts().total_tokens`` — the Monitor aggregates per-step
    token usage across the whole run. We never touch the per-agent token attributes removed in
    1.21.0. Returns 0 if no monitor is present (defensive — a fake-model run still has one).
    """
    monitor = getattr(agent, "monitor", None)
    if monitor is None:
        return 0
    usage = monitor.get_total_token_counts()
    return int(getattr(usage, "total_tokens", 0) or 0)


def evaluate_item(item: dict, *, model=None, judge_model=None, agent=None) -> dict:
    """Run Quill on ONE golden item and score it — returns one frozen ``scores[]`` entry.

    The per-item unit ``run_evals`` loops. It:
    1. builds Quill (``build_quill``) unless an ``agent`` is injected (tests pass a fake-model one),
    2. runs the report task (``build_report_task``) to get a ``QuillReport``,
    3. scores it: ``task_success`` + ``report_quality`` from the JUDGE; ``citations`` DETERMINISTIC
       (``len(report.sources) >= min_sources``, no LLM); ``steps`` + ``cost`` from the trajectory.

    Args:
        item: a golden-set entry.
        model: the model that powers Quill; ``None`` -> ``make_model`` (via ``build_quill``). Tests
            pass a fake model. Ignored when ``agent`` is supplied.
        judge_model: the SEPARATE judge model (06 §2: ideally a different model_id than Quill — never
            self-grading). ``None`` -> ``make_model(role="judge")`` with ``QUILL_JUDGE_MODEL_ID``.
        agent: an already-built agent to reuse (tests inject a fake-model Quill). When ``None`` a
            fresh ``build_quill(model=model)`` is created per item (clean memory per item).

    Returns:
        A dict with EXACTLY the frozen keys ``{id, task_success, report_quality, citations, steps,
        cost}`` (06 §2), plus a non-frozen ``rationale`` for debugging (ignored by the aggregate).
    """
    dataset = _resolve_dataset(item["dataset"])
    min_sources = int(item.get("min_sources", 0))

    own_agent = agent is None
    if own_agent:
        agent = build_quill(model=model)
    try:
        output = agent.run(build_report_task(dataset, item["question"]))
    finally:
        if own_agent:
            _cleanup(agent)

    report = output if isinstance(output, QuillReport) else QuillReport(question=item["question"])

    # citations: DETERMINISTIC, no LLM — did the report carry at least min_sources sources?
    citations = 1 if len(report.sources) >= min_sources else 0

    # the judge scores the outcome axes (coverage drives task_success; total -> report_quality).
    if judge_model is None:
        judge_model = _make_judge_model()
    judged = judge_report(report, item, judge_model)
    coverage = judged["scores"].get("coverage", 0)
    task_success = (
        1 if coverage >= TASK_SUCCESS_COVERAGE_THRESHOLD * (JUDGE_RUBRIC_MAX // 3) else 0
    )

    return {
        "id": item["id"],
        "task_success": task_success,
        "report_quality": judged["report_quality"],  # 0..JUDGE_RUBRIC_MAX (6)
        "citations": citations,
        "steps": count_action_steps(agent),
        "cost": run_cost_tokens(agent),
        "rationale": judged.get("rationale", ""),  # non-frozen: debugging only
    }


def aggregate_scores(scores: list[dict]) -> dict:
    """Aggregate the per-item scores into ``{TSR, avg_report_quality, avg_steps, cost_per_run}``.

    - ``TSR`` (Task Success Rate) = mean of ``task_success`` (the OUTCOME headline).
    - ``avg_report_quality`` = mean of the judge's rubric total.
    - ``avg_steps`` = mean trajectory length.
    - ``cost_per_run`` = mean total tokens per run (the budget line — TRAJECTORY).
    """
    n = len(scores) or 1
    return {
        "TSR": sum(s["task_success"] for s in scores) / n,
        "avg_report_quality": sum(s["report_quality"] for s in scores) / n,
        "avg_steps": sum(s["steps"] for s in scores) / n,
        "cost_per_run": sum(s["cost"] for s in scores) / n,
    }


def apply_regression_gate(
    aggregate: dict,
    *,
    min_tsr: float = DEFAULT_MIN_TSR,
    max_cost: float = DEFAULT_MAX_COST,
) -> tuple[bool, list[str]]:
    """Apply the regression gate: PASS only if ``TSR >= min_tsr`` AND ``cost_per_run <= max_cost``.

    Args:
        aggregate: the dict from :func:`aggregate_scores`.
        min_tsr: the minimum acceptable Task Success Rate (``QUILL_EVAL_MIN_TSR``, default 0.70).
        max_cost: the maximum acceptable cost/run in tokens (``QUILL_EVAL_MAX_COST``, default +inf
            = no cost gate).

    Returns:
        ``(passed, reasons)`` — ``passed`` is ``True`` only when every check holds; ``reasons``
        lists each failed check (empty on pass). The caller turns a fail into ``sys.exit(1)``.
    """
    reasons: list[str] = []
    if aggregate["TSR"] < min_tsr:
        reasons.append(f"TSR {aggregate['TSR']:.2f} < min {min_tsr:.2f}")
    if aggregate["cost_per_run"] > max_cost:
        reasons.append(f"cost/run {aggregate['cost_per_run']:.0f} > budget {max_cost:.0f}")
    return (not reasons, reasons)


def run_evals(
    *,
    run_name: str,
    golden_set: list[dict] | None = None,
    golden_set_path: str | None = None,
    model=None,
    judge_model=None,
    agent_factory=None,
    min_tsr: float = DEFAULT_MIN_TSR,
    max_cost: float = DEFAULT_MAX_COST,
) -> dict:
    """Run Quill over the golden set, score every item, aggregate, and apply the gate.

    Args:
        run_name: the label for this run (e.g. ``"baseline"``) — lands in the frozen ``run_name``.
        golden_set: an in-memory golden set (tests pass a tiny one); else loaded from
            ``golden_set_path`` / the default.
        golden_set_path: path to ``golden_set.json``.
        model: the model that powers Quill; ``None`` -> ``make_model`` (a real run). Tests pass a
            fake model.
        judge_model: the SEPARATE judge model; ``None`` -> ``make_model(role="judge")``.
        agent_factory: a 0-arg callable returning a fresh agent per item (tests inject a fake-model
            Quill builder so the whole harness runs offline). When ``None``, ``evaluate_item``
            builds Quill itself from ``model``.
        min_tsr / max_cost: the regression-gate thresholds.

    Returns:
        The FROZEN results dict ``{run_name, model, scores: [...], aggregate, gate}`` where each
        ``scores[]`` entry has the frozen keys. ``gate`` is non-frozen metadata (pass/fail + reasons)
        the CLI uses for its exit code.
    """
    items = golden_set if golden_set is not None else load_golden_set(golden_set_path)

    scores: list[dict] = []
    for item in items:
        agent = agent_factory() if agent_factory is not None else None
        try:
            entry = evaluate_item(item, model=model, judge_model=judge_model, agent=agent)
        finally:
            if agent is not None:
                _cleanup(agent)
        scores.append(entry)

    aggregate = aggregate_scores(scores)
    passed, reasons = apply_regression_gate(aggregate, min_tsr=min_tsr, max_cost=max_cost)

    return {
        "run_name": run_name,
        "model": Settings.MODEL_ID,
        "scores": scores,
        "aggregate": aggregate,
        "gate": {
            "passed": passed,
            "reasons": reasons,
            "min_tsr": min_tsr,
            "max_cost": (None if max_cost == math.inf else max_cost),
        },
    }


def format_summary(results: dict) -> str:
    """One readable block: golden-set size, model, TSR, avg quality/steps, cost/run, gate verdict.

    Mirrors the brief's observable output:
        Golden set: 5 tasks · model=Qwen/Qwen2.5-Coder-32B-Instruct
        TSR: 0.80 (4/5)  ·  avg report_quality: 4.6/6  ·  avg steps: 6.2  ·  cost/run: ~12.4k tokens
        Regression gate: PASS (TSR>=0.70, cost/run<=budget)
    """
    scores = results["scores"]
    agg = results["aggregate"]
    n = len(scores)
    wins = sum(s["task_success"] for s in scores)
    gate = results["gate"]
    verdict = "PASS" if gate["passed"] else "FAIL"
    cost_k = agg["cost_per_run"] / 1000.0
    budget = "no cap" if gate["max_cost"] is None else f"<={gate['max_cost']:.0f}"
    lines = [
        f"Golden set: {n} tasks · model={results['model']}",
        f"TSR: {agg['TSR']:.2f} ({wins}/{n})  ·  "
        f"avg report_quality: {agg['avg_report_quality']:.1f}/{JUDGE_RUBRIC_MAX}  ·  "
        f"avg steps: {agg['avg_steps']:.1f}  ·  cost/run: ~{cost_k:.1f}k tokens",
        f"Regression gate: {verdict} (TSR>={gate['min_tsr']:.2f}, cost/run {budget})",
    ]
    if not gate["passed"]:
        lines.append("  failed: " + "; ".join(gate["reasons"]))
    return "\n".join(lines)


def write_results(results: dict, out_path: str) -> str:
    """Write the frozen results dict to ``out_path`` (creating ``eval/results/`` if needed)."""
    path = pathlib.Path(out_path)
    if not path.is_absolute():
        path = MODULE_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return str(path)


def _cleanup(agent) -> None:
    """Best-effort sandbox teardown (a remote executor needs it; the local one is a no-op)."""
    cleanup = getattr(agent, "cleanup", None)
    if callable(cleanup):
        try:
            cleanup()
        except Exception:  # pragma: no cover - teardown must never mask a real eval result
            pass


def _make_judge_model():
    """Build the SEPARATE judge model (06 §2): ``make_model`` with ``QUILL_JUDGE_MODEL_ID``.

    The judge ALWAYS goes through the single model factory (no second factory). By default we point
    it at ``QUILL_JUDGE_MODEL_ID`` if set (ideally a different/stronger model than Quill — never
    self-grading); otherwise it falls back to Quill's own model_id with a clear caveat that you
    should swap it for real evaluation.
    """
    from ..config import make_model

    judge_id = os.environ.get("QUILL_JUDGE_MODEL_ID")
    if judge_id:
        return make_model(role="judge", model_id=judge_id)
    return make_model(role="judge")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m quill.eval.run_evals",
        description="Run Quill over the golden set, score it (judge + deterministic), and gate.",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_RESULTS_DIR / "run-baseline.json"),
        help="where to write the results JSON (default: eval/results/run-baseline.json).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="run name (default: derived from --out, e.g. run-baseline.json -> baseline).",
    )
    parser.add_argument(
        "--golden-set",
        default=None,
        help="path to golden_set.json (default: the one shipped with the module).",
    )
    parser.add_argument(
        "--min-tsr",
        type=float,
        default=float(os.environ.get("QUILL_EVAL_MIN_TSR", DEFAULT_MIN_TSR)),
        help="regression gate: minimum Task Success Rate (QUILL_EVAL_MIN_TSR, default 0.70).",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=float(os.environ.get("QUILL_EVAL_MAX_COST", "inf")),
        help="regression gate: max cost/run in tokens (QUILL_EVAL_MAX_COST, default off).",
    )
    return parser


def _name_from_out(out_path: str) -> str:
    """Derive a run name from --out: ``.../run-baseline.json`` -> ``baseline``."""
    stem = pathlib.Path(out_path).stem  # run-baseline
    return stem[len("run-"):] if stem.startswith("run-") else stem


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Runs the golden set with the env-selected model, writes the JSON, prints the
    summary, and EXITS NON-ZERO if the regression gate fails (usable straight in CI).

    Cost note (06 §2): one eval run = N Quill runs + N judge calls. With the HF free tier at
    ~$0.10/month (as of smolagents 1.26.0, subject to change), a 5-item golden set of multi-step
    questions plus 5 judge calls can eat a real slice of that — the cost/run column is not cosmetic.
    """
    args = _build_arg_parser().parse_args(argv)
    run_name = args.name or _name_from_out(args.out)

    results = run_evals(
        run_name=run_name,
        golden_set_path=args.golden_set,
        min_tsr=args.min_tsr,
        max_cost=args.max_cost,
    )
    out_path = write_results(results, args.out)
    print(format_summary(results))
    print(f"Wrote {out_path}")

    # The gate's whole point: a non-zero exit fails the build (CI) when Quill regressed.
    return 0 if results["gate"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
