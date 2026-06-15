"""Quill's step bench (Module 7) — measure the step drop from planning + sharpened prompt.

The headline of Module 7 is a *measurement*: same dataset, same question, fewer LLM calls.
This module runs Quill TWICE on one question and prints a comparison:

    uv run python -m quill.bench --question "Which category grew fastest, and is that growth
        statistically meaningful?" --dataset data/sales.csv

prints something like::

    Baseline  (no planning, bare instructions)        : 11 ActionSteps, ~11 LLM calls
    Improved  (planning_interval=3, sharpened)        :  6 ActionSteps, ~8 LLM calls
    Step reduction: ~45%   (your numbers will vary — LLMs are non-deterministic)

Two honesty rules the article hammers and this code obeys:

1. **Count ``ActionStep``s, not "all steps".** A ``PlanningStep`` is itself one LLM call, so
   the improved config's *total* step count includes the planning calls. We report the
   ``ActionStep`` count (the work the agent actually did) AND the planning-call count
   separately, so "~N LLM calls" for the improved run = ActionSteps + PlanningSteps. Comparing
   raw ``len(memory.steps)`` would flatter or punish planning dishonestly.
2. **The numbers vary every run.** LLMs are non-deterministic; this is a *trend*, not a
   guarantee. The blog's "~30% fewer steps" claim is *reported*, measured against JSON tool
   calling — never attribute it to planning alone.

This bench makes REAL LLM calls (two runs), so its CLI is a ``live`` path: it needs
``HF_TOKEN`` (or a swapped backend). The pure helper ``count_steps(agent)`` and
``run_and_count(agent, task)`` take an already-built agent, so the offline tests drive them
with the shared FakeModel and spend zero tokens.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from smolagents import ActionStep, CodeAgent, PlanningStep

from .agent import DEFAULT_PLANNING_INTERVAL, build_quill, build_task


@dataclass
class StepCount:
    """The two counts that matter when comparing configs (06 §6: count ActionSteps honestly).

    ``action_steps`` is the work Quill did; ``planning_steps`` is the extra tool-free LLM calls
    a periodic plan adds. ``llm_calls`` is their sum — the closest honest proxy for "how many
    times did we hit the model" from memory alone (each ActionStep and each PlanningStep is one
    ``model.generate`` call).
    """

    action_steps: int
    planning_steps: int

    @property
    def llm_calls(self) -> int:
        """ActionSteps + PlanningSteps — every step type here is exactly one LLM call."""
        return self.action_steps + self.planning_steps


def count_steps(agent: CodeAgent) -> StepCount:
    """Count ``ActionStep``s and ``PlanningStep``s in an agent's memory (after a run).

    This is the honest metric (06 §6): we report ``ActionStep``s separately from
    ``PlanningStep``s so adding planning never looks "free". Reads ``agent.memory.steps`` (the
    supported way — NEVER ``agent.logs``, removed in 1.21.0).

    Args:
        agent: an agent that has already ``run()`` at least once.

    Returns:
        A ``StepCount`` with ``action_steps``, ``planning_steps`` and the derived ``llm_calls``.
    """
    actions = sum(1 for s in agent.memory.steps if isinstance(s, ActionStep))
    plans = sum(1 for s in agent.memory.steps if isinstance(s, PlanningStep))
    return StepCount(action_steps=actions, planning_steps=plans)


def run_and_count(agent: CodeAgent, task: str) -> StepCount:
    """Run one task on ``agent`` and return its ``StepCount`` — the unit the bench compares.

    Args:
        agent: a built Quill (real or fake-model).
        task: the full task string (use ``build_task(csv, question)``).

    Returns:
        The ``StepCount`` for the run.
    """
    agent.run(task)
    return count_steps(agent)


def _reduction_pct(baseline: int, improved: int) -> float:
    """Percent drop in ActionSteps, guarded against a zero baseline."""
    if baseline <= 0:
        return 0.0
    return 100.0 * (baseline - improved) / baseline


def format_report(baseline: StepCount, improved: StepCount) -> str:
    """Render the baseline-vs-improved comparison block (the bench's observable output).

    The reduction is computed on ``action_steps`` (the work), and the per-line ``~N LLM calls``
    uses ``llm_calls`` (ActionSteps + PlanningSteps) so the planning cost is visible, not hidden.
    """
    pct = _reduction_pct(baseline.action_steps, improved.action_steps)
    return (
        f"Baseline  (no planning, bare instructions)        : "
        f"{baseline.action_steps:>2} ActionSteps, ~{baseline.llm_calls} LLM calls\n"
        f"Improved  (planning_interval={DEFAULT_PLANNING_INTERVAL}, sharpened)       : "
        f"{improved.action_steps:>2} ActionSteps, ~{improved.llm_calls} LLM calls "
        f"({improved.planning_steps} of them planning)\n"
        f"Step reduction: ~{pct:.0f}%   "
        f"(your numbers will vary — LLMs are non-deterministic)"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI: build baseline + improved Quill, run the same question on each, print the drop.

    ``--dataset <path>`` (default ``data/sales.csv``) and ``--question "<q>"`` (a multi-step
    default). Makes REAL LLM calls (two runs) — a ``live`` path: set ``HF_TOKEN`` or swap the
    backend via ``QUILL_MODEL_BACKEND``/``QUILL_MODEL_ID``.

    Baseline   = ``build_quill(instructions="")``      — bare system prompt, no planning.
    Improved   = ``build_quill(planning_interval=3)``  — periodic plan + Quill's default
                 sharpened ``instructions`` (the default when ``instructions`` is omitted).
    """
    args = sys.argv[1:] if argv is None else argv

    dataset = "data/sales.csv"
    question = (
        "Which category grew fastest from Q1 to Q4 2025, "
        "and is that growth statistically meaningful?"
    )
    if "--dataset" in args:
        dataset = args[args.index("--dataset") + 1]
    if "--question" in args:
        question = args[args.index("--question") + 1]

    task = build_task(dataset, question)

    print("===== BASELINE: no planning, bare instructions =====")
    # instructions="" -> the bare smolagents system prompt; no planning_interval -> no plan.
    with build_quill(instructions="") as baseline_agent:
        baseline = run_and_count(baseline_agent, task)

    print("\n===== IMPROVED: planning_interval=3 + sharpened instructions =====")
    # planning_interval defaults Quill's cadence; instructions omitted -> QUILL_INSTRUCTIONS.
    with build_quill(planning_interval=DEFAULT_PLANNING_INTERVAL) as improved_agent:
        improved = run_and_count(improved_agent, task)

    print("\n===== STEP COMPARISON =====")
    print(format_report(baseline, improved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
