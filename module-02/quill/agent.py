"""Quill v0 — the first stone of the trail.

`build_quill()` returns a bare ``CodeAgent`` (no custom tools yet) that, given a CSV and a
question, writes and runs pandas in the local interpreter and answers via ``final_answer``.

This is the canonical Quill construction entry point. Its signature will be EXTENDED (never
replaced) by later modules: custom tools in M3, ``make_model()`` in M4, ``executor_type`` in
M5, ``planning_interval``/``instructions`` in M7, ``final_answer_checks`` in M8,
``managed_agents`` in M10. Accepting an optional ``model=None`` now keeps the swap to
``make_model()`` (M4) painless.
"""
from __future__ import annotations

import os

from smolagents import CodeAgent, InferenceClientModel, Model

# Explicit, code-capable model_id. We do NOT rely on InferenceClientModel's default model_id:
# it is documented as "subject to change" (as of smolagents 1.26.0), and a data/code agent
# wants a coder/instruct model. Module 4 turns this into make_model() in quill/config.py.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# The sandbox does NOT authorize pandas/numpy by default — Quill must declare them. This list
# is also a *security* boundary (the threat model, the AST checks, and why you never use the
# "*" wildcard are Module 5). Quill keeps it minimal and extends it only by explicit addition.
QUILL_IMPORTS = ["pandas", "numpy"]


def build_quill(model: Model | None = None) -> CodeAgent:
    """Build Quill v0: a ``CodeAgent`` that analyzes a CSV by writing pandas.

    Pass your own ``model`` to run offline (e.g. a fake model in tests); otherwise a hosted
    Hugging Face model (``InferenceClientModel`` with an explicit ``model_id``) is used.

    Two knobs are set deliberately at this stage:
    - ``additional_authorized_imports`` — what the interpreter may import (pandas + numpy).
    - ``max_steps=8`` — the loop ceiling. The library default is 20; for a single-CSV
      analysis a low cap is plenty, and a runaway loop is a budget leak, not a feature.
    """
    return CodeAgent(
        tools=[],  # no custom tools yet — final_answer is always present (M3 adds tools)
        model=model or InferenceClientModel(
            model_id=DEFAULT_MODEL_ID, token=os.environ.get("HF_TOKEN")
        ),
        additional_authorized_imports=QUILL_IMPORTS,
        max_steps=8,
    )


def build_task(csv_path: str, question: str) -> str:
    """Phrase the task so the agent loads the CSV itself with pandas and answers."""
    return (
        f"The CSV is at {csv_path}. Question: {question}\n"
        "Load it with pandas, inspect the columns first, then compute the answer and call "
        "final_answer with a short, specific result."
    )


def main() -> None:
    """CLI: ``uv run python -m quill.agent <csv_path> <question>``.

    Runs Quill, prints the full ReAct trajectory with ``agent.replay()``, then a RunResult
    recap (state, number of steps, token usage) and a per-ActionStep error/observation digest.
    """
    import sys

    from smolagents import ActionStep

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/sales.csv"
    question = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "Which category grew fastest from the first to the last quarter of 2025?"
    )

    agent = build_quill()
    result = agent.run(build_task(csv_path, question), return_full_result=True)

    # Read the trajectory the supported way: replay() + memory.steps (NEVER agent.logs).
    print("\n===== TRAJECTORY (agent.replay) =====")
    agent.replay()

    print("\n===== RUN RESULT =====")
    print(f"state       : {result.state}")
    print(f"steps (dict): {len(result.steps)}")
    print(f"token usage : {result.token_usage}")
    print(f"answer      : {result.output}")

    print("\n===== ACTION STEPS =====")
    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            obs = (step.observations or "").strip().replace("\n", " ")
            obs = (obs[:80] + "...") if len(obs) > 80 else obs
            print(f"  step {step.step_number}: error={type(step.error).__name__ if step.error else None} | obs={obs!r}")


if __name__ == "__main__":
    main()
