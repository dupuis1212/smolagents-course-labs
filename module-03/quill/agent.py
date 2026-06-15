"""Quill gains a toolbox (Module 3).

`build_quill()` returns a ``CodeAgent`` wired with Quill's first reusable, frozen tools —
``load_dataset``, ``profile_dataframe``, ``save_chart`` — plus web access via
``WebSearchTool`` and ``VisitWebpageTool``. Given a CSV and a question, Quill now CALLS
these tools (load → profile → write pandas → save_chart) instead of re-deriving everything
in throwaway code, and answers via ``final_answer``.

This is the canonical Quill construction entry point. Its signature is EXTENDED (never
replaced) by later modules: ``make_model()`` in M4, ``executor_type`` in M5,
``planning_interval``/``instructions`` in M7, ``final_answer_checks`` in M8,
``managed_agents`` in M10. Accepting an optional ``model=None`` now keeps the swap to
``make_model()`` (M4) painless.
"""
from __future__ import annotations

import os

from smolagents import (
    CodeAgent,
    InferenceClientModel,
    Model,
    VisitWebpageTool,
    WebSearchTool,
)

from .tools import load_dataset, profile_dataframe, save_chart

# Explicit, code-capable model_id. We do NOT rely on InferenceClientModel's default model_id:
# it is documented as "subject to change" (as of smolagents 1.26.0), and a data/code agent
# wants a coder/instruct model. Module 4 turns this into make_model() in quill/config.py.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"

# The sandbox does NOT authorize these by default — Quill must declare them. This list is
# also a *security* boundary (the threat model, the AST checks, and why you never use the
# "*" wildcard are Module 5). We add "matplotlib.*" this module so the agent can draw a
# figure BEFORE handing it to save_chart. This is a clean SUPERSET on the way to M5's frozen
# list (["pandas","numpy","matplotlib.*","json","statistics"]) — an addition, not a rename.
QUILL_IMPORTS = ["pandas", "numpy", "matplotlib.*"]


def build_quill(model: Model | None = None) -> CodeAgent:
    """Build Quill: a ``CodeAgent`` with a data toolbox and web access.

    Pass your own ``model`` to run offline (e.g. a fake model in tests); otherwise a hosted
    Hugging Face model (``InferenceClientModel`` with an explicit ``model_id``) is used.

    The toolbox:
    - ``load_dataset`` / ``profile_dataframe`` (``@tool`` functions) and ``save_chart``
      (a ``Tool`` subclass — note ``save_chart()``: it is instantiated here).
    - ``WebSearchTool()`` (engine="duckduckgo" by default; fetches via ``requests`` — it does
      NOT use ``ddgs``; that package is only for the separate ``DuckDuckGoSearchTool``)
      and ``VisitWebpageTool()`` for fetching context off the web.

    Deliberate non-knobs:
    - We do NOT pass ``add_base_tools=True``. ``FinalAnswerTool`` is already added for every
      agent (it is how a run terminates), and for a ``CodeAgent`` the ``python_interpreter``
      tool is excluded anyway (the agent already runs Python). So it would add nothing here.
    - ``additional_authorized_imports`` stays a minimal, explicit list — never ``"*"``.
    - ``max_steps=8`` — the library default is 20; a single-CSV job needs far fewer.
    """
    return CodeAgent(
        tools=[
            load_dataset,
            profile_dataframe,
            save_chart(),  # Tool subclass: instantiate it (setup() runs lazily on 1st call)
            WebSearchTool(),  # name="web_search"; use exactly ONE web-search tool (06 §6)
            VisitWebpageTool(),  # name="visit_webpage"
        ],
        model=model or InferenceClientModel(
            model_id=DEFAULT_MODEL_ID, token=os.environ.get("HF_TOKEN")
        ),
        additional_authorized_imports=QUILL_IMPORTS,
        max_steps=8,
    )


def build_task(csv_path: str, question: str) -> str:
    """Phrase the task so the agent uses its tools, draws a chart, and answers.

    The task NAMES the tools so the model knows the toolbox exists, but Quill is free to
    write its own pandas/matplotlib in between (that is the point of a CodeAgent).
    """
    return (
        f"The dataset is at {csv_path}. Question: {question}\n"
        "Start with profile_dataframe to learn the schema, then write pandas to compute "
        "the answer. Draw a matplotlib chart that backs it up, call save_chart to save the "
        "figure, then call final_answer with a short result that includes the saved "
        "chart path."
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
