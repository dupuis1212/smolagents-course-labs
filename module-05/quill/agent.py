"""Quill's construction entry point (toolbox M3, model layer M4, sandbox policy M5).

`build_quill()` returns a ``CodeAgent`` wired with Quill's first reusable, frozen tools —
``load_dataset``, ``profile_dataframe``, ``save_chart`` — plus web access via
``WebSearchTool`` and ``VisitWebpageTool``. Given a CSV and a question, Quill CALLS these
tools (load → profile → write pandas → save_chart) instead of re-deriving everything in
throwaway code, and answers via ``final_answer``.

**Module 4 change:** the model is no longer built here. ``build_quill`` calls
``make_model(role="analyst")`` (``quill/config.py``) when no model is passed in, so there is
ONE place to swap the backend (env-driven) and nothing in this file knows about a provider.

**Module 5 change:** the executor and the import lock are no longer decided here either.
``build_quill`` calls ``resolve_executor()`` (``quill/sandbox.py``, the FROZEN sandbox
contract) to get ``(executor_type, additional_authorized_imports)`` from ``QUILL_EXECUTOR``.
So ``QUILL_EXECUTOR=docker`` runs Quill's generated Python inside a Docker container
(Approach 1: snippet-in-sandbox), with the imports locked to Quill's least-privilege list —
never the ``"*"`` wildcard. The returned agent is a context manager: use
``with build_quill() as agent:`` so the sandbox is torn down deterministically (no dangling
containers).

This is the canonical Quill construction entry point. Its signature is EXTENDED (never
replaced) by later modules: ``make_model()`` in M4, the executor/import policy in M5,
``planning_interval``/``instructions`` in M7, ``final_answer_checks`` in M8,
``managed_agents`` in M10. The optional ``model=None`` is what lets tests inject an offline
fake model and what lets ``make_model()`` own the real default.
"""
from __future__ import annotations

from smolagents import (
    CodeAgent,
    Model,
    VisitWebpageTool,
    WebSearchTool,
)

from .config import DEFAULT_MODEL_ID, make_model
from .sandbox import QUILL_AUTHORIZED_IMPORTS, resolve_executor
from .tools import load_dataset, profile_dataframe, save_chart

# DEFAULT_MODEL_ID is re-exported from quill.config (the single source of truth as of M4) so
# code and tests that imported it from quill.agent in M3 keep working — same value, one owner.
# QUILL_IMPORTS is re-exported from quill.sandbox (the single source of truth as of M5): it is
# the frozen least-privilege list ["pandas","numpy","matplotlib.*","json","statistics"]. M3/M4
# exposed it from quill.agent, so it stays importable here (a clean superset of M4's list — an
# addition, not a rename), but the OWNER is now sandbox.py so the import lock can never drift.
QUILL_IMPORTS = QUILL_AUTHORIZED_IMPORTS
__all__ = ["DEFAULT_MODEL_ID", "QUILL_IMPORTS", "build_quill", "build_task", "main"]


def build_quill(model: Model | None = None) -> CodeAgent:
    """Build Quill: a ``CodeAgent`` with a data toolbox, web access, and a sandbox policy.

    Pass your own ``model`` to run offline (e.g. a fake model in tests). Otherwise the model
    comes from ``make_model(role="analyst")`` — the single, env-driven model factory in
    ``quill/config.py`` (M4 frozen contract). This file no longer instantiates a model class
    directly: swap the backend with ``QUILL_MODEL_BACKEND``/``QUILL_MODEL_ID``, not by editing
    the agent.

    The executor and import lock come from ``resolve_executor()`` (``quill/sandbox.py``, the
    M5 frozen contract): ``QUILL_EXECUTOR`` in {local, docker, e2b} picks WHERE the generated
    Python runs, and ``additional_authorized_imports`` is locked to Quill's least-privilege
    list — never ``"*"``. This is the ONLY place Quill's isolation is decided.

    The toolbox:
    - ``load_dataset`` / ``profile_dataframe`` (``@tool`` functions) and ``save_chart``
      (a ``Tool`` subclass — note ``save_chart()``: it is instantiated here).
    - ``WebSearchTool()`` (engine="duckduckgo" by default; fetches via ``requests`` — it does
      NOT use ``ddgs``; that package is only for the separate ``DuckDuckGoSearchTool``)
      and ``VisitWebpageTool()`` for fetching context off the web.

    The returned ``CodeAgent`` supports the context-manager protocol
    (``with build_quill() as agent:``) so a remote sandbox is torn down deterministically;
    fall back to ``agent.cleanup()`` if you cannot use ``with``.

    Deliberate non-knobs:
    - We do NOT pass ``add_base_tools=True``. ``FinalAnswerTool`` is already added for every
      agent (it is how a run terminates), and for a ``CodeAgent`` the ``python_interpreter``
      tool is excluded anyway (the agent already runs Python). So it would add nothing here.
    - ``additional_authorized_imports`` stays a minimal, explicit list — never ``"*"``.
    - We do NOT pass ``managed_agents`` (multi-agents is Module 10). A remote executor +
      ``managed_agents`` would raise — that combination needs Approach 2 (Module 15).
    - ``max_steps=8`` — the library default is 20; a single-CSV job needs far fewer.
    """
    executor_type, authorized_imports = resolve_executor()
    return CodeAgent(
        tools=[
            load_dataset,
            profile_dataframe,
            save_chart(),  # Tool subclass: instantiate it (setup() runs lazily on 1st call)
            WebSearchTool(),  # name="web_search"; use exactly ONE web-search tool (06 §6)
            VisitWebpageTool(),  # name="visit_webpage"
        ],
        # ONE place to choose what powers Quill (M4). No provider knowledge lives here anymore.
        model=model or make_model(role="analyst"),
        # ONE place to choose WHERE the code runs and WHAT it can import (M5, sandbox.py).
        executor_type=executor_type,
        additional_authorized_imports=authorized_imports,
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
