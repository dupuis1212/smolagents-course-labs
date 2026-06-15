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

**Module 6 change:** Quill goes multi-turn and learns context engineering. ``build_quill``
now passes ``step_callbacks=quill_callbacks()`` (``quill/callbacks.py``): after every step a
callback prunes the big DataFrame dumps from stale observations (so they stop being re-sent to
the model via ``write_memory_to_messages``) and logs per-step token cost. Nothing else
changes — the model factory, tool signatures, ``QuillReport`` and the sandbox policy are
untouched. The new ``--multi-turn`` CLI runs two questions on ONE agent, the second with
``reset=False`` so it keeps the loaded DataFrame and prior findings in memory.

This is the canonical Quill construction entry point. Its signature is EXTENDED (never
replaced) by later modules: ``make_model()`` in M4, the executor/import policy in M5,
``step_callbacks`` in M6, ``planning_interval``/``instructions`` in M7,
``final_answer_checks`` in M8, ``managed_agents`` in M10. The optional ``model=None`` is what
lets tests inject an offline fake model and what lets ``make_model()`` own the real default.
"""
from __future__ import annotations

from smolagents import (
    CodeAgent,
    Model,
    VisitWebpageTool,
    WebSearchTool,
)

from .callbacks import quill_callbacks
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
__all__ = [
    "DEFAULT_MODEL_ID",
    "QUILL_IMPORTS",
    "build_quill",
    "build_task",
    "run_multi_turn",
    "main",
]


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

    Step callbacks (M6): ``step_callbacks=quill_callbacks()`` runs after every step. The list
    form fires on every step type; the callbacks themselves act only on ``ActionStep``. They
    prune stale big observations (``write_memory_to_messages`` then re-sends a small marker
    instead of the dump) and log per-step token cost.
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
        # Context engineering on memory (M6): prune stale dumps + log cost, after every step.
        step_callbacks=quill_callbacks(),
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


def run_multi_turn(
    csv_path: str,
    question_1: str,
    question_2: str,
    agent: CodeAgent | None = None,
) -> CodeAgent:
    """Run two questions on ONE Quill — the second with ``reset=False`` (Module 6).

    The point of the module: a single agent that REMEMBERS. Turn 1 loads + profiles the CSV and
    answers; turn 2 runs ``reset=False`` so the loaded DataFrame, the code Quill wrote and its
    prior findings stay in memory — the model sees them again and does NOT reload/reprofile from
    scratch. Between the two turns the ``quill_callbacks`` (wired in ``build_quill``) have
    already pruned turn 1's big dumps and logged each step's token cost.

    ``reset=False`` keeps memory in RAM inside this agent object — it persists nothing to disk
    (that is Module 13). Kill the process and the memory is gone.

    Args:
        csv_path: the dataset both turns analyse (same file, no reload on turn 2).
        question_1: the first turn's question (run with the default ``reset=True``).
        question_2: the follow-up (run with ``reset=False`` so it continues the conversation).
        agent: an existing agent to reuse (tests inject a fake-model one); otherwise built here.

    Returns:
        The agent, so the caller can inspect ``agent.memory.steps`` / ``agent.replay()`` after.
    """
    agent = agent if agent is not None else build_quill()

    print("\n===== TURN 1 (reset=True, the default) =====")
    answer_1 = agent.run(build_task(csv_path, question_1))
    print(f"\nTurn 1 answer: {answer_1}")
    steps_after_turn_1 = len(agent.memory.steps)

    print("\n===== TURN 2 (reset=False — keep turn 1 in memory) =====")
    # reset=False keeps the loaded DataFrame and prior findings in memory: Quill does NOT
    # reload or re-profile the CSV, it continues from where turn 1 left off.
    answer_2 = agent.run(build_task(csv_path, question_2), reset=False)
    print(f"\nTurn 2 answer: {answer_2}")

    print("\n===== MEMORY GREW (reset=False did not wipe turn 1) =====")
    print(f"steps after turn 1     : {steps_after_turn_1}")
    print(f"steps after turn 2     : {len(agent.memory.steps)}  (turn 1's steps are still here)")
    return agent


def main() -> None:
    """CLI: ``uv run python -m quill.agent [--multi-turn] <csv_path> [<question>]``.

    Default: one run, then the full ReAct trajectory (``agent.replay()``), a RunResult recap,
    and a per-ActionStep error/observation digest (note how the old big observation now shows
    the prune marker, courtesy of the step callback).

    ``--multi-turn``: two questions on one agent (turn 2 is ``reset=False``), then a replay of
    the WHOLE trajectory so you can see exactly what Quill kept and what got pruned.
    """
    import sys

    from smolagents import ActionStep

    args = sys.argv[1:]
    multi_turn = "--multi-turn" in args
    positional = [a for a in args if not a.startswith("-")]

    csv_path = positional[0] if positional else "data/sales.csv"

    if multi_turn:
        question_1 = (
            positional[1]
            if len(positional) > 1
            else "Which category grew fastest from the first to the last quarter of 2025?"
        )
        question_2 = (
            positional[2]
            if len(positional) > 2
            else "Now answer the same question but exclude any rows from 2020."
        )
        agent = run_multi_turn(csv_path, question_1, question_2)
        print("\n===== FULL TRAJECTORY (agent.replay) =====")
        agent.replay()
        return

    question = (
        positional[1]
        if len(positional) > 1
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
