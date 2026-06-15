"""Quill's research team — managed sub-agents (Module 10, the multi-agent module).

Until Module 9, Quill was ONE ``CodeAgent`` ("one brain") that planned, wrote the pandas,
saved charts, AND went to the web itself (it owned ``WebSearchTool`` + ``VisitWebpageTool``).
Every page it visited landed *whole* in its own memory — the doc's own complaint:
"why fill the memory of the code-generating agent with all the content of the webpages
visited?". As of Module 10, Quill becomes a **manager** ``CodeAgent`` over a specialised
sub-agent, the **``web_researcher``**. The manager plans and analyses; the sub-agent fetches
and summarises web context and returns only a short summary with source URLs. The manager's
memory stays clean — that is **context isolation**, the real reason to reach for multi-agent
(NOT "more agents = better results").

**The current way to wire a team (smolagents 1.26.0).** There is no ``ManagedAgent`` — it was
deprecated in 1.8.0, dropped from the docs in 1.21.0, and is ABSENT from ``agents.py`` in
1.26.0. Any tutorial that does ``from smolagents import ManagedAgent`` is dead. The current
mechanism: ANY agent (``CodeAgent`` or ``ToolCallingAgent``) becomes callable by a manager the
moment you give it a **``name``** and a **``description``** ("to make this agent callable by
its manager"), then pass it to the manager via ``managed_agents=[...]`` (a ``MultiStepAgent``
argument, inherited by both agent types). ``build_quill`` (``quill/agent.py``) does exactly
that with the ``web_researcher`` this module builds.

**Why the worker is a ``ToolCallingAgent`` and the manager a ``CodeAgent``.** The decision
matrix from Module 3: a ``CodeAgent`` is a "problem solver" — it composes Python, ideal for
planning + arithmetic = the analyst manager (Quill). A ``ToolCallingAgent`` is a "dispatcher" —
it emits JSON tool calls validated against a schema, a single timeline = stepping through the
web one page at a time. Web navigation does not need to compose Python; the JSON tool-call is
safer and sufficient. (A ``ToolCallingAgent`` can fan out parallel calls via ``max_tool_threads``
/ a ``ThreadPoolExecutor`` — a real capability, but we do NOT use it here: the researcher is a
single timeline.)

**How the manager calls the sub-agent (the internals).** For a ``CodeAgent`` manager, smolagents
exposes each managed agent INSIDE the Python sandbox as a callable function whose signature is
auto-generated from the system-prompt template::

    def web_researcher(task: str, additional_args: dict[str, Any]) -> str:
        '''<the sub-agent's description>
        Args:
            task: Long detailed description of the task.
            additional_args: Dictionary of extra inputs (e.g. images, dataframes).
        '''

So Quill literally writes ``summary = web_researcher("Find the SaaS industry average churn")``
in its generated code, like calling any tool, and the run prints
"Here is the final answer from your managed agent 'web_researcher': ...". The
``additional_args`` channel is how rich objects (images, DataFrames) flow DOWN to a sub-agent —
the hook Module 11 reuses for vision. (A ``ToolCallingAgent`` manager would instead dispatch via
``execute_tool_call(tool_name, arguments)`` and list the team as
``- {{ agent.name }}: {{ agent.description }}`` in its prompt.)

**The constraint (Module 5 / T10.7).** A remote ``executor_type`` (docker/e2b) PLUS
``managed_agents`` raises ``Exception("Managed agents are not yet supported with remote code
execution.")`` — ``create_python_executor`` refuses the combination. The reason: in Approach 1
(snippet-in-sandbox) the model stays local and secrets (the HF token) are NOT shipped into the
sandbox, so a sub-agent could not authenticate its own LLM calls from inside. So Quill's team
runs in ``executor_type="local"`` here; running the WHOLE team *inside* a sandbox (Approach 2)
is the capstone, Module 15.

What this module does NOT add: no ``vision_browser`` sub-agent / ``run(images=...)`` (Module 11),
no Approach 2 (Module 15), no nested sub-agents-of-sub-agents (mentioned only). Frozen contracts
are untouched: the model ALWAYS comes from ``make_model`` (M4), and the run still ends in a
cited ``QuillReport`` (M8).
"""
from __future__ import annotations

from smolagents import (
    Model,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)

from .config import make_model

# The canonical sub-agent name (06-FIL-ROUGE-SPEC §2). It MUST be exactly "web_researcher"
# (not "web_agent"/"researcher"/"WebAgent"): the manager calls it by this name, the M11
# diagram extends this team, and the tests pin it. `name` + `description` together are what
# make an agent callable by a manager (passed via managed_agents=[...] in build_quill).
WEB_RESEARCHER_NAME = "web_researcher"

# The description the manager reads to decide WHEN and HOW to delegate. For a CodeAgent manager
# this text becomes the docstring of the auto-generated `web_researcher(task, additional_args)`
# function in the sandbox, so it must read like a tool's docstring: what it does, what to give
# it, what it returns. Keep it focused — a vague description makes the manager mis-delegate.
WEB_RESEARCHER_DESCRIPTION = (
    "Searches the web and visits pages to fetch missing external context (industry "
    "benchmarks, market averages, public trends, definitions). Give it ONE focused "
    "question as the task; it returns a short text summary with the source URLs it used. "
    "It does NOT see your dataset — pass any numbers it needs inside the question. Use it "
    "only when the answer is not in the local data."
)

# The researcher's step budget (06 §6, T7 production note). A sub-agent runs its OWN ReAct loop
# up to max_steps, so each delegation can cost several LLM calls — a multi-agent run can triple
# the calls of a single agent. 10 is enough to search, visit a couple of pages and summarise,
# while the HF free tier ($0.10/month as of smolagents 1.26.0) stays a real wall. Lower it (e.g.
# to 3) and a deep search returns "empty-handed", which the manager then has to handle.
WEB_RESEARCHER_MAX_STEPS = 10


def build_web_researcher(
    model: Model | None = None,
    *,
    max_steps: int = WEB_RESEARCHER_MAX_STEPS,
    provide_run_summary: bool = False,
) -> ToolCallingAgent:
    """Build the ``web_researcher`` — Quill's managed web-research sub-agent (Module 10).

    A ``ToolCallingAgent`` (a "dispatcher": JSON tool calls, single timeline) tooled with
    ``WebSearchTool`` + ``VisitWebpageTool`` and capped at ``max_steps``. It carries the
    canonical ``name="web_researcher"`` and a focused ``description`` — the two attributes that
    make it **callable by a manager** when passed to ``managed_agents=[...]`` (the current
    mechanism; ``ManagedAgent`` is gone). ``build_quill`` (``quill/agent.py``) constructs this and
    registers it on the manager.

    The model ALWAYS comes from ``make_model`` (the M4 frozen contract — ONE place to swap the
    backend); pass your own ``model`` to run offline (a fake model in tests). ``role="researcher"``
    is forwarded so a later module could give the researcher a cheaper/faster model than the
    analyst manager without touching any call site — in M10 every role uses the same default.

    Args:
        model: a ``smolagents.Model``; if ``None``, ``make_model(role="researcher")`` is used.
        max_steps: the sub-agent's own ReAct step budget (default ``WEB_RESEARCHER_MAX_STEPS`` =
            10). Each step is an LLM call, so this is the per-delegation cost ceiling. Lower it to
            see the "empty-handed" failure mode surface back to the manager.
        provide_run_summary: when ``True``, the manager sees the sub-agent's full reasoning trace,
            not only its final answer (smolagents' ``provide_run_summary`` knob). Default ``False``
            (the manager gets only the summary string) — the "Try it yourself" toggle in the lab.

    Returns:
        A ``ToolCallingAgent`` named ``web_researcher``, ready to drop into ``managed_agents=[...]``.
    """
    return ToolCallingAgent(
        # The dispatcher's toolset: search the web, then read a page. Exactly ONE web-search tool
        # (web_search), per 06 §6. The FinalAnswerTool is added automatically (it is how the
        # sub-agent returns its summary). We do NOT pass add_base_tools, so the toolset stays
        # focused: exactly {web_search, visit_webpage, final_answer} — no python_interpreter.
        tools=[WebSearchTool(), VisitWebpageTool()],
        # M4 frozen contract: the model comes from make_model — never an InferenceClientModel built
        # here, never HfApiModel. role="researcher" is forwarded for a future per-role swap.
        model=model or make_model(role="researcher"),
        # These TWO attributes are what make this agent callable by a manager (06 §2). The manager
        # calls it by this exact name; the description becomes the delegated function's docstring.
        name=WEB_RESEARCHER_NAME,
        description=WEB_RESEARCHER_DESCRIPTION,
        # The sub-agent's own ReAct budget — the per-delegation cost ceiling (T7 production note).
        max_steps=max_steps,
        # Optional knob (off by default): surface the sub-agent's reasoning to the manager, not
        # just its final answer. The lab's "Try it yourself" flips this to True to compare.
        provide_run_summary=provide_run_summary,
    )


__all__ = [
    "WEB_RESEARCHER_NAME",
    "WEB_RESEARCHER_DESCRIPTION",
    "WEB_RESEARCHER_MAX_STEPS",
    "build_web_researcher",
]
