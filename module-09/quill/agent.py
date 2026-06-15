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

**Module 7 change:** Quill stops winging it. ``build_quill`` is EXTENDED (never broken) with
two new keyword arguments — both default to the smolagents defaults, so every prior call site
behaves exactly as before:

- ``planning_interval: int | None = None`` — when set (Quill ships ``DEFAULT_PLANNING_INTERVAL
  = 3``), the agent inserts a **``PlanningStep``** at step 1 and then every ``planning_interval``
  steps (trigger ``self.step_number == 1 or (self.step_number - 1) % planning_interval == 0``,
  as of smolagents 1.26.0). A planning step is a tool-free LLM call where Quill (re)states its
  facts and plan, which re-centres it on the goal and cuts redundant exploration — at the cost
  of one extra LLM call, so it is a trade-off worth taking only on longer, multi-step jobs.
- ``instructions: str | None = None`` — when omitted, Quill uses ``QUILL_INSTRUCTIONS``, a
  data-analyst brief that is **APPENDED to the system prompt** (smolagents' own wording:
  "Custom instructions for the agent, will be inserted in the system prompt"). It does NOT
  replace the system prompt — we never touch ``prompt_templates["system_prompt"]``, which the
  docs call "generally not advised" (you would lose the Jinja2 injection of the tool list,
  authorized imports and code-block tags). Pass ``instructions=""`` for the bare default prompt.

The tools' docstrings are sharpened in ``quill/tools/data.py`` (the "write better tools"
principle) — their FROZEN signatures (M3) are unchanged; only the ``description``/``Args:``,
the ``print()`` summaries and the ``ValueError`` messages improved. ``quill/bench.py`` measures
the step drop baseline-vs-improved.

**Module 8 change:** Quill's answer becomes a *contract*. ``build_quill`` is EXTENDED (never
broken) with two more keyword arguments, both defaulting to the smolagents default so every
prior call site is unchanged:

- ``final_answer_checks: list | None`` — when omitted, Quill ships ``quill_final_answer_checks()``
  (``quill/report.py``): two **3-arg** ``(final_answer, memory, agent) -> bool`` validators wired
  into ``CodeAgent(final_answer_checks=...)``. ``check_has_chart`` refuses an answer that is not a
  ``QuillReport`` or has no saved chart; ``check_has_source_for_web_claims`` refuses an empty
  ``sources`` list WHEN the run actually called a web tool. A check that returns ``False`` (or
  raises) does NOT crash: smolagents stores the resulting ``AgentError`` in ``ActionStep.error``
  and loops, so Quill self-corrects on the next step — the same recovery path as a ``KeyError`` in
  its own code. Pass ``final_answer_checks=[]`` to opt out (e.g. the bench's bare baseline).
- ``use_structured_outputs_internally: bool = False`` — the smolagents default (off). This is the
  OPTIONAL extra, off Quill's mandatory path: when ``True``, ``CodeAgent`` loads
  ``structured_code_agent.yaml`` and sets ``response_format`` on every step. It only works with a
  backend that supports ``response_format`` — and Quill's default ``InferenceClientModel`` does so
  ONLY with ``provider in {"cerebras", "fireworks-ai"}`` (``STRUCTURED_GENERATION_PROVIDERS``, as
  of smolagents 1.26.0); otherwise use ``make_model(backend="litellm")`` / an ``OpenAIModel``. It
  reliability-checks the *parsing of each step's output* — it does NOT validate the final answer's
  content (that is ``final_answer_checks``). Form vs validity: keep them separate.

**Module 9 change:** Quill's toolbox stops being a silo. ``build_quill`` is EXTENDED (never
broken) with ONE more keyword argument, defaulting to the empty case so every prior call site
is unchanged:

- ``extra_tools: list | None = None`` — extra ``Tool`` objects to APPEND to Quill's frozen
  toolbox (its order and the local tools are untouched). This is how tools that come from OUTSIDE
  this repo reach the agent: MCP tools (from a stdio/http MCP server), a tool loaded from the
  Hub, a LangChain tool wrapped via ``Tool.from_langchain``. They are added at construction; you
  can also add one at runtime with ``agent.tools[t.name] = t`` (the toolbox is a name-keyed dict).

The new ``run_with_mcp(task)`` connects Quill to a **stdio MCP data server** for the duration of
ONE run: it opens ``ToolCollection.from_mcp(params, trust_remote_code=True, structured_output=
False)`` as a context manager, builds Quill with ``extra_tools=[*tc.tools]``, runs the task, and
the ``with`` exit tears the server subprocess down. ``trust_remote_code=True`` is the interop
security gate (a stdio server executes code locally — M5's threat model); ``structured_output``
is pinned ``False`` explicitly because the smolagents default will flip to ``True`` in a future
release. The MCP plumbing itself lives in ``quill/tools/mcp.py``; this file only wires the tools
into the agent. ``make_model``, the tool signatures, ``QuillReport`` and the sandbox policy are
all untouched — and ``save_chart`` (unchanged) is what M9 PUBLISHES to the Hub
(``quill/scripts/push_save_chart.py``), proof its M3 "pushable" contract held.

This is the canonical Quill construction entry point. Its signature is EXTENDED (never
replaced) by later modules: ``make_model()`` in M4, the executor/import policy in M5,
``step_callbacks`` in M6, ``planning_interval``/``instructions`` in M7,
``final_answer_checks`` in M8, ``extra_tools`` in M9, ``managed_agents`` in M10. The optional
``model=None`` is what lets tests inject an offline fake model and what lets ``make_model()``
own the real default.
"""
from __future__ import annotations

from smolagents import (
    CodeAgent,
    Model,
    Tool,
    VisitWebpageTool,
    WebSearchTool,
)

from .callbacks import quill_callbacks
from .config import DEFAULT_MODEL_ID, make_model
from .report import QuillReport, Source, quill_final_answer_checks
from .sandbox import QUILL_AUTHORIZED_IMPORTS, resolve_executor
from .tools import load_dataset, profile_dataframe, save_chart
from .tools.mcp import data_mcp_server_params

# DEFAULT_MODEL_ID is re-exported from quill.config (the single source of truth as of M4) so
# code and tests that imported it from quill.agent in M3 keep working — same value, one owner.
# QUILL_IMPORTS is re-exported from quill.sandbox (the single source of truth as of M5): it is
# the frozen least-privilege list ["pandas","numpy","matplotlib.*","json","statistics"]. M3/M4
# exposed it from quill.agent, so it stays importable here (a clean superset of M4's list — an
# addition, not a rename), but the OWNER is now sandbox.py so the import lock can never drift.
QUILL_IMPORTS = QUILL_AUTHORIZED_IMPORTS

# Quill's default periodic-planning cadence (M7). With planning_interval set, a `PlanningStep`
# fires at step 1, then every DEFAULT_PLANNING_INTERVAL steps. 3 is the persona's rule of thumb:
# on a 3-4 step job leave planning off; from ~6 steps up, planning_interval=3 usually earns the
# extra LLM call back by stopping the agent re-profiling/reloading and chasing missing columns.
DEFAULT_PLANNING_INTERVAL = 3

# Quill's default `instructions=` (M7). These are APPENDED to the system prompt (smolagents:
# "will be inserted in the system prompt") — they SHARPEN Quill for data analysis WITHOUT
# replacing the prompt. Editing prompt_templates["system_prompt"] is "generally not advised":
# you would drop the Jinja2-injected tool list, authorized imports and code-block tags. The
# headline win is "profile once": the baseline Quill re-loads and re-profiles the same CSV
# across steps, burning LLM calls — these instructions forbid that.
QUILL_INSTRUCTIONS = (
    "You are Quill, a meticulous data analyst. Work in this order:\n"
    "1. Profile the dataset ONCE with profile_dataframe before writing any analysis code. "
    "Never reload or re-profile a file you have already loaded — its DataFrame and your "
    "earlier results persist between code blocks.\n"
    "2. State a short plan, then execute it step by step with pandas/numpy.\n"
    "3. Only use column names you have actually seen in the profile; do not guess columns.\n"
    "4. Back a quantitative claim with a chart: draw it with matplotlib, then call save_chart.\n"
    "5. Finish with final_answer, naming the columns you used and the chart path you saved.\n"
    "Prefer the fewest steps that answer the question correctly."
)

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_PLANNING_INTERVAL",
    "QUILL_IMPORTS",
    "QUILL_INSTRUCTIONS",
    "build_quill",
    "build_task",
    "build_report_task",
    "build_sql_task",
    "run_multi_turn",
    "run_with_mcp",
    "main",
]


# Sentinel so the THREE intents of `instructions=` stay distinguishable (M7):
#   omitted          -> Quill's default QUILL_INSTRUCTIONS (the recommended path)
#   instructions=""  -> the BARE smolagents system prompt (the baseline, for the bench)
#   instructions="…" -> the caller's own brief, appended to the system prompt
# `None` is the library's own default and means "no instructions", which is the same as "" for
# our purposes, so we treat an explicit None like "" and reserve _DEFAULT for "use Quill's".
_DEFAULT = object()

# Sentinel so the TWO intents of `final_answer_checks=` stay distinguishable (M8):
#   omitted                    -> Quill's default checks (chart + web-source) — recommended path
#   final_answer_checks=[...]  -> the caller's own list (use [] to opt OUT, e.g. the bench baseline)
# We reserve _DEFAULT_CHECKS for "use Quill's" so an explicit [] is honoured as "no checks".
_DEFAULT_CHECKS = object()


def build_quill(
    model: Model | None = None,
    *,
    planning_interval: int | None = None,
    instructions: str | None = _DEFAULT,  # type: ignore[assignment]
    final_answer_checks: list | None = _DEFAULT_CHECKS,  # type: ignore[assignment]
    use_structured_outputs_internally: bool = False,
    extra_tools: list[Tool] | None = None,
) -> CodeAgent:
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
    - ``WebSearchTool()`` (engine="duckduckgo" by default, needs ``ddgs`` from ``[toolkit]``)
      and ``VisitWebpageTool()`` for fetching context off the web.

    The returned ``CodeAgent`` supports the context-manager protocol
    (``with build_quill() as agent:``) so a remote sandbox is torn down deterministically;
    fall back to ``agent.cleanup()`` if you cannot use ``with``.

    Args:
        model: a ``smolagents.Model``; if ``None``, ``make_model(role="analyst")`` is used.
        planning_interval (M7): if set, the agent inserts a ``PlanningStep`` at step 1 and then
            every ``planning_interval`` steps (trigger ``step_number == 1 or (step_number - 1) %
            planning_interval == 0``). Default ``None`` = no planning (unchanged from M2-M6).
            ``DEFAULT_PLANNING_INTERVAL`` (3) is Quill's recommended cadence; the bench uses it.
            Each planning step is itself ONE extra LLM call — worth it on long jobs, wasteful on
            short ones.
        instructions (M7): text APPENDED to the system prompt (smolagents: "will be inserted in
            the system prompt"). Omit it to get Quill's default ``QUILL_INSTRUCTIONS`` (the
            recommended path); pass ``""`` (or ``None``) for the BARE smolagents prompt — the
            baseline the bench compares against; pass your own string to specialise Quill. We
            NEVER edit ``prompt_templates["system_prompt"]`` directly ("generally not advised").
        final_answer_checks (M8): a list of 3-arg ``(final_answer, memory, agent) -> bool``
            validators wired into ``CodeAgent(final_answer_checks=...)``. Omit it to get Quill's
            default ``quill_final_answer_checks()`` (chart + web-source); pass your own list to
            customise; pass ``[]`` to opt OUT entirely (the bench's bare baseline). A check that
            returns ``False``/raises does NOT crash the run — smolagents stores the ``AgentError``
            in ``ActionStep.error`` and loops, so Quill self-corrects on the next step.
        use_structured_outputs_internally (M8, OPTIONAL extra): the smolagents default ``False``.
            When ``True``, ``CodeAgent`` loads ``structured_code_agent.yaml`` and sets
            ``response_format`` on every step to make each step's Thought+code parse reliably. It
            only works with a backend that supports ``response_format`` — Quill's default
            ``InferenceClientModel`` does so ONLY with ``provider in {"cerebras","fireworks-ai"}``
            (``STRUCTURED_GENERATION_PROVIDERS``, as of smolagents 1.26.0); use a ``LiteLLMModel``/
            ``OpenAIModel`` otherwise. This is OFF Quill's mandatory path and validates step
            *parsing*, never the final answer's *content* (that is ``final_answer_checks``).
        extra_tools (M9): a list of additional ``Tool`` objects APPENDED to Quill's frozen
            toolbox (the local data tools + web tools stay first and unchanged). This is the seam
            for tools from OUTSIDE this repo — MCP tools (``ToolCollection.from_mcp`` /
            ``MCPClient``), a Hub tool (``load_tool`` / ``Tool.from_hub``), a LangChain tool
            (``Tool.from_langchain``). Default ``None`` = no extras (the M2-M8 toolbox exactly).
            ``run_with_mcp`` passes the MCP server's tools here. You can also add a tool at runtime
            via ``agent.tools[t.name] = t``. NOTE: MCP tools run OUTSIDE the sandbox (server-side
            / in the stdio subprocess); only the Python Quill writes to CALL them runs in the
            sandbox — see ``quill/tools/mcp.py`` for the security gate.

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

    # Resolve the M7 `instructions` intent: omitted -> Quill's default brief; explicit ""/None
    # -> the bare smolagents system prompt (no instructions appended). Whatever we pass goes to
    # MultiStepAgent.__init__(instructions=...), which APPENDS it to the system prompt via the
    # Jinja2 template — we do NOT touch prompt_templates["system_prompt"] (NOTE the commented
    # anti-pattern below). Editing the raw template would drop the injected tool list / imports.
    resolved_instructions = QUILL_INSTRUCTIONS if instructions is _DEFAULT else instructions

    # Resolve the M8 `final_answer_checks` intent: omitted -> Quill's default chart + web-source
    # checks; an explicit list (including []) is honoured verbatim. These are 3-arg validators;
    # a False/raise loops the agent (the error lands in ActionStep.error), it does NOT crash.
    resolved_checks = (
        quill_final_answer_checks()
        if final_answer_checks is _DEFAULT_CHECKS
        else final_answer_checks
    )

    # M9: Quill's frozen local toolbox FIRST, then any ecosystem tools (MCP / Hub / LangChain)
    # the caller supplies via extra_tools. Appending keeps the local tools' names/order stable;
    # the agent's toolbox is ultimately a name-keyed dict, so a duplicate name would override —
    # MCP/Hub tools we attach have distinct names (e.g. read_query), so there is no clash.
    tools = [
        load_dataset,
        profile_dataframe,
        save_chart(),  # Tool subclass: instantiate it (setup() runs lazily on 1st call)
        WebSearchTool(),  # name="web_search"; use exactly ONE web-search tool (06 §6)
        VisitWebpageTool(),  # name="visit_webpage"
    ]
    if extra_tools:
        tools.extend(extra_tools)

    agent = CodeAgent(
        tools=tools,
        # ONE place to choose what powers Quill (M4). No provider knowledge lives here anymore.
        model=model or make_model(role="analyst"),
        # ONE place to choose WHERE the code runs and WHAT it can import (M5, sandbox.py).
        executor_type=executor_type,
        additional_authorized_imports=authorized_imports,
        # Context engineering on memory (M6): prune stale dumps + log cost, after every step.
        step_callbacks=quill_callbacks(),
        # Periodic planning (M7): None = off (M2-M6 behaviour); an int inserts a PlanningStep at
        # step 1 then every N steps. One extra LLM call per plan, so it is a trade-off.
        planning_interval=planning_interval,
        # M7: instructions are APPENDED to the system prompt — they DON'T replace it. We never do
        #   agent.prompt_templates["system_prompt"] = "..."   # <- "generally not advised": this
        #   would drop the Jinja2-injected tool list, authorized imports and code-block tags.
        instructions=resolved_instructions,
        # M8: turn the final answer into a CONTRACT. These 3-arg checks run AFTER smolagents
        # detects the final answer and BEFORE it accepts it; a rejection loops the agent (the
        # AgentError lands in ActionStep.error) so Quill self-corrects. They validate CONTENT —
        # response_format / use_structured_outputs_internally only constrain shape.
        final_answer_checks=resolved_checks,
        # M8 (OPTIONAL extra, off the mandatory path): per-step structured outputs. Default False.
        # When True, CodeAgent loads structured_code_agent.yaml and sets response_format per step —
        # but the default InferenceClientModel only supports response_format with
        # provider in {"cerebras","fireworks-ai"} (STRUCTURED_GENERATION_PROVIDERS, as of 1.26.0),
        # so use make_model(backend="litellm")/OpenAIModel if you turn this on.
        use_structured_outputs_internally=use_structured_outputs_internally,
        max_steps=8,
    )

    # M8: make the report types constructible INSIDE the sandbox without touching the FROZEN
    # import lock (M5). The agent's generated code cannot `import quill.report` — that import is
    # not on the least-privilege allow-list and we will NOT widen it. Instead we inject the two
    # dataclasses as executor variables, so Quill writes `QuillReport(question=..., chart_paths=
    # [path], ...)` directly. This is the "form is guaranteed by the code Quill writes" half of
    # the contract; the `final_answer_checks` above are the "content is validated" half.
    # NOTE (Approach-1 caveat): for a remote executor (docker/e2b) sending live classes has the
    # same serialization limit as Quill's @tool data tools (see the M5/M7 sandbox tests) — the
    # report contract is exercised on the local executor; Approach 2 (M15) runs the whole agent
    # in the sandbox so the types are simply importable there.
    _expose_report_types(agent)
    return agent


def _expose_report_types(agent: CodeAgent) -> None:
    """Inject ``QuillReport``/``Source`` into the agent's executor namespace (M8).

    Lets the agent's sandboxed code build a report (``QuillReport(...)``) without importing
    ``quill.report`` — which the FROZEN least-privilege import lock (M5) deliberately forbids. We
    use the executor's supported ``send_variables`` hook; on the LOCAL executor this just updates
    the interpreter state. We never widen ``additional_authorized_imports`` to do this.

    Approach-1 caveat (06 §5.6): only the LOCAL executor gets the live classes. A remote executor
    (docker/e2b) serializes sent variables with ``allow_pickle=False``, which cannot ship a Python
    class — the same Approach-1 limit that stops Quill's @tool data tools from being sent (see the
    M5/M7 ``sandbox``-marked tests). So the report contract is exercised on the local executor;
    Approach 2 (M15) runs the WHOLE agent in the sandbox, where ``quill.report`` is simply
    importable there. We therefore skip the injection for a non-local executor rather than crash
    its construction.
    """
    if getattr(agent, "executor_type", "local") != "local":
        return
    executor = getattr(agent, "python_executor", None)
    send = getattr(executor, "send_variables", None)
    if send is not None:
        send({"QuillReport": QuillReport, "Source": Source})


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


def build_report_task(csv_path: str, question: str) -> str:
    """Phrase the task so Quill returns a validated ``QuillReport`` (M8).

    Like ``build_task`` but it tells the model to package the answer as a ``QuillReport`` (the
    ``QuillReport`` / ``Source`` classes are already available in the sandbox — ``build_quill``
    injects them, so the agent does NOT import ``quill.report``). The default
    ``final_answer_checks`` then refuse a report with no saved chart, or with web claims but no
    source — so a half-finished report bounces back and Quill self-corrects. Used by the CLI and
    the live test; offline tests script the report code directly.
    """
    return (
        f"The dataset is at {csv_path}. Question: {question}\n"
        "Steps:\n"
        "1. profile_dataframe to learn the schema, then write pandas to compute the answer.\n"
        "2. Draw a matplotlib chart that backs the answer and call save_chart to save it.\n"
        "3. If you use web_search/visit_webpage for any claim, keep each result's URL and title.\n"
        "4. Build a report with the QuillReport class (already available — do NOT import it):\n"
        "     rep = QuillReport(question=<the question>, findings=[...], chart_paths=[<saved "
        "path>], sources=[Source(url=..., title=...)], caveats=[...])\n"
        "   Cite web-backed findings as [1], [2], ... matching the sources list order.\n"
        "5. Call final_answer(rep). A report with no saved chart, or with web claims but no "
        "source, will be REJECTED — fix it and answer again."
    )


def build_sql_task(question: str, table: str = "sales") -> str:
    """Phrase a task that nudges Quill to use the MCP SQL tool (M9).

    The MCP data server exposes a ``read_query`` tool (run a SELECT against ``sales.db``) plus
    ``list_tables`` / ``describe_table``. This task tells Quill those tools exist and asks it to
    answer with a ``QuillReport`` — Quill is free to interleave MCP SQL calls with its own
    pandas/matplotlib (that is the point of a CodeAgent). The MCP tool runs in the server
    subprocess; only the Python Quill writes to call it runs in the sandbox.

    Args:
        question: the analytical question.
        table: the SQLite table the server serves (``sales`` here).

    Returns:
        A task string that names the MCP SQL tools and asks for a validated ``QuillReport``.
    """
    return (
        f"Question: {question}\n"
        f"You have an MCP data server with SQL tools over a SQLite database. Use list_tables / "
        f"describe_table to learn the schema of the '{table}' table, then read_query to run "
        f"SELECTs (e.g. aggregate net_rev by category). Draw a matplotlib chart that backs the "
        f"answer and call save_chart to save it.\n"
        f"Then build a report with the QuillReport class (already available — do NOT import it):\n"
        f"    rep = QuillReport(question=<the question>, findings=[...], "
        f"chart_paths=[<saved path>], sources=[], caveats=[...])\n"
        f"Call final_answer(rep). A report with no saved chart will be REJECTED — fix it and "
        f"answer again."
    )


def run_with_mcp(
    task: str,
    server_parameters=None,
    *,
    model: Model | None = None,
    trust_remote_code: bool = True,
):
    """Run ONE Quill task with a **stdio MCP data server** attached (Module 9 — T9.2).

    Opens ``ToolCollection.from_mcp(server_parameters, trust_remote_code=True,
    structured_output=False)`` as a context manager (it launches the MCP server subprocess in a
    background asyncio thread), builds Quill with the server's tools as ``extra_tools``, runs the
    task, and tears the connection down on ``with`` exit. This is the one-shot path; a
    long-running service would use ``MCPClient`` (connect once, reuse, ``disconnect()`` at
    shutdown) instead — see ``quill/tools/mcp.py``.

    Security gate (T9.5, ties to M5): ``trust_remote_code=True`` is REQUIRED to actually run MCP
    tools, and a **stdio server executes code on your machine** (it is a local subprocess). Set
    it ``True`` only for a server you trust as much as your own code. We expose the flag so a
    caller can be explicit; it defaults to ``True`` because the tools are unusable otherwise (and
    we own the server we launch). ``structured_output`` is pinned ``False`` below ON PURPOSE: the
    smolagents default will flip to ``True`` in a future release, and it is only informational
    anyway (it enriches the prompt with the tool's output schema — it does NOT validate the
    output; Quill's validation stays ``final_answer_checks``).

    Args:
        task: the task string (e.g. from ``build_sql_task``).
        server_parameters: the MCP server to attach; defaults to ``data_mcp_server_params()``
            (the local ``uvx mcp-server-sqlite`` over ``data/sales.db``).
        model: a ``smolagents.Model``; if ``None``, ``make_model()`` is used (a real run).
        trust_remote_code: forwarded to ``ToolCollection.from_mcp`` — the interop security gate.

    Returns:
        The agent's run output (a ``QuillReport`` when the task asks for one).
    """
    # Imported lazily so importing quill.agent does NOT require the [mcp] extra unless you call
    # this. (Building params in quill.tools.mcp is import-time safe; from_mcp is the connection.)
    from smolagents import ToolCollection

    if server_parameters is None:
        server_parameters = data_mcp_server_params()

    # The context manager owns the server subprocess + asyncio thread: it is connected for the
    # body and closed on exit. structured_output=False is pinned explicitly (default will flip to
    # True — 06 §9); trust_remote_code=True is the security gate (stdio runs local code — M5/T9.5).
    with ToolCollection.from_mcp(
        server_parameters,
        trust_remote_code=trust_remote_code,
        structured_output=False,  # default will flip to True in a future release — pin it (T9.6)
    ) as tool_collection:
        with build_quill(model=model, extra_tools=[*tool_collection.tools]) as agent:
            return agent.run(task)


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
    # This demo showcases the M6 multi-turn memory mechanic with the plain ``build_task`` (free-form
    # answers), so we opt OUT of the M8 report contract here (final_answer_checks=[]). The M8 report
    # path is demonstrated by the single-run ``main()`` below and the CLI in ``quill/run.py``.
    agent = agent if agent is not None else build_quill(final_answer_checks=[])

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

    Default (M8): one run that returns a VALIDATED ``QuillReport`` (``build_report_task`` +
    Quill's default ``final_answer_checks``). It prints the full ReAct trajectory
    (``agent.replay()``), a RunResult recap, the rendered report Markdown, and a per-ActionStep
    error/observation digest — where a rejected final answer shows up as an ``AgentError`` in
    ``step.error`` (the self-correction signal) rather than a crash.

    ``--multi-turn``: two questions on one agent (turn 2 is ``reset=False``), then a replay of
    the WHOLE trajectory so you can see exactly what Quill kept and what got pruned. (That demo
    uses the M6 free-form path, so it opts out of the M8 report contract — see ``run_multi_turn``.)
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
        else "Which category grew fastest, and is that consistent with the public trend?"
    )

    # M8: ask for a QuillReport and validate it with the default final_answer_checks. A rejected
    # report loops the agent (the error lands in step.error) so we can OBSERVE the self-correction.
    agent = build_quill()
    result = agent.run(build_report_task(csv_path, question), return_full_result=True)

    # Read the trajectory the supported way: replay() + memory.steps (NEVER agent.logs).
    print("\n===== TRAJECTORY (agent.replay) =====")
    agent.replay()

    print("\n===== RUN RESULT =====")
    print(f"state       : {result.state}")
    print(f"steps (dict): {len(result.steps)}")
    print(f"token usage : {result.token_usage}")

    print("\n===== REPORT (rendered Markdown) =====")
    print(result.output.to_markdown() if isinstance(result.output, QuillReport) else result.output)

    print("\n===== ACTION STEPS (note rejected final answers land in step.error) =====")
    for step in agent.memory.steps:
        if isinstance(step, ActionStep):
            obs = (step.observations or "").strip().replace("\n", " ")
            obs = (obs[:80] + "...") if len(obs) > 80 else obs
            print(f"  step {step.step_number}: error={type(step.error).__name__ if step.error else None} | obs={obs!r}")


if __name__ == "__main__":
    main()
