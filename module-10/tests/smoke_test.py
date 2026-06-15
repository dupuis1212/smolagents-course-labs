"""Module 10 smoke tests — multi-agent systems: Quill gets a research team.

Live call budget: a manager + sub-agent run can make 5-15 LLM calls (the manager's ReAct loop
PLUS the web_researcher's own ReAct loop, capped at max_steps=10). The NEW M10 live test does
ONE such team run; with the carried-forward live tests the whole file still stays a small,
documented live budget — and EVERY live test is skipped unless QUILL_LIVE_TESTS=1 and HF_TOKEN
is set. Sandbox budget: 1-2 Docker runs (marked `sandbox`, skipped if Docker is absent).

Everything else runs OFFLINE with no network and no token. The M10 multi-agent tests build BOTH
the manager (a CodeAgent with a fake_model) AND the web_researcher sub-agent (a ToolCallingAgent
with a fake tool-call model that emits a `final_answer` tool call), so the WHOLE team loop runs
deterministically with zero LLM calls — proving the wiring (registration, delegation, the cited
report) without paying for or depending on a model.

Module 10 (NEW) — quill/team.py + quill/agent.py:
- `build_web_researcher(model)` returns a ToolCallingAgent named EXACTLY "web_researcher" with
  WebSearchTool + VisitWebpageTool and max_steps=10; name+description make it manager-callable;
- there is NO `ManagedAgent` in smolagents 1.26.0 (the headline freshness ban) — a sub-agent is
  managed purely by name+description + passing it via managed_agents=[...];
- `build_quill` is EXTENDED by ADD with `managed_agents=` (keyword-only): omitted -> Quill's
  DEFAULT team (just the web_researcher); [] -> a SOLO manager; a list -> the caller's own team.
  The M2-M9 call sites are unbroken;
- the web tools LEFT the manager's own toolbox (context isolation) and now live on the sub-agent;
- the manager (a CodeAgent) registers the sub-agent in agent.managed_agents (a name-keyed dict)
  and reaches it by writing `web_researcher("...")` in its sandboxed code — an OFFLINE end-to-end
  run delegates to the (fake-model) sub-agent and finishes in a cited QuillReport;
- the constraint (T10.7): a remote executor + managed_agents raises — Quill stays local in M10.

This file ALSO carries the Module 2/3/4/5/6/7/8/9 tests forward (the toolbox, the agent loop,
make_model(), the Monitor cost accessor, the sandbox policy, the callbacks/multi-turn, the
planning/instructions, the QuillReport + final_answer_checks contract, and the MCP interop still
work). Two carried-forward assertions were UPDATED for the M10 reality (not deleted): the
manager's toolbox no longer contains web_search/visit_webpage (they moved to the sub-agent), and
the M8 web-source check is now exercised through a `web_researcher(...)` delegation.

Module 9 (carried forward) — quill/tools/mcp.py + quill/agent.py + quill/scripts/* + quill/demos/*:
- the MCP entry points import: `ToolCollection` and `MCPClient` are importable from smolagents,
  and `Tool.from_mcp` does NOT exist (the freshness trap — MCP lives on ToolCollection/MCPClient);
- `data_mcp_server_params()` builds a stdio `StdioServerParameters` (command="uvx",
  args=[..., "--db-path", "data/sales.db"], env has UV_PYTHON) WITHOUT starting a subprocess;
  `http_server_params()` builds the streamable-http dict (default transport), and the deprecated
  `sse` is opt-in only; `describe_server_params` is pure (no connection);
- `build_quill` is EXTENDED by ADD with `extra_tools=` (keyword-only, default None): extra tools
  are APPENDED after the frozen local toolbox, reachable by name, and the M2-M8 call sites are
  unbroken; a runtime `agent.tools[name] = t` still works;
- `run_with_mcp` exists with the right signature; `build_sql_task` mentions the MCP SQL tools;
- the Hub scripts: `load_hub_tool`/`attach_to_quill` have the right shape (no network at import);
  `push_save_chart` proves the pushable rules — `save_chart.save(dir)` writes save_chart.py +
  app.py + requirements.txt OFFLINE (a clean save IS the proof the M3 contract held);
- a `live`-marked test connects to a REAL stdio MCP server (skipped by default).

This file ALSO carries the Module 2/3/4/5/6/7/8 tests forward (the toolbox, the agent loop,
make_model(), the Monitor cost accessor, the sandbox policy, the callbacks/multi-turn, the
planning/instructions, and the QuillReport + final_answer_checks contract still work — the
carried-forward agent-loop tests pass final_answer_checks=[] to opt OUT of the report contract,
since they assert pre-M8 mechanics). The Module 8 tests kept here:
- QuillReport / Source are the FROZEN M8 schema (06 §2): exactly five / two fields, no more;
- QuillReport.to_markdown() renders findings with numbered [n] citations into sources (a one-
  source report contains the marker [1] and "[1] [title](url)" in its Sources section);
- the two final_answer_checks are 3-arg (final_answer, memory, agent): check_has_chart rejects
  a non-QuillReport and an empty chart_paths; check_has_source_for_web_claims rejects an empty
  sources list ONLY when the run actually called a web tool (heuristic on the memory's code);
- build_quill is EXTENDED by ADD (final_answer_checks, use_structured_outputs_internally are
  keyword-only, both default to the smolagents default) — the M2-M7 call sites are unbroken;
- build_quill wires Quill's default checks into agent.final_answer_checks and exposes
  QuillReport/Source to the sandbox (so the agent builds a report without widening the FROZEN
  import lock); a fake-model agent that builds a QuillReport and calls final_answer(report)
  PASSES, while one that answers with a bare string is REJECTED and the AgentError lands in
  ActionStep.error (the self-correction loop, NOT a crash);
- the self-correction demo: a fake model that first answers without a chart (rejected) then
  draws + saves a chart and returns a complete QuillReport (accepted) on the next step.

Run from the repo root: ``uv run pytest module-10/tests/``
"""
from __future__ import annotations

import dataclasses
import inspect
import os
import pathlib
import sys

import pytest
from smolagents import (
    ActionStep,
    AgentError,
    CodeAgent,
    InferenceClientModel,
    LiteLLMModel,
    Model,
    Monitor,
    PlanningStep,
    RunResult,
    TokenUsage,
    Tool,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
)
from smolagents.local_python_executor import InterpreterError
from smolagents.memory import Timing
from smolagents.models import (
    ChatMessage,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
)

# Make THIS module's `quill` package importable when running from the repo root, even in the
# cumulative suite where earlier modules also ship a `quill` package. Every module-NN/ is a
# self-contained snapshot, so several dirs define a top-level `quill`; whichever is imported
# first would otherwise win in sys.modules. We prepend this module's dir and drop any cached
# `quill*` so this file always binds to module-10/quill.
MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]
for _name in [n for n in list(sys.modules) if n == "quill" or n.startswith("quill.")]:
    del sys.modules[_name]
sys.path.insert(0, str(MODULE_DIR))
from quill.agent import (  # noqa: E402
    DEFAULT_MODEL_ID,
    DEFAULT_PLANNING_INTERVAL,
    QUILL_IMPORTS,
    QUILL_INSTRUCTIONS,
    build_quill,
    build_sql_task,
    build_task,
    run_multi_turn,
    run_with_mcp,
)
from quill.team import (  # noqa: E402
    WEB_RESEARCHER_DESCRIPTION,
    WEB_RESEARCHER_MAX_STEPS,
    WEB_RESEARCHER_NAME,
    build_web_researcher,
)
from quill.report import (  # noqa: E402
    QUILL_FINAL_ANSWER_CHECKS,
    QuillReport,
    Source,
    check_has_chart,
    check_has_source_for_web_claims,
    quill_final_answer_checks,
)
from quill.bench import (  # noqa: E402
    StepCount,
    count_steps,
    format_report,
    run_and_count,
)
from quill.callbacks import (  # noqa: E402
    KEEP_LAST,
    MAX_OBS_CHARS,
    PRUNE_MARKER,
    log_step_cost,
    prune_old_observations,
    quill_callbacks,
)
from quill.config import (  # noqa: E402
    DEFAULT_LOCAL_MODEL_ID,
    OLLAMA_NUM_CTX,
    Settings,
    make_model,
)
from quill.run import _format_cost  # noqa: E402
from quill.run import main as run_main  # noqa: E402
from quill.sandbox import (  # noqa: E402
    DEFAULT_EXECUTOR,
    QUILL_AUTHORIZED_IMPORTS,
    SUPPORTED_EXECUTORS,
    resolve_executor,
)
from quill.tools import load_dataset, profile_dataframe, save_chart  # noqa: E402
from quill.tools.mcp import (  # noqa: E402
    DEFAULT_MCP_SQLITE_PACKAGE,
    DEFAULT_SQLITE_DB_PATH,
    MCPClient,
    StdioServerParameters,
    ToolCollection,
    data_mcp_server_params,
    describe_server_params,
    http_server_params,
)
from quill.scripts.build_sales_db import build_sales_db  # noqa: E402
from quill.scripts.load_hub_tool import (  # noqa: E402
    DEFAULT_HUB_TOOL_REPO,
    attach_to_quill,
    load_hub_tool,
)
from quill.scripts.push_save_chart import save_save_chart_locally  # noqa: E402

CSV = str(MODULE_DIR / "data" / "sales.csv")
# M10 seeds data/customers.csv (a NEW entry, not a rename — data/sales.csv stays the fil-rouge
# dataset, 06 §5.4). It carries a `churned` column, which the M10 churn-vs-industry demo uses.
CUSTOMERS = str(MODULE_DIR / "data" / "customers.csv")
OUTPUTS = MODULE_DIR / "outputs"
SALES_DB = str(MODULE_DIR / "data" / "sales.db")


def _plan(text: str = "Facts: a CSV. Plan: profile once, then aggregate.") -> str:
    """A scripted planning-step output for the FakeModel — bounded by the <end_plan> stop token
    the agent uses (smolagents 1.26.0). The agent strips at <end_plan>, so the plan text is
    self-contained."""
    return f"{text}\n<end_plan>"


def _load(path: str) -> str:
    return f'import pandas as pd\ndf = pd.read_csv({path!r})'


def _action_step(step_number: int, observations: str | None = None,
                 token_usage: TokenUsage | None = None) -> ActionStep:
    """Build an ActionStep by hand for the offline callback tests (no agent, no LLM)."""
    return ActionStep(
        step_number=step_number,
        timing=Timing(start_time=0.0),
        observations=observations,
        token_usage=token_usage,
    )


class _FakeAgentMemory:
    """The tiny slice of an agent a step_callback touches: just ``memory.steps``."""

    def __init__(self, steps):
        self.steps = steps


class _FakeAgent:
    """A stand-in agent exposing ``.memory.steps`` so prune_old_observations can mutate it
    OFFLINE (no model, no run). The callbacks only ever read agent.memory.steps."""

    def __init__(self, steps):
        self.memory = _FakeAgentMemory(steps)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a clean model/executor environment so order never matters."""
    monkeypatch.delenv("QUILL_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("QUILL_MODEL_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("QUILL_EXECUTOR", raising=False)


# Reusable fixtures of a complete and an empty report for the M8 tests.
def _complete_report() -> QuillReport:
    """A fully-populated QuillReport: a question, a finding, a saved chart, a source, a caveat."""
    return QuillReport(
        question="Which category grew fastest in 2025?",
        findings=["Team grew fastest, +38% from Q1 to Q4 [1]."],
        chart_paths=["outputs/category_growth.png"],
        sources=[Source(url="https://example.com/saas-2025", title="SaaS Trends 2025")],
        caveats=["Q4 is a partial quarter."],
    )


class _StubMemory:
    """A stand-in AgentMemory exposing only ``steps`` — what the source check reads."""

    def __init__(self, steps=None):
        self.steps = steps or []


class _StubStep:
    """A stand-in ActionStep carrying just ``code_action`` (and optional tool_calls)."""

    def __init__(self, code_action=None, tool_calls=None):
        self.code_action = code_action
        self.tool_calls = tool_calls


# ======================================================================================
# Module 10 (NEW): multi-agent systems — quill/team.py + quill/agent.py.
# OFFLINE: the manager is a fake-model CodeAgent; the web_researcher sub-agent runs on a fake
# TOOL-CALL model (it emits a `final_answer` tool call so a ToolCallingAgent terminates with no
# network). So the WHOLE team loop runs deterministically with ZERO LLM calls. NEVER ManagedAgent.
# ======================================================================================


class _FakeToolCallModel(Model):
    """A deterministic offline model for a ToolCallingAgent (the web_researcher sub-agent).

    A ToolCallingAgent terminates by EMITTING a `final_answer` tool call (not by returning text,
    the way a CodeAgent does). So this model returns a ChatMessage whose tool_calls carry a single
    `final_answer(answer=...)` call — letting the sub-agent finish offline, with no LLM, no network.
    It is the sub-agent counterpart of the repo-wide CodeAgent fake_model fixture.
    """

    def __init__(self, answer: str, model_id: str = "fake/toolcall"):
        super().__init__(model_id=model_id)
        self.answer = answer
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs) -> ChatMessage:
        self.calls += 1
        call = ChatMessageToolCall(
            id="call_1",
            type="function",
            function=ChatMessageToolCallFunction(
                name="final_answer", arguments={"answer": self.answer}
            ),
        )
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[call])


def test_no_managed_agent_class_in_smolagents_1_26():
    """The headline freshness ban (06 §6, T10.1): `ManagedAgent` is GONE in smolagents 1.26.0 —
    deprecated 1.8.0, dropped from docs 1.21.0, absent from agents.py. A tutorial that does
    `from smolagents import ManagedAgent` is dead. The current mechanism is name + description +
    managed_agents=[...]."""
    import smolagents

    assert not hasattr(smolagents, "ManagedAgent"), "ManagedAgent must not exist in 1.26.0"
    with pytest.raises(ImportError):
        from smolagents import ManagedAgent  # noqa: F401


def test_build_web_researcher_is_a_toolcalling_agent_named_web_researcher():
    """build_web_researcher returns a ToolCallingAgent named EXACTLY 'web_researcher' (06 §2
    canonical name — not web_agent/researcher/WebAgent), with the web tools and max_steps=10.
    name + description are what make it callable by a manager."""
    sub = build_web_researcher(model=_FakeToolCallModel("ok"))
    assert isinstance(sub, ToolCallingAgent)
    assert sub.name == "web_researcher" == WEB_RESEARCHER_NAME
    assert sub.description == WEB_RESEARCHER_DESCRIPTION
    assert sub.description.strip(), "the description is what the manager reads to delegate"
    assert sub.max_steps == WEB_RESEARCHER_MAX_STEPS == 10


def test_web_researcher_carries_the_web_tools_not_the_manager():
    """The web tools (web_search + visit_webpage) live ON the sub-agent (context isolation) —
    exactly ONE web-search tool (web_search), per 06 §6. final_answer is auto-added (it is how the
    sub-agent returns its summary)."""
    sub = build_web_researcher(model=_FakeToolCallModel("ok"))
    names = set(sub.tools)
    assert "web_search" in names  # the canonical WebSearchTool name (NOT "search")
    assert "visit_webpage" in names
    assert "final_answer" in names
    # Sanity: the tool objects are the real smolagents tools.
    assert isinstance(sub.tools["web_search"], WebSearchTool)
    assert isinstance(sub.tools["visit_webpage"], VisitWebpageTool)


def test_web_researcher_uses_make_model_when_no_model_passed():
    """The model ALWAYS comes from make_model (M4 frozen contract) — never an InferenceClientModel
    built in team.py, never HfApiModel. Default backend hf -> InferenceClientModel."""
    sub = build_web_researcher()
    assert isinstance(sub.model, InferenceClientModel)
    assert sub.model.model_id == DEFAULT_MODEL_ID


def test_web_researcher_max_steps_is_overridable():
    """max_steps is the per-delegation cost ceiling (T7 production note): lower it (e.g. to 3) to
    surface the 'empty-handed' failure mode. The default is 10."""
    sub = build_web_researcher(model=_FakeToolCallModel("ok"), max_steps=3)
    assert sub.max_steps == 3


def test_web_researcher_provide_run_summary_toggle():
    """provide_run_summary (default False) is a real smolagents knob: when True the manager sees
    the sub-agent's reasoning, not just its final answer (the lab's 'Try it yourself')."""
    off = build_web_researcher(model=_FakeToolCallModel("ok"))
    assert off.provide_run_summary is False
    on = build_web_researcher(model=_FakeToolCallModel("ok"), provide_run_summary=True)
    assert on.provide_run_summary is True


def test_build_quill_signature_extends_with_m10_managed_agents_by_addition():
    """build_quill is EXTENDED by ADD (06 §6): managed_agents is a NEW keyword-only arg; `model`
    still leads and every M4-M9 arg survives. No prior call site breaks."""
    params = inspect.signature(build_quill).parameters
    assert list(params)[0] == "model"
    assert "managed_agents" in params
    assert params["managed_agents"].kind == inspect.Parameter.KEYWORD_ONLY
    # The M7/M8/M9 args survive the M10 extension.
    for prior in ("planning_interval", "instructions", "final_answer_checks",
                  "use_structured_outputs_internally", "extra_tools"):
        assert prior in params


def test_build_quill_default_team_registers_the_web_researcher(fake_model):
    """Omitting managed_agents -> Quill's DEFAULT team: a single web_researcher registered on the
    manager. agent.managed_agents is a name-keyed dict, callable by exactly 'web_researcher'."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert isinstance(agent.managed_agents, dict)
    assert "web_researcher" in agent.managed_agents
    sub = agent.managed_agents["web_researcher"]
    assert isinstance(sub, ToolCallingAgent)
    assert sub.name == "web_researcher"
    assert sub.max_steps == 10


def test_build_quill_is_a_manager_codeagent_without_web_tools(fake_model):
    """The manager is a CodeAgent whose OWN toolbox has the data tools but NOT the web tools
    (they moved to the sub-agent — context isolation). save_chart STAYS on the manager."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert isinstance(agent, CodeAgent)
    names = set(agent.tools)
    assert {"load_dataset", "profile_dataframe", "save_chart", "final_answer"} <= names
    assert "web_search" not in names
    assert "visit_webpage" not in names


def test_build_quill_empty_managed_agents_is_a_solo_manager(fake_model):
    """managed_agents=[] is honoured verbatim: a SOLO manager with no team (the pre-M10
    single-agent shape, minus the web tools)."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]), managed_agents=[])
    assert agent.managed_agents == {}


def test_build_quill_accepts_a_custom_team(fake_model):
    """A caller's own sub-agent list is wired verbatim (registered by its name)."""
    custom = build_web_researcher(model=_FakeToolCallModel("ok"))
    agent = build_quill(model=fake_model(["final_answer('ok')"]), managed_agents=[custom])
    assert agent.managed_agents["web_researcher"] is custom


def test_build_quill_shares_the_injected_model_with_the_default_team(fake_model):
    """When you inject a model (a fake model in tests) AND let Quill build the default team, the
    SAME model powers the sub-agent — so one fake model drives the whole team offline. (Here a
    CodeAgent fake_model is shared; the dedicated end-to-end test below gives the sub-agent its own
    tool-call model so the ToolCallingAgent can actually terminate.)"""
    shared = fake_model(["final_answer('ok')"])
    agent = build_quill(model=shared)
    assert agent.model is shared
    assert agent.managed_agents["web_researcher"].model is shared


def test_manager_stays_local_executor_for_managed_agents(fake_model):
    """T10.7: the manager MUST stay executor_type='local' to run managed agents. (A remote
    executor + managed_agents raises Exception('Managed agents are not yet supported with remote
    code execution.') — Approach 1 ships no secrets into the sandbox; Approach 2 is M15.) Here we
    pin the local default + the untouched least-privilege import lock."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert agent.executor_type == "local"
    assert agent.additional_authorized_imports == [
        "pandas", "numpy", "matplotlib.*", "json", "statistics",
    ]
    assert "*" not in agent.additional_authorized_imports


def test_remote_executor_plus_managed_agents_raises_the_documented_exception(
    monkeypatch, fake_model
):
    """T10.7, the constraint nobody warns you about (06 §5.6): a REMOTE executor + managed_agents
    raises Exception('Managed agents are not yet supported with remote code execution.'). The
    guard is in create_python_executor and fires at construction, BEFORE any container is built —
    so this is a pure OFFLINE test (no Docker needed). Quill's default team makes this matter:
    QUILL_EXECUTOR=docker would crash, which is exactly why M10 stays local (Approach 2 is M15)."""
    monkeypatch.setenv("QUILL_EXECUTOR", "docker")
    with pytest.raises(Exception, match="Managed agents are not yet supported"):
        build_quill(model=fake_model(["final_answer('ok')"]))
    # A SOLO manager (no team) is fine under a remote executor — it is the COMBINATION that fails.
    # (We do NOT actually build the container here; we only prove the team-vs-no-team distinction
    # by checking the docker path no longer trips on managed_agents when the team is empty — it
    # would instead proceed to build a DockerExecutor, which needs Docker, so we assert via the
    # error type: with [] there is no "Managed agents" message.)
    try:
        build_quill(model=fake_model(["final_answer('ok')"]), managed_agents=[])
    except Exception as exc:  # may fail later trying to reach Docker — that is NOT our error
        assert "Managed agents are not yet supported" not in str(exc)


def test_manager_delegates_to_the_sub_agent_and_finishes_offline(fake_model):
    """The headline M10 assertion (06 §6): a fake-model manager DELEGATES to the web_researcher
    by writing `web_researcher("...")` in its sandboxed code, gets the sub-agent's summary back as
    the observation, and finishes. BOTH agents run offline — the manager on a fake CodeAgent model,
    the sub-agent on a fake tool-call model that emits final_answer. Zero LLM calls, no network."""
    # The manager: delegate, capture the summary, then answer with it.
    script = [
        "summary = web_researcher('What is the SaaS industry average annual churn?')\n"
        "print(summary)",
        "final_answer(summary)",
    ]
    agent = build_quill(model=fake_model(script), final_answer_checks=[])
    # Give the registered sub-agent a real (offline) tool-call model so it can actually run its
    # OWN ReAct loop and return — no monkeypatch of .run(), the sub-agent loop genuinely executes.
    agent.managed_agents["web_researcher"].model = _FakeToolCallModel(
        "SaaS median annual churn is ~5% [https://example.com/saas-churn]."
    )
    out = agent.run(build_task(CSV, "Is our churn high vs the SaaS average?"))
    assert "SaaS median annual churn is ~5%" in str(out)
    # The delegation really happened: the manager's code referenced web_researcher, and the run
    # ended cleanly (the last ActionStep is the accepted final answer).
    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    assert any(s.is_final_answer for s in action_steps)
    assert action_steps[-1].error is None


def test_team_run_ends_in_a_cited_quillreport_offline(fake_model):
    """End-to-end OFFLINE (06 §6): the manager delegates web research to the (fake-model)
    web_researcher, then builds a cited QuillReport — a finding with a [1] marker, a saved chart,
    and a Source. The default final_answer_checks ACCEPT it (chart present; the web-source check
    fires because the manager's code called web_researcher, and a Source is present)."""
    delegate = (
        "summary = web_researcher('What is the SaaS industry average annual churn rate?')\n"
        "print(summary)"
    )
    report_step = (
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        f"df = pd.read_csv({CUSTOMERS!r})\n"
        "rate = float(df['churned'].mean())\n"
        "plt.figure(); plt.bar(['churned'], [rate])\n"
        "path = save_chart('m10_churn_vs_industry')\n"
        "rep = QuillReport(question='Is our churn high vs the SaaS industry average?', "
        "findings=['Our churn is above the ~5% SaaS median [1]'], chart_paths=[path], "
        "sources=[Source(url='https://example.com/saas-churn', title='SaaS Churn Benchmarks')])\n"
        "final_answer(rep)"
    )
    agent = build_quill(model=fake_model([delegate, report_step]))
    agent.managed_agents["web_researcher"].model = _FakeToolCallModel(
        "SaaS median annual churn is ~5% [https://example.com/saas-churn]."
    )
    out = agent.run(build_task(CUSTOMERS, "Is our churn high vs the SaaS industry average?"))

    assert isinstance(out, QuillReport)
    assert out.chart_paths and out.chart_paths[0].endswith(".png")
    assert os.path.exists(out.chart_paths[0]), "the report's chart was really saved"
    # A real citation: the [1] marker resolves to a Source in the rendered Markdown.
    assert out.sources and out.sources[0].url == "https://example.com/saas-churn"
    md = out.to_markdown()
    assert "[1]" in md
    assert "[1] [SaaS Churn Benchmarks](https://example.com/saas-churn)" in md
    os.remove(out.chart_paths[0])


def test_web_researcher_name_in_the_manager_system_prompt(fake_model):
    """A CodeAgent manager exposes each managed agent as a callable function in the sandbox; the
    system prompt advertises the team by name + description so the model knows it can delegate."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    sp = agent.system_prompt
    assert "web_researcher" in sp


def test_web_researcher_heuristic_trips_the_web_source_check():
    """M8 web-source check, M10 path (06 §2 frozen check, extended heuristic): a report with web
    claims (the manager called web_researcher in its code) but NO source is rejected; with a Source
    it passes. We exercise the check FUNCTION directly with stub memory — no agent, no LLM."""
    no_sources = QuillReport(question="Q", chart_paths=["outputs/c.png"], sources=[])
    memory = _StubMemory([_StubStep(code_action="s = web_researcher('saas average churn')")])
    with pytest.raises(ValueError, match="sources is empty"):
        check_has_source_for_web_claims(no_sources, memory, None)
    # With a Source present, the same delegated-web run passes.
    assert check_has_source_for_web_claims(_complete_report(), memory, None) is True


# ======================================================================================
# Module 9 (carried forward): MCP entry points exist and are imported the CORRECT way (T9.1, 06 §6).
# OFFLINE: assert imports/shape only. We NEVER open an MCP connection here (that is the live test).
# ======================================================================================

def test_mcp_entry_points_import_from_smolagents():
    """ToolCollection and MCPClient are the smolagents MCP entry points (T9.2/T9.3). They import
    from smolagents and carry the from_mcp classmethod (ToolCollection) / the lifecycle methods
    (MCPClient)."""
    from smolagents import MCPClient as SAMCPClient
    from smolagents import ToolCollection as SAToolCollection

    assert SAToolCollection is ToolCollection
    assert SAMCPClient is MCPClient
    assert hasattr(ToolCollection, "from_mcp")
    assert hasattr(ToolCollection, "from_hub")
    for method in ("connect", "get_tools", "disconnect"):
        assert hasattr(MCPClient, method), f"MCPClient must expose {method}()"
    # MCPClient is usable as a context manager (with MCPClient(params) as tools: ...).
    assert hasattr(MCPClient, "__enter__") and hasattr(MCPClient, "__exit__")


def test_tool_has_no_from_mcp_the_freshness_trap():
    """The headline freshness ban (06 §6, T3.13): there is NO `Tool.from_mcp`. MCP lives ONLY on
    ToolCollection.from_mcp / MCPClient. Inventing Tool.from_mcp is the #1 stale-tutorial error."""
    from smolagents import Tool

    assert not hasattr(Tool, "from_mcp"), "Tool.from_mcp must NOT exist — MCP is on ToolCollection"
    # The real Tool.from_* importers DO exist (other ecosystems — T3.13).
    for importer in ("from_hub", "from_space", "from_gradio", "from_langchain"):
        assert hasattr(Tool, importer), f"Tool.{importer} should exist"


def test_from_mcp_is_a_context_manager_with_the_pinned_signature():
    """ToolCollection.from_mcp(server_parameters, trust_remote_code=False, structured_output=...)
    is a CONTEXT MANAGER (T9.2): it returns an object you use in `with ... as tc:`. We assert the
    signature shape and that trust_remote_code / structured_output are parameters — WITHOUT
    calling it (calling would start a server)."""
    sig = inspect.signature(ToolCollection.from_mcp)
    params = list(sig.parameters)
    assert params[0] == "server_parameters"
    assert "trust_remote_code" in params
    assert "structured_output" in params  # T9.6 — the informational schema flag


# ======================================================================================
# Module 9 (NEW): the server-parameter builders — quill/tools/mcp.py.
# OFFLINE: building params is pure (no subprocess); we assert shape/transport, never connect.
# ======================================================================================

def test_data_mcp_server_params_builds_stdio_without_starting_a_server():
    """data_mcp_server_params() returns a stdio StdioServerParameters for `uvx
    mcp-server-sqlite --db-path data/sales.db` (T9.4 — stdio). Building it is side-effect-free:
    no subprocess starts (that only happens at the `with from_mcp(...)` site)."""
    params = data_mcp_server_params()
    assert isinstance(params, StdioServerParameters)
    assert params.command == "uvx"
    assert DEFAULT_MCP_SQLITE_PACKAGE in params.args
    assert "--db-path" in params.args
    assert DEFAULT_SQLITE_DB_PATH in params.args
    # UV_PYTHON is pinned and the env is forwarded (so the subprocess has a normal PATH).
    assert params.env.get("UV_PYTHON") == "3.12"
    assert "PATH" in params.env  # forwarded from os.environ


def test_data_mcp_server_params_accepts_custom_db_and_package():
    """The db path and package are overridable (so a pinned version / another DB can be used)."""
    params = data_mcp_server_params(db_path="data/other.db", package="mcp-server-sqlite==1.2.3")
    assert "data/other.db" in params.args
    assert "mcp-server-sqlite==1.2.3" in params.args


def test_http_server_params_defaults_to_streamable_http_not_sse():
    """http_server_params() builds the streamable-http dict (T9.4): streamable-http is the
    CURRENT default transport (as of 1.26.0); SSE is deprecated and must be opt-in only."""
    params = http_server_params()
    assert isinstance(params, dict)
    assert params["transport"] == "streamable-http"  # default, NOT sse
    assert params["url"].endswith("/mcp")
    # SSE is available but only when explicitly asked for (the deprecated legacy path).
    legacy = http_server_params(url="http://host/sse", transport="sse")
    assert legacy["transport"] == "sse"


def test_describe_server_params_is_pure_and_reports_transport():
    """describe_server_params() is a pure helper (no connection) that names the transport — used
    by the demo to print WHAT it will connect to before opening the subprocess/request."""
    stdio = describe_server_params(data_mcp_server_params())
    assert stdio.startswith("stdio:")
    assert "uvx" in stdio and DEFAULT_MCP_SQLITE_PACKAGE in stdio
    http = describe_server_params(http_server_params())
    assert http.startswith("streamable-http:")


# ======================================================================================
# Module 9 (NEW): build_quill gains extra_tools= (EXTEND by ADD) — quill/agent.py.
# OFFLINE: a fake @tool stands in for an MCP/Hub tool; we never open a server.
# ======================================================================================

def _fake_sql_tool():
    """A stand-in for an MCP-served tool (a smolagents @tool), so we can test extra_tools=
    OFFLINE without connecting to an MCP server."""
    from smolagents import tool

    @tool
    def read_query(query: str) -> str:
        """Run a read-only SQL query against the sales database and return the rows as text.

        Args:
            query: a single SQL SELECT statement.
        """
        return f"STUB ROWS for: {query}"

    return read_query


def test_build_quill_signature_extends_with_m9_extra_tools_by_addition(fake_model):
    """build_quill is EXTENDED by ADD (06 §6): extra_tools is a NEW keyword-only arg defaulting
    to None; `model` still leads and every M4-M8 arg survives. No prior call site breaks."""
    params = inspect.signature(build_quill).parameters
    assert list(params)[0] == "model"
    assert "extra_tools" in params
    assert params["extra_tools"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["extra_tools"].default is None
    # The M7/M8 args survive the M9 extension.
    for prior in ("planning_interval", "instructions", "final_answer_checks",
                  "use_structured_outputs_internally"):
        assert prior in params


def test_build_quill_default_has_no_extra_tools_only_the_frozen_toolbox(fake_model):
    """Omitting extra_tools -> Quill's data toolbox only (no ecosystem tools added). NOTE (M10):
    the web tools are NO LONGER on the manager — they moved into the web_researcher sub-agent —
    so we assert the DATA tools + final_answer, and that web_search/visit_webpage are absent here."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    names = set(agent.tools)
    assert {"load_dataset", "profile_dataframe", "save_chart", "final_answer"} <= names
    assert "web_search" not in names  # M10: web tools live on the web_researcher sub-agent now
    assert "visit_webpage" not in names
    assert "read_query" not in names  # nothing extra by default


def test_build_quill_appends_extra_tools_after_the_local_toolbox(fake_model):
    """extra_tools are APPENDED to the frozen local toolbox (the local tools stay; the extra is
    reachable by name). This is the seam MCP/Hub/LangChain tools use."""
    extra = _fake_sql_tool()
    agent = build_quill(model=fake_model(["final_answer('ok')"]), extra_tools=[extra])
    names = set(agent.tools)
    # The frozen local tools are untouched...
    assert {"load_dataset", "profile_dataframe", "save_chart"} <= names
    # ...and the extra (MCP-style) tool is now in the toolbox, reachable by its name.
    assert "read_query" in names
    assert agent.tools["read_query"] is extra


def test_extra_tools_does_not_widen_the_frozen_import_lock(fake_model):
    """Adding ecosystem tools must NOT touch the FROZEN least-privilege import lock (06 §2): MCP
    tools run OUTSIDE the sandbox, so the in-sandbox import allow-list is unchanged."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]), extra_tools=[_fake_sql_tool()])
    assert agent.additional_authorized_imports == [
        "pandas", "numpy", "matplotlib.*", "json", "statistics",
    ]
    assert "*" not in agent.additional_authorized_imports


def test_runtime_attach_of_a_tool_still_works(fake_model):
    """A tool can also be added at RUNTIME via agent.tools[name] = t (the toolbox is a name-keyed
    dict) — the same seam load_hub_tool.attach_to_quill uses for a Hub tool."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    extra = _fake_sql_tool()
    attach_to_quill(agent, extra)
    assert agent.tools["read_query"] is extra


def test_fake_mcp_style_tool_runs_through_the_agent_loop_offline(fake_model):
    """End-to-end OFFLINE: a fake-model Quill CALLS an MCP-style extra tool from its generated
    code and the tool's return becomes the observation — exactly how a real MCP `read_query` would
    behave, minus the server. No subprocess, no network."""
    extra = _fake_sql_tool()
    script = (
        "rows = read_query('SELECT category, SUM(net_rev) FROM sales GROUP BY category')\n"
        "print(rows)\n"
        "final_answer(rows)"
    )
    agent = build_quill(model=fake_model([script]), extra_tools=[extra], final_answer_checks=[])
    out = agent.run(build_sql_task("Which category has the most revenue?"))
    assert "STUB ROWS for:" in str(out)


# ======================================================================================
# Module 9 (NEW): run_with_mcp + build_sql_task shape — quill/agent.py.
# OFFLINE: assert signatures / task text; the real connection is the live test only.
# ======================================================================================

def test_run_with_mcp_has_the_expected_signature():
    """run_with_mcp(task, server_parameters=None, *, model=None, trust_remote_code=True) — the
    one-shot MCP path (T9.2). We assert the shape WITHOUT calling it (no server started)."""
    sig = inspect.signature(run_with_mcp)
    params = sig.parameters
    assert list(params)[0] == "task"
    assert "server_parameters" in params
    assert params["server_parameters"].default is None  # defaults to data_mcp_server_params()
    assert params["model"].kind == inspect.Parameter.KEYWORD_ONLY
    assert params["trust_remote_code"].default is True  # the security gate, on by default


def test_build_sql_task_mentions_the_mcp_sql_tools():
    """build_sql_task tells Quill the MCP SQL tools exist (list_tables/describe_table/read_query)
    and asks for a QuillReport — so the model knows to use the server's tools."""
    task = build_sql_task("Which category grew fastest?")
    assert "read_query" in task
    assert "QuillReport" in task
    assert "save_chart" in task


# ======================================================================================
# Module 9 (NEW): the Hub scripts — quill/scripts/*. OFFLINE: shape only (no network/Hub).
# ======================================================================================

def test_load_hub_tool_has_the_right_shape_and_default_repo():
    """load_hub_tool(repo_id=DEFAULT, *, trust_remote_code=True) wraps load_tool. We assert the
    signature + default WITHOUT calling it (it would hit the Hub). trust_remote_code defaults True
    because Hub tool code runs LOCALLY (the same interop gate as MCP)."""
    sig = inspect.signature(load_hub_tool)
    params = sig.parameters
    assert list(params)[0] == "repo_id"
    assert params["repo_id"].default == DEFAULT_HUB_TOOL_REPO
    assert params["trust_remote_code"].default is True
    assert isinstance(DEFAULT_HUB_TOOL_REPO, str) and "/" in DEFAULT_HUB_TOOL_REPO


def test_load_tool_is_a_top_level_smolagents_function():
    """load_tool is the top-level Hub loader (T3.15); Tool.from_hub is the classmethod twin. Both
    require trust_remote_code=True for the same reason (Hub code runs locally)."""
    from smolagents import Tool, load_tool

    assert callable(load_tool)
    sig = inspect.signature(load_tool)
    assert "trust_remote_code" in sig.parameters
    assert "trust_remote_code" in inspect.signature(Tool.from_hub).parameters


def test_save_chart_obeys_the_pushable_rules_via_a_clean_local_save(tmp_path):
    """The pushable proof (T3.15, 06 §6 step 5), OFFLINE: save_chart.save(dir) writes the Space
    files (save_chart.py + app.py + requirements.txt) with NO error. A clean save IS the proof
    the M3 pushable rules held — a top-level import or an __init__ arg would make save() raise."""
    out_dir = save_save_chart_locally(str(tmp_path / "save_chart_tool"))
    written = {p.name for p in pathlib.Path(out_dir).iterdir()}
    assert "save_chart.py" in written
    assert "app.py" in written
    assert "requirements.txt" in written


def test_save_chart_init_takes_no_args_pushable_rule_3():
    """Pushable rule 3 (T3.15): save_chart's __init__ takes no arg other than self (init args are
    not serializable to the Hub). save_chart does not override __init__, so it is compliant."""
    sig = inspect.signature(save_chart.__init__)
    # Only `self` (and possibly **kwargs from the base Tool) — never a required custom arg.
    required = [p for n, p in sig.parameters.items()
               if n != "self" and p.default is inspect.Parameter.empty
               and p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                              inspect.Parameter.POSITIONAL_ONLY)]
    assert required == [], f"save_chart.__init__ must take no required arg beyond self: {required}"


def test_save_chart_imports_live_inside_methods_pushable_rule_2():
    """Pushable rule 2 (T3.15): every import is INSIDE a method (setup/forward), never at module
    top level — that is what lets the Hub re-execute the tool in a fresh Space. We check the source
    of setup/forward references matplotlib imports inside the function bodies."""
    setup_src = inspect.getsource(save_chart.setup)
    forward_src = inspect.getsource(save_chart.forward)
    assert "import matplotlib" in setup_src  # imported inside setup(), not at top of data.py
    assert "import matplotlib.pyplot" in forward_src  # imported inside forward()


# ======================================================================================
# Module 9 (NEW): build the SQLite DB the MCP server serves — quill/scripts/build_sales_db.py.
# OFFLINE: pure pandas + sqlite3, no network.
# ======================================================================================

def test_build_sales_db_creates_a_queryable_sqlite_from_the_csv(tmp_path):
    """build_sales_db(csv, db) materializes data/sales.csv into a SQLite `sales` table (so the MCP
    server has something to serve). We build into a tmp file and assert it is queryable OFFLINE."""
    import sqlite3

    db = tmp_path / "sales.db"
    out = build_sales_db(CSV, str(db))
    assert out == str(db)
    assert db.exists()
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        cats = {r[0] for r in con.execute("SELECT DISTINCT category FROM sales")}
    finally:
        con.close()
    assert rows == 108
    assert {"Free", "Pro", "Team"} <= cats


def test_build_sales_db_raises_on_missing_csv(tmp_path):
    """A missing CSV fails loud (FileNotFoundError), not a silent empty DB."""
    with pytest.raises(FileNotFoundError):
        build_sales_db(str(tmp_path / "nope.csv"), str(tmp_path / "x.db"))


# ======================================================================================
# Module 8 (NEW): the QuillReport / Source schema — FROZEN contract (06 §2) — quill/report.py
# ======================================================================================

def test_quillreport_is_the_frozen_five_field_schema():
    """FROZEN (06 §2): QuillReport has EXACTLY these five fields, in this order, no more. A sixth
    field would break M12 (citations) and M14 (eval), so it must trip this test first."""
    field_names = [f.name for f in dataclasses.fields(QuillReport)]
    assert field_names == ["question", "findings", "chart_paths", "sources", "caveats"]


def test_source_is_the_frozen_two_field_schema():
    """FROZEN (06 §2): Source is exactly {url, title}."""
    field_names = [f.name for f in dataclasses.fields(Source)]
    assert field_names == ["url", "title"]


def test_quillreport_defaults_are_empty_lists_not_shared():
    """The list fields default to FRESH empty lists (default_factory) — never a shared mutable."""
    a = QuillReport(question="Q")
    b = QuillReport(question="Q")
    assert a.findings == [] and a.chart_paths == [] and a.sources == [] and a.caveats == []
    a.findings.append("x")
    assert b.findings == [], "each instance must get its OWN list (no shared default)"


def test_to_markdown_contains_a_numbered_citation_for_one_source():
    """The headline rendering assertion (06 §6 step 2): a report with ONE source renders the [1]
    marker in its body and a '[1] [title](url)' line in the Sources section."""
    report = _complete_report()
    md = report.to_markdown()
    assert "[1]" in md  # the citation marker is present
    assert "[1] [SaaS Trends 2025](https://example.com/saas-2025)" in md  # mapped to the source
    assert report.question in md  # the question is the title
    assert "Team grew fastest" in md  # the finding is rendered


def test_to_markdown_numbers_multiple_sources_in_order():
    """With two sources, the Sources section numbers them [1], [2] in list order."""
    report = QuillReport(
        question="Q",
        findings=["A [1]", "B [2]"],
        chart_paths=["outputs/c.png"],
        sources=[Source(url="http://a", title="A src"), Source(url="http://b", title="B src")],
    )
    md = report.to_markdown()
    assert "[1] [A src](http://a)" in md
    assert "[2] [B src](http://b)" in md


def test_to_markdown_omits_empty_sections_but_always_has_the_question():
    """A minimal report (just a question) still renders cleanly: the title is there, and the
    Sources/Charts/Caveats headings are omitted rather than shown empty."""
    md = QuillReport(question="Just a question").to_markdown()
    assert "# Just a question" in md
    assert "## Sources" not in md
    assert "## Charts" not in md
    assert "## Caveats" not in md


def test_to_markdown_lists_charts_and_caveats_when_present():
    """Charts and caveats get their own sections when populated."""
    md = _complete_report().to_markdown()
    assert "## Charts" in md and "outputs/category_growth.png" in md
    assert "## Caveats" in md and "Q4 is a partial quarter." in md


# ======================================================================================
# Module 8 (NEW): the final_answer_checks — 3-arg validators (T8.3) — quill/report.py
# OFFLINE: assert the check FUNCTIONS directly (no LLM), then through a fake-model agent run.
# ======================================================================================

def test_checks_have_the_frozen_three_arg_signature():
    """FROZEN (06 §2): every check is (final_answer, memory, agent) — the smolagents call site is
    `check(final_answer, self.memory, agent=self)`. The 2-arg guided-tour form is the trap."""
    for check in (check_has_chart, check_has_source_for_web_claims):
        params = list(inspect.signature(check).parameters)
        assert params == ["final_answer", "memory", "agent"], f"{check.__name__}: {params}"


def test_quill_final_answer_checks_is_the_two_checks_fresh_copy():
    """quill_final_answer_checks() returns Quill's two checks as a FRESH list (callers can't
    mutate ours)."""
    checks = quill_final_answer_checks()
    assert checks == [check_has_chart, check_has_source_for_web_claims]
    assert checks == QUILL_FINAL_ANSWER_CHECKS
    assert checks is not QUILL_FINAL_ANSWER_CHECKS  # a copy, not the shared constant
    checks.append(lambda *a: True)
    assert quill_final_answer_checks() == [check_has_chart, check_has_source_for_web_claims]


def test_check_has_chart_accepts_a_report_with_a_chart():
    """A complete QuillReport (chart_paths non-empty) passes the chart check."""
    assert check_has_chart(_complete_report(), _StubMemory(), None) is True


def test_check_has_chart_rejects_a_report_without_a_chart():
    """An empty chart_paths is rejected with an actionable message (it names save_chart)."""
    empty = QuillReport(question="Q")
    with pytest.raises(ValueError, match="at least one saved chart"):
        check_has_chart(empty, _StubMemory(), None)


def test_check_has_chart_rejects_a_non_quillreport():
    """A bare string (or any non-QuillReport) is rejected: shape is wrong, not just content."""
    with pytest.raises(ValueError, match="must be a QuillReport"):
        check_has_chart("growth is strong", _StubMemory(), None)


def test_check_source_passes_when_no_web_tool_was_used():
    """A purely LOCAL analysis (no web tool in memory) needs no sources — even sources=[] passes."""
    local_only = QuillReport(question="Q", chart_paths=["outputs/c.png"])
    memory = _StubMemory([_StubStep(code_action="df.groupby('category').sum()")])
    assert check_has_source_for_web_claims(local_only, memory, None) is True


def test_check_source_rejects_empty_sources_when_a_web_tool_was_used():
    """The conditional rule (06 §6 step 3): the run called web_search, so an empty sources list is
    rejected. The message is actionable (it names Source and the [n] citation)."""
    no_sources = QuillReport(question="Q", chart_paths=["outputs/c.png"], sources=[])
    memory = _StubMemory([_StubStep(code_action="hits = web_search('saas growth 2025')")])
    with pytest.raises(ValueError, match="sources is empty"):
        check_has_source_for_web_claims(no_sources, memory, None)


def test_check_source_accepts_sources_when_a_web_tool_was_used():
    """With a web tool used AND a source present, the check passes."""
    memory = _StubMemory([_StubStep(code_action="page = visit_webpage('https://example.com')")])
    assert check_has_source_for_web_claims(_complete_report(), memory, None) is True


def test_check_source_detects_web_tool_in_tool_call_arguments():
    """The heuristic also reads python_interpreter ToolCall.arguments (where a CodeAgent records
    the executed code), not just code_action — so a web call there is still caught."""
    no_sources = QuillReport(question="Q", chart_paths=["outputs/c.png"])
    call = dataclasses.make_dataclass("TC", ["name", "arguments", "id"])(
        name="python_interpreter", arguments="res = web_search('q')", id="call_1"
    )
    memory = _StubMemory([_StubStep(code_action=None, tool_calls=[call])])
    with pytest.raises(ValueError, match="sources is empty"):
        check_has_source_for_web_claims(no_sources, memory, None)


# ======================================================================================
# Module 8 (NEW): build_quill wires the checks + the report contract end to end (fake model).
# ======================================================================================

def _agent_that_builds_report(fake_model, report_code: str, **build_kwargs):
    """Build a fake-model Quill that constructs a QuillReport in the sandbox. QuillReport/Source
    are injected by build_quill into the executor (no quill.report import needed — the FROZEN
    import lock forbids it)."""
    return build_quill(model=fake_model([report_code]), **build_kwargs)


def test_build_quill_signature_extends_with_m8_args_by_addition():
    """build_quill is EXTENDED by ADD (06 §6): final_answer_checks +
    use_structured_outputs_internally are NEW keyword-only args; model still leads; the M7 args
    are still there. No prior call site breaks."""
    params = inspect.signature(build_quill).parameters
    assert list(params)[0] == "model"
    assert "final_answer_checks" in params
    assert params["final_answer_checks"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "use_structured_outputs_internally" in params
    assert params["use_structured_outputs_internally"].default is False
    assert params["use_structured_outputs_internally"].kind == inspect.Parameter.KEYWORD_ONLY
    # The M7 args survive the M8 extension.
    assert "planning_interval" in params and "instructions" in params


def test_build_quill_wires_quills_default_checks(fake_model):
    """Omitting final_answer_checks -> the agent gets Quill's two default checks (06 §6 step 4)."""
    agent = build_quill(model=fake_model(["final_answer('x')"]))
    assert agent.final_answer_checks == [check_has_chart, check_has_source_for_web_claims]


def test_build_quill_empty_checks_opts_out(fake_model):
    """final_answer_checks=[] is honoured verbatim (the bench baseline / pre-M8 mechanic tests)."""
    agent = build_quill(model=fake_model(["final_answer('x')"]), final_answer_checks=[])
    assert agent.final_answer_checks == []


def test_build_quill_accepts_custom_checks(fake_model):
    """A caller's own list is wired verbatim."""
    def my_check(final_answer, memory, agent):
        return True

    agent = build_quill(model=fake_model(["final_answer('x')"]), final_answer_checks=[my_check])
    assert agent.final_answer_checks == [my_check]


def test_build_quill_default_does_not_enable_structured_outputs(fake_model):
    """Quill's default keeps use_structured_outputs_internally OFF (06 §6 step 4): the default
    InferenceClientModel only supports response_format on cerebras/fireworks-ai, so structured
    outputs are NOT on the mandatory path. (CodeAgent stores it as the private
    `_use_structured_outputs_internally`.)"""
    agent = build_quill(model=fake_model(["final_answer('x')"]))
    assert agent._use_structured_outputs_internally is False


def test_build_quill_can_enable_structured_outputs_optionally(fake_model):
    """The OPTIONAL extra (06 §6 step 6): you CAN turn it on — when True, CodeAgent loads
    structured_code_agent.yaml (not code_agent.yaml). We assert the flag flips, the agent still
    builds, and the structured YAML's distinctive instruction landed in the system prompt
    (offline, fake model)."""
    agent = build_quill(
        model=fake_model(["final_answer('x')"]),
        use_structured_outputs_internally=True,
        final_answer_checks=[],
    )
    assert agent._use_structured_outputs_internally is True
    assert isinstance(agent, CodeAgent)
    # structured_code_agent.yaml tells the model to answer as JSON with thought+code keys.
    assert '"code"' in agent.system_prompt or "JSON" in agent.system_prompt


def test_build_quill_exposes_report_types_to_the_sandbox(fake_model):
    """build_quill injects QuillReport/Source into the executor so the agent can build a report
    WITHOUT importing quill.report (the FROZEN import lock forbids that). They land in the
    interpreter state, and the import lock is untouched."""
    agent = build_quill(model=fake_model(["final_answer('x')"]), final_answer_checks=[])
    state = agent.python_executor.state
    assert state.get("QuillReport") is QuillReport
    assert state.get("Source") is Source
    # The FROZEN least-privilege import lock is unchanged — we did NOT widen it to allow the import.
    assert "quill.report" not in agent.additional_authorized_imports
    assert agent.additional_authorized_imports == [
        "pandas", "numpy", "matplotlib.*", "json", "statistics",
    ]


def test_fake_model_agent_returns_a_validated_quillreport(fake_model):
    """The headline M8 assertion (06 §6): a fake-model Quill that builds a complete QuillReport
    (a finding + a chart_path) and calls final_answer(report) PASSES the default checks and the
    run returns the QuillReport itself."""
    report_code = (
        "rep = QuillReport(question='Which category grew fastest?', "
        "findings=['Team grew fastest [n/a]'], chart_paths=['outputs/category_growth.png'])\n"
        "final_answer(rep)"
    )
    agent = _agent_that_builds_report(fake_model, report_code)
    out = agent.run(build_task(CSV, "Which category grew fastest?"))
    assert isinstance(out, QuillReport)
    assert out.chart_paths == ["outputs/category_growth.png"]
    # And the run succeeded (the checks accepted it).
    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    assert any(s.is_final_answer for s in action_steps)
    assert action_steps[-1].error is None


def test_bare_string_answer_is_rejected_and_loops_not_crashes(fake_model):
    """A fake-model Quill that answers with a BARE STRING is rejected by check_has_chart: the run
    does NOT crash — the AgentError is captured in ActionStep.error and the agent loops (we cap
    max_steps so it terminates in max_steps_error rather than burning the default 8)."""
    agent = build_quill(model=fake_model(["final_answer('growth is strong')"]))
    result = agent.run(build_task(CSV, "q"), max_steps=2, return_full_result=True)

    # The bad answer was NOT accepted (it looped to max_steps instead of succeeding).
    assert result.state == "max_steps_error"
    # Every attempt's rejection is visible as a captured AgentError (the self-correction signal).
    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    errored = [s for s in action_steps if s.error is not None]
    assert errored, "a rejected final answer must be captured in ActionStep.error, not crash"
    assert isinstance(errored[0].error, AgentError)
    assert "QuillReport" in str(errored[0].error) or "chart" in str(errored[0].error)


def test_self_correction_after_a_rejected_report(fake_model):
    """The full self-correction loop (06 §6 step 5), OFFLINE: step 1 answers without a chart
    (rejected by check_has_chart), step 2 draws + saves a chart and returns a complete
    QuillReport (accepted). We see the rejection in step 1's error and a QuillReport at the end."""
    step1 = (
        "rep = QuillReport(question='Which category grew fastest?', "
        "findings=['Team grew fastest'], chart_paths=[])\n"  # NO chart -> rejected
        "final_answer(rep)"
    )
    step2 = (
        "import matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\n"
        "plt.figure(); plt.bar(['Free','Pro','Team'],[1,2,3])\n"
        "path = save_chart('m8_self_correction')\n"
        "rep = QuillReport(question='Which category grew fastest?', "
        "findings=['Team grew fastest'], chart_paths=[path])\n"
        "final_answer(rep)"
    )
    agent = build_quill(model=fake_model([step1, step2]))
    out = agent.run(build_task(CSV, "Which category grew fastest?"))

    assert isinstance(out, QuillReport)
    assert out.chart_paths and out.chart_paths[0].endswith(".png")
    assert os.path.exists(out.chart_paths[0]), "the corrective step saved a real chart"

    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    # Step 1 was rejected (its final answer failed the chart check -> error captured).
    assert action_steps[0].error is not None
    assert isinstance(action_steps[0].error, AgentError)
    # The last step is the accepted final answer (no error).
    assert action_steps[-1].error is None
    assert action_steps[-1].is_final_answer
    os.remove(out.chart_paths[0])


def test_web_backed_report_without_a_source_is_rejected_then_corrected(fake_model):
    """The web-source check end to end (06 §6 step 3 + 5), in the M10 multi-agent world: step 1
    DELEGATES to the web_researcher sub-agent (the evidence lands in memory as a `web_researcher(
    ...)` call — the heuristic now recognises that, since the web tools left the manager), step 2
    returns a report with NO source (rejected by check_has_source_for_web_claims), step 3 adds a
    Source and is accepted. The delegation is in an EARLIER step than the answer, so it is already
    in agent.memory.steps when the check reads it. The sub-agent is stubbed (OFFLINE — no LLM)."""
    step1 = "summary = web_researcher('saas category growth 2025')\nprint(str(summary)[:30])"
    step2 = (
        "rep = QuillReport(question='Is Team growth consistent with the public trend?', "
        "findings=['Team leads, consistent with the market'], "
        "chart_paths=['outputs/x.png'], sources=[])\n"  # web claim but NO source -> rejected
        "final_answer(rep)"
    )
    step3 = (
        "rep = QuillReport(question='Is Team growth consistent with the public trend?', "
        "findings=['Team leads, consistent with the market [1]'], "
        "chart_paths=['outputs/x.png'], "
        "sources=[Source(url='https://example.com/saas-2025', title='SaaS Trends 2025')])\n"
        "final_answer(rep)"
    )
    agent = build_quill(model=fake_model([step1, step2, step3]))
    # Stub the sub-agent's run so no network/LLM is hit (offline): the manager calls
    # web_researcher(...) as a function and gets this summary back as the observation.
    agent.managed_agents["web_researcher"].run = (
        lambda task, **kw: f"STUB SUMMARY for {task} [https://example.com/saas-2025]"
    )

    out = agent.run(build_task(CSV, "Is Team growth consistent with the public trend?"))
    assert isinstance(out, QuillReport)
    assert out.sources and out.sources[0].url == "https://example.com/saas-2025"

    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    # The attempt with a web claim but no source was rejected and captured, not crashed.
    rejected = [s for s in action_steps if s.error is not None]
    assert rejected, "a web-backed report with no source must be rejected (captured error)"
    assert "sources is empty" in str(rejected[0].error)
    assert action_steps[-1].error is None  # the corrected answer was accepted


# ======================================================================================
# Module 7 (NEW): planning_interval -> a PlanningStep appears (T7.1) — quill/agent.py
# OFFLINE: the FakeModel scripts BOTH the plan and the action, so the planning LLM call is
# answered deterministically and a real PlanningStep lands in agent.memory.steps (zero tokens).
# ======================================================================================

def test_build_quill_signature_is_extended_by_addition_not_broken():
    """build_quill is EXTENDED by ADD (06 §6): planning_interval + instructions are NEW
    keyword-only args; the old positional `model` still leads. No prior call site breaks."""
    params = inspect.signature(build_quill).parameters
    assert list(params)[0] == "model"
    assert params["model"].default is None
    assert "planning_interval" in params
    assert params["planning_interval"].default is None  # default = the smolagents default (off)
    assert params["planning_interval"].kind == inspect.Parameter.KEYWORD_ONLY
    assert "instructions" in params
    assert params["instructions"].kind == inspect.Parameter.KEYWORD_ONLY


def test_default_planning_interval_is_three():
    """Quill's recommended cadence (the bench uses it): plan at step 1, then every 3 steps."""
    assert DEFAULT_PLANNING_INTERVAL == 3


def test_no_planning_interval_means_no_planning_step(fake_model):
    """Default (M2-M6 behaviour): planning_interval=None -> NO PlanningStep in memory."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert agent.planning_interval is None
    agent.run(build_task(CSV, "anything"))
    assert not any(isinstance(s, PlanningStep) for s in agent.memory.steps)


def test_planning_interval_one_inserts_a_planning_step_at_step_one(fake_model):
    """The headline M7 assertion (06 §6 step 7): with planning_interval=1 a PlanningStep is
    inserted at step 1, BEFORE the first action — visible in agent.memory.steps after a run.
    The FakeModel answers the planning call (entry 0) then the action call (entry 1)."""
    # final_answer_checks=[] opts OUT of the M8 report contract: this test asserts the M7
    # planning mechanic, so a bare string final answer is fine (M8's checks are tested below).
    agent = build_quill(
        model=fake_model([_plan(), "final_answer('done')"]),
        planning_interval=1,
        final_answer_checks=[],
    )
    assert agent.planning_interval == 1
    out = agent.run(build_task(CSV, "Which category grew fastest?"))
    assert out == "done"

    planning_steps = [s for s in agent.memory.steps if isinstance(s, PlanningStep)]
    assert planning_steps, "planning_interval=1 must insert at least one PlanningStep"
    assert planning_steps[0].plan.strip(), "the PlanningStep carries the model's plan text"

    # The plan precedes the first ActionStep in memory order (plan, then act).
    types = [type(s).__name__ for s in agent.memory.steps]
    assert types.index("PlanningStep") < types.index("ActionStep")


def test_planning_step_is_an_llm_call_without_a_tool_action(fake_model):
    """A PlanningStep is a tool-free LLM call: it carries `plan` text but no executed action.
    (Honest accounting: it still costs one model.generate call — the bench counts it.)"""
    agent = build_quill(
        model=fake_model([_plan(), "final_answer('done')"]),
        planning_interval=1,
        final_answer_checks=[],  # M7 mechanic test: opt out of the M8 report contract
    )
    agent.run(build_task(CSV, "anything"))
    plan = next(s for s in agent.memory.steps if isinstance(s, PlanningStep))
    assert isinstance(plan.plan, str) and plan.plan.strip()
    assert not hasattr(plan, "code_action") or getattr(plan, "code_action", None) is None


def test_planning_interval_three_fires_at_step_one_then_step_four(fake_model):
    """The trigger is `step_number == 1 or (step_number - 1) % interval == 0` (06 §9, T7.1):
    with planning_interval=3, planning fires at step 1 and step 4. We script 4 ActionSteps so
    the run reaches step 4 and assert exactly two PlanningSteps appear."""
    script = [
        _plan("Plan A"),        # planning before step 1
        "print('step 1')",      # action step 1
        "print('step 2')",      # action step 2
        "print('step 3')",      # action step 3
        _plan("Plan B"),        # planning before step 4 ((4-1)%3==0)
        "final_answer('done')",  # action step 4
    ]
    agent = build_quill(model=fake_model(script), planning_interval=3, final_answer_checks=[])
    out = agent.run(build_task(CSV, "long task"), max_steps=6)
    assert out == "done"
    planning_steps = [s for s in agent.memory.steps if isinstance(s, PlanningStep)]
    assert len(planning_steps) == 2, "planning_interval=3 fires at step 1 and step 4"


def test_planning_trigger_predicate_matches_step1_then_every_interval():
    """Pin the documented trigger arithmetic itself (T7.1): for interval=3 the steps that plan
    are 1, 4, 7 — `n == 1 or (n - 1) % interval == 0`. A pure-Python check, no agent run."""
    def plans(n: int, interval: int) -> bool:
        return n == 1 or (n - 1) % interval == 0

    fired = [n for n in range(1, 11) if plans(n, 3)]
    assert fired == [1, 4, 7, 10]
    assert plans(1, 3) and not plans(2, 3) and not plans(3, 3) and plans(4, 3)


# ======================================================================================
# Module 7 (NEW): instructions= is APPENDED to the system prompt (T7.7) — never replaces it.
# ======================================================================================

def test_default_quill_instructions_are_appended_to_system_prompt(fake_model):
    """Omit instructions -> Quill's QUILL_INSTRUCTIONS default is APPENDED to the system prompt.
    The system prompt STILL contains the injected tool list (so it was appended, not replaced)."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert agent.instructions == QUILL_INSTRUCTIONS
    sp = agent.system_prompt
    assert "meticulous data analyst" in sp  # our instructions text landed in the prompt
    assert "Profile the dataset ONCE" in sp
    # ...AND the Jinja2-injected tool list is STILL there (proof: appended, not a replacement).
    assert "load_dataset" in sp
    assert "save_chart" in sp
    assert "final_answer" in sp


def test_custom_instructions_land_in_the_system_prompt(fake_model):
    """A caller's own brief is inserted into the system prompt verbatim (T7.7)."""
    marker = "QUILL_TEST_MARKER_specialise_for_finance"
    agent = build_quill(model=fake_model(["final_answer('ok')"]), instructions=marker)
    assert agent.instructions == marker
    assert marker in agent.system_prompt
    # The tool injections survive a custom brief too — it is appended, not a replacement.
    assert "load_dataset" in agent.system_prompt


def test_empty_instructions_give_the_bare_prompt_without_quill_default(fake_model):
    """instructions="" -> the BARE smolagents system prompt (the bench's baseline): Quill's
    default brief is NOT present, but the tool list still is (the prompt is intact)."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]), instructions="")
    assert agent.instructions == ""
    assert "meticulous data analyst" not in agent.system_prompt
    assert "load_dataset" in agent.system_prompt  # base prompt + injections still present


def test_build_quill_never_edits_the_raw_system_prompt_template(fake_model):
    """Anti-pattern guard (T7.7 / 06 §6): we use instructions=, we do NOT overwrite
    prompt_templates["system_prompt"]. The template still contains the Jinja2 placeholders that
    inject tools/imports/code-block tags at init — proof we left the raw template alone."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    raw = agent.prompt_templates["system_prompt"]
    assert "{{" in raw and "}}" in raw, "the raw template must still be a Jinja2 template"
    # Our instructions text is NOT hard-coded into the raw template (it is injected separately).
    assert "meticulous data analyst" not in raw


# ======================================================================================
# Module 7 (NEW): sharpened tool docstrings (principle 3 / T3.12) — signatures still frozen.
# ======================================================================================

def test_tool_docstrings_are_nonempty_and_sharpened():
    """The 'write better tools' principle: each tool's description is non-empty and names the
    supported formats / what it returns. Trivial offline check on tool.description (06 §6 step 7)."""
    assert load_dataset.description.strip()
    assert profile_dataframe.description.strip()
    assert save_chart().description.strip()

    # Sharpened specifics the model needs (formats + return contract), without changing schema.
    # (`tool.description` is the text BEFORE the Args: block; the extensions also appear there.)
    assert "CSV or Parquet" in load_dataset.description  # supported formats named up front
    assert "ONCE" in load_dataset.description  # the "load once" frugality nudge
    assert "ONCE" in profile_dataframe.description  # the "profile once" frugality nudge
    sc_desc = save_chart().description
    assert ".png" in sc_desc and "plt.show()" in sc_desc  # says what to return and what to avoid


def test_frozen_tool_signatures_unchanged_in_m7():
    """M7 only sharpens docstrings/print/ValueError — the FROZEN M3 signatures (name, inputs,
    output_type) are byte-for-byte the same. Re-pin them here so a docstring edit can't drift."""
    assert load_dataset.name == "load_dataset"
    assert load_dataset.output_type == "string"
    assert set(load_dataset.inputs) == {"path"}
    assert profile_dataframe.name == "profile_dataframe"
    assert set(profile_dataframe.inputs) == {"path"}
    sc = save_chart()
    assert sc.name == "save_chart"
    assert set(sc.inputs) == {"filename"}
    assert sc.inputs["filename"]["nullable"] is True


def test_tool_valueerror_messages_are_informative():
    """Good-tool rule 3 (T3.12): a bad path raises a ValueError that tells the agent how to fix
    it, so it self-corrects instead of crashing. The message names the path and is actionable."""
    with pytest.raises(ValueError, match="No file at"):
        load_dataset("does/not/exist.csv")
    # The message is actionable (mentions the working directory / fixing the path).
    try:
        load_dataset("nope.csv")
    except ValueError as exc:
        assert "working directory" in str(exc) or "path" in str(exc)


# ======================================================================================
# Module 7 (NEW): the step bench — quill/bench.py. OFFLINE via the FakeModel (zero tokens).
# Counts ActionSteps and PlanningSteps HONESTLY (06 §6: don't count "all steps").
# ======================================================================================

def test_count_steps_counts_action_and_planning_separately(fake_model):
    """count_steps reports ActionSteps and PlanningSteps separately so planning is never 'free'.
    With planning_interval=3 on a 2-action run, planning fires only at step 1 (step 2 is
    (2-1)%3 != 0), so we get exactly 1 plan + 2 actions; llm_calls is their sum."""
    agent = build_quill(
        model=fake_model([_plan(), "print('a')", "final_answer('done')"]),
        planning_interval=3,
        final_answer_checks=[],  # bench mechanic: opt out of the M8 report contract
    )
    agent.run(build_task(CSV, "two-step job"))
    counts = count_steps(agent)
    assert isinstance(counts, StepCount)
    assert counts.action_steps == 2          # print + final_answer
    assert counts.planning_steps == 1        # one plan at step 1 (interval=3 -> none at step 2)
    assert counts.llm_calls == 3             # ActionSteps + PlanningSteps


def test_run_and_count_runs_then_counts(fake_model):
    """run_and_count(agent, task) runs once and returns the StepCount (no planning here)."""
    agent = build_quill(model=fake_model(["print('x')", "final_answer('ok')"]),
                        final_answer_checks=[])
    counts = run_and_count(agent, build_task(CSV, "q"))
    assert counts.action_steps == 2
    assert counts.planning_steps == 0
    assert counts.llm_calls == 2


def test_format_report_shows_reduction_and_variance_caveat():
    """format_report renders the comparison block: a reduction % on ACTION steps (not all
    steps) and the honesty caveat that numbers vary. Pure function — no agent, no tokens."""
    baseline = StepCount(action_steps=11, planning_steps=0)
    improved = StepCount(action_steps=6, planning_steps=2)
    report = format_report(baseline, improved)
    assert "Baseline" in report and "Improved" in report
    assert "11 ActionSteps" in report
    assert "6 ActionSteps" in report
    assert "~45%" in report  # (11-6)/11 ~= 45%, computed on ActionSteps
    assert "vary" in report.lower()  # the non-determinism caveat is present


def test_bench_counts_action_steps_for_reduction_not_all_steps(fake_model):
    """The honesty rule pinned end-to-end (06 §6): a planned run has MORE total steps than its
    ActionStep count (the plan is extra), so a fair comparison uses action_steps. Here the
    'improved' agent has 2 ActionSteps + 1 PlanningStep = 3 total steps, but action_steps=2."""
    improved = build_quill(
        model=fake_model([_plan(), "print('work')", "final_answer('done')"]),
        planning_interval=3,  # plan at step 1 only on this 2-action run
        final_answer_checks=[],  # bench mechanic: opt out of the M8 report contract
    )
    improved.run(build_task(CSV, "q"))
    counts = count_steps(improved)
    total_steps = len([s for s in improved.memory.steps
                       if isinstance(s, (ActionStep, PlanningStep))])
    assert counts.action_steps == 2
    assert total_steps == 3  # would over-count the work if we naively used "all steps"
    assert counts.action_steps < total_steps


# ======================================================================================
# Module 6 (carried forward): the step callbacks — quill/callbacks.py (06 §3, T6.8)
# OFFLINE: build ActionSteps by hand, call the callbacks directly — zero tokens spent.
# ======================================================================================

def test_quill_callbacks_returns_the_two_callbacks():
    """quill_callbacks() is the list build_quill wires into step_callbacks=."""
    cbs = quill_callbacks()
    assert cbs == [prune_old_observations, log_step_cost]
    assert all(callable(cb) for cb in cbs)


def test_callback_signature_is_memory_step_then_agent():
    """The FROZEN smolagents callback signature is (memory_step, agent) — never inverted,
    never a single arg. CallbackRegistry calls a 2-arg callback as cb(memory_step, agent=...)."""
    import inspect

    for cb in (prune_old_observations, log_step_cost):
        params = list(inspect.signature(cb).parameters)
        assert params[:2] == ["memory_step", "agent"], f"{cb.__name__} has params {params}"


def test_prune_keeps_recent_steps_and_prunes_an_old_big_observation():
    """The core M6 demonstration, OFFLINE: a big observation on a step older than KEEP_LAST is
    replaced by the prune marker; the recent step's big observation is left untouched."""
    big = "x" * (MAX_OBS_CHARS + 500)
    old = _action_step(1, observations=big)      # will be > KEEP_LAST behind
    recent = _action_step(KEEP_LAST + 1, observations=big)  # exactly KEEP_LAST-1 behind current
    current = _action_step(KEEP_LAST + 2)        # the step that just finished
    agent = _FakeAgent([old, recent])            # current is NOT yet in memory (real behaviour)

    prune_old_observations(current, agent)

    assert old.observations == PRUNE_MARKER, "an old big dump should be pruned"
    assert recent.observations == big, "a recent observation must NOT be pruned"


def test_prune_leaves_small_observations_alone():
    """Below MAX_OBS_CHARS, an old observation is cheap and sometimes the only record — keep it."""
    small = "Loaded sales.csv: 108 rows x 6 columns."  # well under MAX_OBS_CHARS
    old = _action_step(1, observations=small)
    current = _action_step(1 + KEEP_LAST + 1)
    agent = _FakeAgent([old])

    prune_old_observations(current, agent)

    assert old.observations == small


def test_prune_does_not_touch_the_current_or_within_keep_last_steps():
    """Steps within KEEP_LAST of the current one stay verbatim (the model still needs them)."""
    big = "y" * (MAX_OBS_CHARS + 1)
    within = _action_step(5, observations=big)
    current = _action_step(5 + KEEP_LAST - 1)  # within KEEP_LAST of `within`
    agent = _FakeAgent([within])

    prune_old_observations(current, agent)

    assert within.observations == big


def test_prune_ignores_non_actionstep_current():
    """A non-ActionStep memory_step (e.g. a TaskStep/PlanningStep) is a no-op for pruning."""
    from smolagents import TaskStep

    big = "z" * (MAX_OBS_CHARS + 1)
    old = _action_step(1, observations=big)
    agent = _FakeAgent([old])

    prune_old_observations(TaskStep(task="hi"), agent)  # must not raise, must not prune

    assert old.observations == big


def test_prune_is_idempotent():
    """Running the callback twice does not double-mark or re-grow the observation."""
    big = "q" * (MAX_OBS_CHARS + 10)
    old = _action_step(1, observations=big)
    current = _action_step(1 + KEEP_LAST + 1)
    agent = _FakeAgent([old])

    prune_old_observations(current, agent)
    prune_old_observations(current, agent)

    assert old.observations == PRUNE_MARKER


def test_log_step_cost_is_quiet_and_safe_when_token_usage_is_none(capsys):
    """token_usage is legitimately None (offline model / pre-model error). The callback must
    NOT raise and must NOT invent a cost — no silent try/except, just an `is not None` test."""
    step = _action_step(1, token_usage=None)
    log_step_cost(step, agent=None)  # must not raise
    assert "tokens" not in capsys.readouterr().out


def test_log_step_cost_prints_input_plus_output_when_usage_present(capsys):
    """With a TokenUsage, the callback prints one `step N: <in>+<out> tokens` line."""
    step = _action_step(3, token_usage=TokenUsage(input_tokens=812, output_tokens=97))
    log_step_cost(step, agent=None)
    out = capsys.readouterr().out
    assert "step 3" in out
    assert "812+97 tokens" in out
    assert "total 909" in out


def test_log_step_cost_ignores_non_actionstep():
    """A non-ActionStep is a no-op (only ActionStep carries token_usage we report)."""
    from smolagents import PlanningStep

    plan = PlanningStep(
        model_input_messages=[],
        plan="step 1, step 2",
        model_output_message=None,
        timing=Timing(start_time=0.0),
    )
    log_step_cost(plan, agent=None)  # must not raise


# ======================================================================================
# Module 6 (NEW): build_quill wires step_callbacks; they FIRE during a fake-model run.
# ======================================================================================

def test_build_quill_wires_the_step_callbacks(fake_model):
    """build_quill registers quill_callbacks() in the agent's CallbackRegistry (06 §6 step 4).
    We assert the callbacks are reachable for ActionStep — the registry keys by step type."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    registered = agent.step_callbacks._callbacks.get(ActionStep, [])
    assert prune_old_observations in registered
    assert log_step_cost in registered


def test_callbacks_fire_during_a_two_step_run_and_prune_the_old_dump(fake_model):
    """End-to-end OFFLINE (fake model, no token): a run where step 1 prints a big
    DataFrame-sized dump and the run finishes a couple steps later. The wired
    prune_old_observations FIRES each step, so by the time the run ends step 1's big
    observation is pruned in memory (token saving achieved). (The fake model reports no
    token_usage, so log_step_cost stays quiet here — its printing is covered by the dedicated
    unit test above; this test proves the PRUNING half fires through the real loop.)"""
    big_dump = "col_a,col_b\n" + ("1,2\n" * 2000)  # a fat print(), well over MAX_OBS_CHARS
    script = [
        f"print({big_dump!r})",   # step 1: emits the big observation
        "print('small follow-up')",  # step 2: small
        "final_answer('done')",   # step 3: finish (now > KEEP_LAST behind step 1)
    ]
    agent = build_quill(model=fake_model(script), final_answer_checks=[])
    out = agent.run(build_task(CSV, "anything"))
    assert out == "done"

    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    assert len(action_steps) == 3, "expected exactly the 3 scripted ActionSteps"

    # The OLD big dump (step 1) is now pruned out of memory (token saving achieved): proof the
    # callback ran inside the loop and mutated agent.memory.steps.
    step_one = next(s for s in action_steps if s.step_number == 1)
    assert step_one.observations == PRUNE_MARKER
    # The most recent step's small observation is untouched.
    step_two = next(s for s in action_steps if s.step_number == 2)
    assert PRUNE_MARKER not in (step_two.observations or "")


def test_callback_records_each_action_step_number(fake_model):
    """A custom recording callback wired via step_callbacks fires once per ActionStep — proof
    the (memory_step, agent) hook actually runs inside the loop (recorded == [1, 2])."""
    recorded: list[int] = []

    def record(memory_step, agent):
        if isinstance(memory_step, ActionStep):
            recorded.append(memory_step.step_number)

    agent = build_quill(model=fake_model(["print('hi')", "final_answer('ok')"]),
                        final_answer_checks=[])
    agent.step_callbacks._callbacks.setdefault(ActionStep, []).append(record)
    agent.run(build_task(CSV, "anything"))
    assert recorded == [1, 2]


# ======================================================================================
# Module 6 (NEW): multi-turn via reset=False — memory CONTINUES across runs (T6.7)
# ======================================================================================

def test_reset_false_continues_memory_more_steps_than_a_single_run(fake_model):
    """The headline behaviour: run twice on ONE agent with reset=False and memory keeps the
    first run's steps — so the second run leaves MORE steps in memory than a single run would."""
    # First run: 2 ActionSteps (a print, then final_answer). M6 mechanic: opt out of M8 checks.
    agent = build_quill(model=fake_model(["print('turn 1')", "final_answer('a1')"]),
                        final_answer_checks=[])
    agent.run(build_task(CSV, "Q1"))
    steps_after_one_run = len([s for s in agent.memory.steps if isinstance(s, ActionStep)])
    assert steps_after_one_run == 2

    # Second run on the SAME agent, reset=False: memory is NOT wiped, it continues.
    agent.model = fake_model(["final_answer('a2')"])
    out2 = agent.run(build_task(CSV, "Q2"), reset=False)
    assert out2 == "a2"
    steps_after_two_runs = len([s for s in agent.memory.steps if isinstance(s, ActionStep)])
    assert steps_after_two_runs > steps_after_one_run
    assert steps_after_two_runs == 3  # 2 from turn 1 + 1 from turn 2


def test_reset_true_default_wipes_memory_between_runs(fake_model):
    """The default reset=True clears memory: a second default run does NOT accumulate steps."""
    agent = build_quill(model=fake_model(["print('turn 1')", "final_answer('a1')"]),
                        final_answer_checks=[])
    agent.run(build_task(CSV, "Q1"))
    assert len([s for s in agent.memory.steps if isinstance(s, ActionStep)]) == 2

    agent.model = fake_model(["final_answer('a2')"])
    agent.run(build_task(CSV, "Q2"))  # reset=True by default -> wipes turn 1
    assert len([s for s in agent.memory.steps if isinstance(s, ActionStep)]) == 1


def test_run_multi_turn_reuses_one_agent_and_grows_memory(fake_model):
    """run_multi_turn(csv, q1, q2, agent=...) runs turn 1 (reset=True) then turn 2
    (reset=False) on ONE agent and returns it with BOTH turns in memory."""
    agent = build_quill(
        model=fake_model([
            _load(CSV) + "\nprint(df.head().to_string())",  # turn 1: load + big-ish dump
            "final_answer('turn 1 answer')",
            "final_answer('turn 2 answer: reused the loaded df')",  # turn 2 reuses memory
        ]),
        final_answer_checks=[],  # M6 multi-turn mechanic: opt out of the M8 report contract
    )
    returned = run_multi_turn(CSV, "Which category grew fastest?",
                              "Now exclude 2020.", agent=agent)
    assert returned is agent
    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    # turn 1: 2 ActionSteps, turn 2: 1 ActionStep, all kept in one memory.
    assert len(action_steps) == 3


# ======================================================================================
# Module 5 (carried forward): the sandbox policy — quill/sandbox.py, the FROZEN contract (06 §2)
# ======================================================================================

def test_resolve_executor_defaults_to_local():
    """No QUILL_EXECUTOR set -> the safe-for-dev default 'local' (instant, free; NOT a
    security sandbox). The default is the documented one."""
    executor_type, imports = resolve_executor()
    assert executor_type == "local"
    assert DEFAULT_EXECUTOR == "local"
    assert isinstance(imports, list)


def test_resolve_executor_supported_set_is_local_docker_e2b():
    """Frozen contract: QUILL_EXECUTOR is validated against exactly {local, docker, e2b}.
    'wasm' is intentionally NOT in the set (removed from smolagents in 1.26.0)."""
    assert SUPPORTED_EXECUTORS == ("local", "docker", "e2b")
    assert "wasm" not in SUPPORTED_EXECUTORS


@pytest.mark.parametrize("value", ["local", "docker", "e2b"])
def test_resolve_executor_accepts_each_supported_value(monkeypatch, value):
    monkeypatch.setenv("QUILL_EXECUTOR", value)
    executor_type, _ = resolve_executor()
    assert executor_type == value


def test_resolve_executor_is_case_insensitive_and_trims(monkeypatch):
    monkeypatch.setenv("QUILL_EXECUTOR", "  Docker  ")
    executor_type, _ = resolve_executor()
    assert executor_type == "docker"


def test_resolve_executor_rejects_unknown(monkeypatch):
    """We fail LOUD on an unknown executor instead of silently dropping the sandbox."""
    monkeypatch.setenv("QUILL_EXECUTOR", "bogus")
    with pytest.raises(ValueError, match="Unknown QUILL_EXECUTOR"):
        resolve_executor()


def test_resolve_executor_rejects_removed_wasm(monkeypatch):
    """The banner ban of this module: executor_type='wasm' was removed in 1.26.0. A stale
    QUILL_EXECUTOR=wasm must be rejected (and the error mentions it), never silently honoured."""
    monkeypatch.setenv("QUILL_EXECUTOR", "wasm")
    with pytest.raises(ValueError, match="wasm"):
        resolve_executor()


def test_authorized_imports_are_the_frozen_least_privilege_list():
    """Frozen contract (06 §2): the import lock is EXACTLY this minimal list, never '*'."""
    _, imports = resolve_executor()
    assert imports == ["pandas", "numpy", "matplotlib.*", "json", "statistics"]
    assert QUILL_AUTHORIZED_IMPORTS == ["pandas", "numpy", "matplotlib.*", "json", "statistics"]
    assert "*" not in imports


def test_resolve_executor_returns_a_fresh_list_copy():
    """The returned import list is a COPY, so a caller cannot mutate the frozen constant."""
    _, imports = resolve_executor()
    imports.append("os")  # try to poison it
    _, again = resolve_executor()
    assert "os" not in again
    assert again == ["pandas", "numpy", "matplotlib.*", "json", "statistics"]


def test_quill_imports_is_the_sandbox_list_one_owner():
    """QUILL_IMPORTS (importable from quill.agent for back-compat) IS the sandbox list now —
    one owner (sandbox.py), so the lock cannot drift between modules."""
    assert QUILL_IMPORTS is QUILL_AUTHORIZED_IMPORTS


# ======================================================================================
# Module 5 (NEW): build_quill wires the executor + import lock from resolve_executor()
# ======================================================================================

def test_build_quill_uses_local_executor_by_default(fake_model):
    """Default QUILL_EXECUTOR -> the agent runs on the in-process LocalPythonExecutor."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert agent.executor_type == "local"
    assert type(agent.python_executor).__name__ == "LocalPythonExecutor"


def test_build_quill_passes_the_executor_type_from_env(monkeypatch, fake_model):
    """QUILL_EXECUTOR flows through build_quill to CodeAgent.executor_type. We assert the
    attribute WITHOUT constructing a remote executor (docker would build a container): we set
    'local' here; the real docker wiring is covered by the `sandbox`-marked tests."""
    monkeypatch.setenv("QUILL_EXECUTOR", "local")
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert agent.executor_type == "local"


def test_build_quill_locks_imports_no_wildcard(fake_model):
    """Frozen contract: build_quill's agent has the minimal import list, never '*'."""
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert "*" not in agent.additional_authorized_imports
    for imp in ("pandas", "numpy", "matplotlib.*", "json", "statistics"):
        assert imp in agent.additional_authorized_imports


def test_build_quill_effective_authorized_imports_union_base_modules(fake_model):
    """agent.authorized_imports is the EFFECTIVE union: BASE_BUILTIN_MODULES ∪ our additions.
    So 'os' is NOT there but our additions (pandas, matplotlib.*) and base ones (statistics,
    datetime) are. (additional_authorized_imports is what we PASS; authorized_imports is the
    computed union — two different names, 06 §6.)"""
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    eff = set(agent.authorized_imports)
    assert {"pandas", "numpy", "matplotlib.*", "json"} <= eff
    assert {"statistics", "datetime", "math"} <= eff  # from BASE_BUILTIN_MODULES
    assert "os" not in eff
    assert "subprocess" not in eff
    assert "*" not in eff


def test_build_quill_is_a_context_manager(fake_model):
    """The agent supports `with build_quill() as agent:` (deterministic sandbox cleanup) and
    exposes cleanup(). For a LOCAL agent cleanup is a no-op; for Docker/E2B it tears the
    container down."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert hasattr(agent, "__enter__")
    assert hasattr(agent, "__exit__")
    assert hasattr(agent, "cleanup")
    with build_quill(model=fake_model(["final_answer('ok')"])) as ctx_agent:
        assert isinstance(ctx_agent, CodeAgent)


# ======================================================================================
# Module 5 (NEW): the local executor BLOCKS a dangerous import and a runaway loop
# ======================================================================================

def test_dangerous_import_is_blocked_on_local_executor(fake_model):
    """The core M5 demonstration: a fake_model emitting `import os` on a LOCAL-executor agent
    is BLOCKED — the import never succeeds, os.getcwd() never reaches final_answer, and the
    run cannot end in 'success' with the os result. The error is captured in the step (the
    agent does NOT crash the host process)."""
    attack = "import os\nfinal_answer(os.getcwd())"
    agent = build_quill(model=fake_model([attack]))
    result = agent.run(build_task(CSV, "exfiltrate"), max_steps=2, return_full_result=True)

    # The dangerous import did NOT succeed: the run did not return the os.getcwd() value.
    assert result.state != "success"
    assert result.output != os.getcwd()

    # The block is visible as a captured step error mentioning the unauthorized import.
    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    errored = [s for s in action_steps if s.error is not None]
    assert errored, "expected the import to be captured as a step error"
    msg = str(errored[0].error)
    assert "os" in msg
    assert ("not allowed" in msg or "unauthorized" in msg.lower()
            or "InterpreterError" in msg)


def test_dangerous_import_raises_interpretererror_through_the_executor(fake_model):
    """Drive the same attack straight through the agent's executor: `import os` raises
    InterpreterError (a ValueError subclass) — the AST allow-list refuses it before it runs."""
    agent = build_quill(model=fake_model(["final_answer('noop')"]))
    with pytest.raises(InterpreterError, match="os"):
        agent.python_executor("import os\nos.system('echo pwned')\n")
    assert issubclass(InterpreterError, ValueError)


def test_subprocess_import_is_also_blocked(fake_model):
    """Not just os: subprocess is equally outside the allow-list."""
    agent = build_quill(model=fake_model(["final_answer('noop')"]))
    with pytest.raises(InterpreterError):
        agent.python_executor("import subprocess\nsubprocess.run(['ls'])\n")


def test_runaway_loop_is_cut_off_by_the_cap(fake_model):
    """A `while True` loop is stopped by the executor's iteration cap (MAX_WHILE_ITERATIONS =
    1_000_000 as of 1.26.0), raising InterpreterError — the loop never runs forever."""
    agent = build_quill(model=fake_model(["final_answer('noop')"]))
    with pytest.raises(InterpreterError, match="iterations"):
        agent.python_executor("x = 0\nwhile True:\n    x += 1\n")


def test_authorized_imports_still_work(fake_model):
    """The lock is least-privilege, not a wall: pandas/numpy/json DO import. (We call the
    executor directly, so we avoid host builtins like print/int — those are injected during a
    full agent.run(), not a raw executor call — and only exercise the IMPORTS, which is the
    point: authorized imports succeed where os/subprocess raise.)"""
    agent = build_quill(model=fake_model(["final_answer('noop')"]))
    # No exception: these are all in the authorized list (import + attribute/operator use only).
    agent.python_executor(
        "import pandas as pd\nimport numpy as np\nimport json\nimport statistics\n"
        "arr = np.array([1, 2, 3])\ntotal = arr.sum()\n"
    )


def test_authorized_imports_work_through_a_full_run(fake_model):
    """And through a full agent.run() (where builtins are available): a pandas/numpy/json
    snippet computes and answers — the lock permits exactly what a data analysis needs."""
    script = (
        _load(CSV)
        + "\nimport numpy as np"
        + "\nimport json"
        + "\nn = int(np.array([len(df)]).sum())"
        + "\nfinal_answer(json.dumps({'rows': n}))"
    )
    agent = build_quill(model=fake_model([script]), final_answer_checks=[])
    out = agent.run(build_task(CSV, "How many rows, as JSON?"))
    assert out == '{"rows": 108}'


# ======================================================================================
# Module 4 (carried forward): the model factory — the FROZEN model contract (06 §2)
# ======================================================================================

def test_default_model_id_is_the_explicit_coder_pin():
    assert DEFAULT_MODEL_ID == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert Settings.DEFAULT_MODEL_ID == DEFAULT_MODEL_ID


def test_make_model_signature_is_frozen():
    import inspect

    sig = inspect.signature(make_model)
    params = sig.parameters
    assert "role" in params
    assert params["role"].default == "analyst"
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_make_model_default_is_inferenceclientmodel():
    model = make_model("analyst")
    assert isinstance(model, InferenceClientModel)
    assert isinstance(model, Model)
    assert model.model_id == DEFAULT_MODEL_ID


def test_make_model_default_backend_is_hf():
    assert Settings.MODEL_BACKEND == "hf"
    assert Settings.DEFAULT_BACKEND == "hf"


def test_make_model_swaps_to_litellm(monkeypatch):
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "litellm")
    monkeypatch.setenv("QUILL_MODEL_ID", "gpt-4o")
    model = make_model("analyst")
    assert isinstance(model, LiteLLMModel)
    assert not isinstance(model, InferenceClientModel)
    assert model.model_id == "gpt-4o"


def test_make_model_local_backend_uses_ollama_via_litellm(monkeypatch):
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "local")
    model = make_model("analyst")
    assert isinstance(model, LiteLLMModel)
    assert model.model_id == DEFAULT_LOCAL_MODEL_ID
    assert model.model_id.startswith("ollama_chat/")
    assert OLLAMA_NUM_CTX >= 8192


def test_make_model_local_respects_explicit_model_id(monkeypatch):
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "local")
    monkeypatch.setenv("QUILL_MODEL_ID", "ollama_chat/qwen2.5-coder")
    model = make_model("analyst")
    assert isinstance(model, LiteLLMModel)
    assert model.model_id == "ollama_chat/qwen2.5-coder"


def test_make_model_backend_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "LiteLLM")
    monkeypatch.setenv("QUILL_MODEL_ID", "gpt-4o")
    assert isinstance(make_model("analyst"), LiteLLMModel)


def test_make_model_rejects_unknown_backend(monkeypatch):
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "bogus")
    with pytest.raises(ValueError, match="Unknown QUILL_MODEL_BACKEND"):
        make_model("analyst")


def test_make_model_forwards_overrides(monkeypatch):
    model = make_model("analyst", provider="together")
    assert isinstance(model, InferenceClientModel)
    assert model.model_id == DEFAULT_MODEL_ID


def test_settings_rereads_env_at_access_time(monkeypatch):
    assert Settings.MODEL_BACKEND == "hf"
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "litellm")
    monkeypatch.setenv("QUILL_MODEL_ID", "anthropic/claude-3-5-sonnet-latest")
    assert Settings.MODEL_BACKEND == "litellm"
    assert Settings.MODEL_ID == "anthropic/claude-3-5-sonnet-latest"


def test_build_quill_uses_make_model_when_no_model_passed():
    agent = build_quill()
    assert isinstance(agent.model, InferenceClientModel)
    assert agent.model.model_id == DEFAULT_MODEL_ID


def test_build_quill_swap_flows_through_env(monkeypatch):
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "litellm")
    monkeypatch.setenv("QUILL_MODEL_ID", "gpt-4o")
    agent = build_quill()
    assert isinstance(agent.model, LiteLLMModel)
    assert agent.model.model_id == "gpt-4o"


def test_build_quill_still_accepts_an_injected_model(fake_model):
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert isinstance(agent, CodeAgent)
    assert agent.model.model_id == "fake/deterministic"


# ======================================================================================
# Module 4 (carried forward): token cost in plain sight via Monitor / TokenUsage
# ======================================================================================

def test_agent_has_a_monitor_with_total_token_counts(fake_model):
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert isinstance(agent.monitor, Monitor)
    usage = agent.monitor.get_total_token_counts()
    assert isinstance(usage, TokenUsage)
    assert hasattr(usage, "input_tokens")
    assert hasattr(usage, "output_tokens")
    assert hasattr(usage, "total_tokens")
    assert not hasattr(agent, "logs")  # removed in 1.21.0


def test_token_usage_total_is_input_plus_output():
    usage = TokenUsage(input_tokens=8142, output_tokens=1203)
    assert usage.total_tokens == 9345


def test_run_cost_accessor_after_a_run(fake_model):
    agent = build_quill(model=fake_model([_load(CSV) + "\nfinal_answer('Team')"]))
    agent.run(build_task(CSV, "Top category?"))
    usage = agent.monitor.get_total_token_counts()
    assert isinstance(usage, TokenUsage)
    assert usage.total_tokens >= 0


def test_format_cost_line_shape():
    line = _format_cost(TokenUsage(input_tokens=8142, output_tokens=1203))
    assert "input tokens: 8,142" in line
    assert "output tokens: 1,203" in line
    assert "total: 9,345" in line


def test_run_main_prints_backend_model_and_cost(fake_model, monkeypatch, capsys):
    """The M8 CLI (06 §6 observable result): `python -m quill "<question>" --data <csv>` builds a
    QuillReport, validates it, and renders Markdown with the cost line. The fake-model agent builds
    a complete report (chart_path set) so the default checks accept it."""
    import quill.run as run_mod

    report_code = (
        "rep = QuillReport(question='Which category grew fastest?', "
        "findings=['Team grew fastest [n/a]'], chart_paths=['outputs/category_growth.png'])\n"
        "final_answer(rep)"
    )
    monkeypatch.setattr(
        run_mod, "build_quill",
        lambda: build_quill(model=fake_model([report_code])),
    )
    # New CLI signature: question is positional, the CSV comes from --data.
    code = run_main(["Which category grew fastest?", "--data", CSV])
    assert code == 0
    out = capsys.readouterr().out
    assert "[Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct" in out
    assert "[Quill] Run cost — input tokens:" in out
    # The report rendered as Markdown: the title and a chart path appear.
    assert "# Which category grew fastest?" in out
    assert "Team grew fastest" in out


# ======================================================================================
# Module 2/3 (carried forward): the agent loop + the toolbox still work, contracts unchanged
# ======================================================================================

def test_build_quill_is_a_codeagent(fake_model):
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert isinstance(agent, CodeAgent)


def test_quill_imports_locked_no_wildcard(fake_model):
    """Frozen contract (06 §2): additional_authorized_imports is a minimal explicit list and
    is NEVER the '*' wildcard. As of M5 it is the frozen 5-item least-privilege list."""
    assert QUILL_IMPORTS == ["pandas", "numpy", "matplotlib.*", "json", "statistics"]
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert "*" not in agent.additional_authorized_imports
    for imp in ("pandas", "numpy", "matplotlib.*"):
        assert imp in agent.additional_authorized_imports


def test_max_steps_is_capped_low(fake_model):
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert agent.max_steps == 8


def test_quill_answers_a_csv_question_offline(fake_model):
    script = (
        _load(CSV)
        + "\ntotals = df.groupby('category')['net_rev'].sum()"
        + "\nprint(totals)"
        + "\nfinal_answer('Team (highest net revenue)')"
    )
    # M2/M3 agent-loop mechanic: opt out of the M8 report contract (bare string is fine here).
    agent = build_quill(model=fake_model([script]), final_answer_checks=[])
    out = agent.run(build_task(CSV, "Which category has the highest revenue?"))
    assert out == "Team (highest net revenue)"


def test_pandas_actually_imports_in_the_sandbox(fake_model):
    script = _load(CSV) + "\nfinal_answer(int(len(df)))"
    agent = build_quill(model=fake_model([script]), final_answer_checks=[])
    out = agent.run(build_task(CSV, "How many rows?"))
    assert out == 108


def test_return_full_result_gives_a_runresult(fake_model):
    script = _load(CSV) + "\nfinal_answer('Team')"
    agent = build_quill(model=fake_model([script]), final_answer_checks=[])
    result = agent.run(build_task(CSV, "Top category?"), return_full_result=True)
    assert isinstance(result, RunResult)
    assert result.state == "success"
    assert result.output == "Team"
    assert isinstance(result.steps, list)
    assert all(isinstance(s, dict) for s in result.steps)
    assert hasattr(result, "token_usage")
    assert hasattr(result, "timing")


def test_agent_self_corrects_after_an_error(fake_model):
    bad = _load(CSV) + "\nprint(df['Catgory'].unique())"  # typo -> KeyError
    good = (
        _load(CSV)
        + "\nprint(df['category'].unique())"
        + "\nfinal_answer('Team')"
    )
    agent = build_quill(model=fake_model([bad, good]), final_answer_checks=[])
    out = agent.run(build_task(CSV, "Which category grew fastest?"))
    assert out == "Team"

    action_steps = [s for s in agent.memory.steps if isinstance(s, ActionStep)]
    assert len(action_steps) >= 2
    assert action_steps[0].error is not None
    assert action_steps[-1].error is None
    assert any(s.is_final_answer for s in action_steps)


def test_max_steps_error_when_final_answer_never_called(fake_model):
    looping = _load(CSV) + "\nprint('still thinking...')"
    agent = build_quill(model=fake_model([looping]))
    result = agent.run(build_task(CSV, "Loop forever"), max_steps=2, return_full_result=True)
    assert result.state == "max_steps_error"


def test_trajectory_is_readable_via_memory_steps(fake_model):
    agent = build_quill(model=fake_model([_load(CSV) + "\nfinal_answer('ok')"]),
                        final_answer_checks=[])
    agent.run(build_task(CSV, "anything"))
    assert hasattr(agent, "memory")
    assert not hasattr(agent, "logs")  # removed in 1.21.0
    assert any(isinstance(s, ActionStep) for s in agent.memory.steps)
    agent.replay()  # must not raise


def test_data_tools_are_validated_tool_instances():
    for t in (load_dataset, profile_dataframe, save_chart()):
        assert isinstance(t, Tool)
        t.validate_arguments()


def test_frozen_tool_signatures_and_schema():
    assert load_dataset.name == "load_dataset"
    assert load_dataset.output_type == "string"
    assert set(load_dataset.inputs) == {"path"}
    assert load_dataset.inputs["path"]["type"] == "string"

    assert profile_dataframe.name == "profile_dataframe"
    assert profile_dataframe.output_type == "string"
    assert set(profile_dataframe.inputs) == {"path"}
    assert profile_dataframe.inputs["path"]["type"] == "string"

    sc = save_chart()
    assert sc.name == "save_chart"
    assert sc.output_type == "string"
    assert sc.inputs["filename"]["nullable"] is True


def test_tools_have_descriptions_injected_into_the_prompt():
    assert load_dataset.description.strip()
    assert "summary" in load_dataset.description.lower()
    assert profile_dataframe.description.strip()
    assert save_chart().description.strip()


def test_save_chart_lazy_setup_runs_once():
    sc = save_chart()
    assert sc.is_initialized is False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])
    sc("setup_probe")
    assert sc.is_initialized is True


def test_save_chart_writes_a_png_and_returns_the_path():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = save_chart()
    plt.figure()
    plt.bar(["Free", "Pro", "Team"], [1, 2, 3])
    plt.title("smoke chart")

    out = sc("smoke_test_chart")
    assert isinstance(out, str)
    assert out.endswith(".png")
    assert os.path.exists(out), f"expected a saved PNG at {out}"
    os.remove(out)


def test_save_chart_errors_cleanly_with_no_figure():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")
    sc = save_chart()
    with pytest.raises(ValueError, match="draw a chart"):
        sc("nothing_drawn")


def test_profile_dataframe_returns_a_nonempty_summary():
    out = profile_dataframe(CSV)
    assert isinstance(out, str)
    assert out.strip()
    assert "category" in out
    assert "net_rev" in out
    assert "108 rows" in out


def test_load_dataset_returns_a_summary_string():
    out = load_dataset(CSV)
    assert isinstance(out, str)
    assert "108 rows" in out
    assert "category" in out


def test_data_tools_raise_valueerror_on_bad_path():
    with pytest.raises(ValueError, match="No file at"):
        load_dataset("does/not/exist.csv")
    with pytest.raises(ValueError, match="No file at"):
        profile_dataframe("does/not/exist.csv")


def test_data_tools_reject_unsupported_format(tmp_path):
    bad = tmp_path / "data.txt"
    bad.write_text("not a table")
    with pytest.raises(ValueError, match="Unsupported format"):
        load_dataset(str(bad))


def test_quill_is_wired_with_the_expected_toolbox(fake_model):
    """The manager's toolbox = the DATA tools + final_answer. As of M10 the web tools are NOT on
    the manager (they moved into the web_researcher sub-agent — context isolation). The manager
    reaches the web by delegating, not by owning web_search/visit_webpage."""
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    names = set(agent.tools)
    assert {"load_dataset", "profile_dataframe", "save_chart"} <= names
    assert "web_search" not in names  # M10: web tools live on the web_researcher sub-agent
    assert "visit_webpage" not in names
    assert "final_answer" in names
    assert "python_interpreter" not in names
    # The web tools ARE present — one level down, inside the managed web_researcher sub-agent.
    sub_names = set(agent.managed_agents["web_researcher"].tools)
    assert {"web_search", "visit_webpage"} <= sub_names


def test_toolbox_is_runtime_mutable_a_dict_keyed_by_name(fake_model):
    from smolagents import VisitWebpageTool

    agent = build_quill(model=fake_model(["final_answer('done')"]))
    extra = VisitWebpageTool()
    agent.tools[extra.name] = extra
    assert agent.tools[extra.name] is extra


def test_quill_uses_its_tools_end_to_end_offline(fake_model):
    script = (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        f"summary = profile_dataframe({CSV!r})\n"
        "print(summary)\n"
        f"df = pd.read_csv({CSV!r})\n"
        "totals = df.groupby('category')['net_rev'].sum()\n"
        "totals.plot(kind='bar')\n"
        "path = save_chart('category_revenue')\n"
        "final_answer(f'Team grew fastest. Chart saved at {path}')"
    )
    # Carried-forward M3 toolbox test (string answer): opt out of the M8 report contract — the
    # dedicated M8 tests below assert the QuillReport-and-checks path end to end.
    agent = build_quill(model=fake_model([script]), final_answer_checks=[])
    out = agent.run(build_task(CSV, "Which category grew fastest? Back it with a chart."))
    assert "Team" in str(out)
    assert ".png" in str(out)
    saved = str(out).split("Chart saved at ")[-1].strip()
    assert os.path.exists(saved), f"expected the saved chart at {saved}"
    os.remove(saved)


# ======================================================================================
# Sandbox (skipped if Docker absent; QUILL_EXECUTOR=docker uv run pytest -m sandbox to run).
# Budget: 1-2 Docker runs. First run may take ~180s to build the jupyter-kernel image.
# Always cleans up via the context manager (no dangling containers).
# ======================================================================================

@pytest.mark.sandbox
def test_codeagent_runs_a_trivial_calc_inside_docker(fake_model):
    """A bare CodeAgent(executor_type='docker') + FakeModel runs a trivial calc INSIDE a real
    Docker container and returns the value. We build a bare agent (no pandas/matplotlib imports
    to pip-install) so the test exercises the sandbox boundary itself. Cleanup is guaranteed by
    the context manager (no dangling container)."""
    with CodeAgent(
        tools=[],
        model=fake_model(["final_answer(6 * 7)"]),
        executor_type="docker",
        max_steps=2,
    ) as agent:
        assert agent.executor_type == "docker"
        assert type(agent.python_executor).__name__ == "DockerExecutor"
        out = agent.run("Compute 6 times 7.")
        assert out == 42


@pytest.mark.sandbox
def test_build_quill_wires_a_real_docker_executor(monkeypatch, fake_model):
    """QUILL_EXECUTOR=docker -> build_quill CONSTRUCTS a real DockerExecutor (container starts,
    Approach 1 is wired) and the context manager tears it down. We assert the wiring, not a
    full toolbox run — see the next test for the real Approach-1 caveat about Quill's tools.

    M10 note (T10.7): a remote executor + managed_agents RAISES (Approach 1 ships no secrets into
    the sandbox), so to test the docker executor itself we build a SOLO manager (managed_agents=[]).
    Running the actual TEAM in a sandbox is Approach 2, the capstone (Module 15)."""
    monkeypatch.setenv("QUILL_EXECUTOR", "docker")
    with build_quill(model=fake_model(["final_answer('ok')"]), managed_agents=[]) as agent:
        assert agent.executor_type == "docker"
        assert type(agent.python_executor).__name__ == "DockerExecutor"


@pytest.mark.sandbox
def test_quills_tool_tools_are_not_remotely_serializable_yet(monkeypatch, fake_model):
    """A REAL Approach-1 caveat, pinned as a test so it can't silently change: running
    build_quill() under a remote executor fails when smolagents tries to send Quill's @tool
    data tools into the container — `load_dataset` references the module-level `_read_table`
    helper, which `SimpleTool.to_dict()` (the remote-serialization path) rejects ('Name
    _read_table is undefined'). The construction + container are fine; the SEND of these tools
    is the blocker. (Making the tools self-contained for remote sending is a later concern;
    the bare-executor docker run above proves the sandbox boundary itself works.)

    M10 note (T10.7): we use a SOLO manager (managed_agents=[]) because a remote executor +
    managed_agents raises before we ever reach the tool-serialization path."""
    monkeypatch.setenv("QUILL_EXECUTOR", "docker")
    script = "import numpy as np\nfinal_answer(int(np.array([1, 2, 3, 4]).sum()))"
    with build_quill(model=fake_model([script]), managed_agents=[]) as agent:
        assert agent.executor_type == "docker"
        with pytest.raises(ValueError, match="SimpleTool validation failed"):
            agent.run("Sum 1..4.")


# ======================================================================================
# Module 9 LIVE: a REAL stdio MCP server (skipped by default). Needs QUILL_LIVE_TESTS=1 + uvx +
# HF_TOKEN + a built data/sales.db. Budget: ~1 real LLM run + 1 uvx subprocess. Skips cleanly if
# uvx / HF_TOKEN / sales.db are missing (06 §2 convention).
# ======================================================================================

@pytest.mark.live
def test_quill_live_connects_to_a_real_stdio_mcp_server():
    """The M9 promise with a REAL stdio MCP server (06 §6 observable result): Quill connects to
    `uvx mcp-server-sqlite --db-path data/sales.db`, gets its SQL tools as smolagents Tools, and
    answers a data question — combining an MCP tool with its own pandas/matplotlib, returning a
    validated QuillReport. Connection is opened/closed by run_with_mcp (a ToolCollection.from_mcp
    context manager) with trust_remote_code=True (stdio runs local code — the M5/T9.5 gate) and
    structured_output=False pinned. Skips cleanly if uvx / HF_TOKEN / sales.db are missing."""
    import shutil

    if shutil.which("uvx") is None:
        pytest.skip("live MCP test needs `uvx` (ships with uv) on PATH")
    if not os.environ.get("HF_TOKEN"):
        pytest.skip("live MCP test needs HF_TOKEN")
    if not os.path.exists(SALES_DB):
        # Build it on the fly so the live test is self-contained (still skips if data/ is absent).
        if not os.path.exists(CSV):
            pytest.skip("live MCP test needs data/sales.csv to build data/sales.db")
        build_sales_db(CSV, SALES_DB)

    params = data_mcp_server_params(db_path=SALES_DB)
    output = run_with_mcp(
        build_sql_task("Which product category has the highest total net_rev?"),
        server_parameters=params,
    )
    # Quill must return a validated QuillReport (its default checks are ON inside run_with_mcp).
    assert isinstance(output, QuillReport), "Quill must return a QuillReport from the MCP run"
    assert output.chart_paths, "the report must carry at least one saved chart"
    md = output.to_markdown()
    assert md.strip()


# ======================================================================================
# Live (skipped by default; QUILL_LIVE_TESTS=1 + HF_TOKEN). Budget: ~4 real LLM runs
# (multi-turn = 2 runs on one agent; the M7 live bench = 2 runs, baseline + improved).
# ======================================================================================

@pytest.mark.live
def test_quill_live_planning_inserts_a_real_planning_step():
    """M7 with a REAL model: planning_interval=3 makes Quill emit a real PlanningStep on a
    multi-step question. Budget: ~1 LLM run (a couple of action steps + at least one plan)."""
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    # Carried-forward M7 live test: opt out of the M8 report contract so a plain text answer is
    # accepted (the M8 live test below asserts the QuillReport path with a real model).
    with build_quill(planning_interval=DEFAULT_PLANNING_INTERVAL,
                     final_answer_checks=[]) as agent:
        result = agent.run(
            build_task(CSV, "Which category grew fastest from Q1 to Q4 2025? Back it with a chart."),
            return_full_result=True,
        )
    assert result.state == "success"
    assert any(isinstance(s, PlanningStep) for s in agent.memory.steps), \
        "planning_interval should insert at least one PlanningStep in a real run"


@pytest.mark.live
def test_quill_live_bench_runs_baseline_and_improved():
    """The module's measurement, live (06 §6): the bench runs Quill twice (baseline vs improved)
    and prints the comparison. We assert it completes and reports honest counts (action steps
    counted, planning steps tracked) — NOT a fixed reduction (numbers vary). Budget: ~2 LLM runs."""
    import quill.bench as bench

    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    task = build_task(CSV, "Which category grew fastest from Q1 to Q4 2025?")

    with build_quill(instructions="", final_answer_checks=[]) as baseline_agent:
        baseline = bench.run_and_count(baseline_agent, task)
    with build_quill(planning_interval=DEFAULT_PLANNING_INTERVAL,
                     final_answer_checks=[]) as improved_agent:
        improved = bench.run_and_count(improved_agent, task)

    assert baseline.action_steps >= 1
    assert improved.action_steps >= 1
    assert improved.planning_steps >= 1, "the improved config plans at least once"
    report = bench.format_report(baseline, improved)
    assert "Baseline" in report and "Improved" in report


@pytest.mark.live
def test_quill_live_run_reports_a_real_token_cost():
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    # make_model() default: InferenceClientModel(Qwen coder). Carried-forward M4 cost test:
    # opt out of the M8 report contract so a plain text answer is accepted.
    with build_quill(final_answer_checks=[]) as agent:
        task = build_task(
            CSV,
            "Which product category grew fastest from Q1 to Q4 2025? Back it with a saved chart.",
        )
        result = agent.run(task, return_full_result=True)
        usage = agent.monitor.get_total_token_counts()
    assert result.state == "success"
    assert str(result.output).strip()
    assert usage.total_tokens > 0, "expected a real run to report token usage > 0"


@pytest.mark.live
def test_quill_live_multi_turn_continues_memory_with_reset_false():
    """The M6 multi-turn promise with a REAL model: two questions on ONE agent, the second
    with reset=False. Memory carries turn 1 forward (so turn 2 has MORE steps in memory than
    turn 1 left), and the step callbacks have pruned/logged along the way. Budget: ~2 LLM runs."""
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    # Carried-forward M6 multi-turn live test: opt out of the M8 report contract.
    with build_quill(final_answer_checks=[]) as agent:
        agent.run(build_task(CSV, "Which category has the highest total net_rev?"))
        steps_after_turn_1 = len([s for s in agent.memory.steps if isinstance(s, ActionStep)])

        # reset=False: keep the loaded DataFrame + turn-1 findings in memory.
        out2 = agent.run(
            build_task(CSV, "Now answer the same question but exclude any rows from 2020."),
            reset=False,
        )
        steps_after_turn_2 = len([s for s in agent.memory.steps if isinstance(s, ActionStep)])

    assert str(out2).strip()
    assert steps_after_turn_2 > steps_after_turn_1, "reset=False should carry turn 1 forward"


@pytest.mark.live
def test_quill_live_returns_a_validated_quillreport():
    """The M8 promise with a REAL model (06 §6): Quill returns a validated QuillReport for a
    question that needs a chart — the default final_answer_checks accept it only when it has a
    saved chart (and a source if it went to the web). We assert the answer is a QuillReport with a
    non-empty chart_paths, the run succeeded, and to_markdown() renders. If a check rejected an
    intermediate attempt, that AgentError is visible in some ActionStep.error (self-correction).
    Budget: ~1 real LLM run (may take a couple of steps if it self-corrects)."""
    from quill.agent import build_report_task

    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    with build_quill() as agent:  # Quill's default checks are ON
        result = agent.run(
            build_report_task(CSV, "Which category grew fastest from Q1 to Q4 2025?"),
            return_full_result=True,
        )
    assert result.state == "success", "the validated report should be accepted (after any retries)"
    assert isinstance(result.output, QuillReport), "Quill must return a QuillReport"
    assert result.output.chart_paths, "the report must carry at least one saved chart"
    md = result.output.to_markdown()
    assert md.strip() and result.output.question in md


@pytest.mark.live
@pytest.mark.sandbox
def test_live_codeagent_in_docker_sandbox_approach_1():
    """The Approach-1 promise: a REAL model decides the code (locally), but every snippet runs
    inside a Docker container. We use a bare CodeAgent (no custom tools to send) with Quill's
    real model factory, so this exercises the model-local / code-remote split end to end.
    Needs both HF_TOKEN and Docker. Budget: ~1 real LLM run."""
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    with CodeAgent(
        tools=[],
        model=make_model(role="analyst"),
        executor_type="docker",
        max_steps=4,
    ) as agent:
        assert agent.executor_type == "docker"
        result = agent.run("What is 12 * 12? Use Python.", return_full_result=True)
    assert result.state == "success"


# ======================================================================================
# Module 10 LIVE: the REAL team (skipped by default; QUILL_LIVE_TESTS=1 + HF_TOKEN). The manager
# delegates to the web_researcher, which runs its OWN ReAct loop over the real web. Budget: 5-15
# LLM calls (manager loop + sub-agent loop, capped at max_steps=10). Needs network for the search.
# ======================================================================================

@pytest.mark.live
def test_quill_live_manager_delegates_to_the_web_researcher():
    """The M10 promise with a REAL model + the real web (06 §6 observable result): Quill (manager
    CodeAgent) is asked a question that needs external context, delegates to the web_researcher
    sub-agent, and returns a validated QuillReport. We assert the answer is a QuillReport with a
    chart and at least one Source (the web-backed claim must be cited — the default checks enforce
    it), and that the trajectory shows the manager calling web_researcher. Budget: 5-15 LLM calls."""
    from quill.agent import build_report_task

    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    with build_quill() as agent:  # Quill's default team + default checks are ON
        assert "web_researcher" in agent.managed_agents
        result = agent.run(
            build_report_task(
                CSV,
                "Is our Q3 churn (the 'churned' column) high versus the SaaS industry average? "
                "Use the web_researcher to find the industry benchmark.",
            ),
            return_full_result=True,
        )
    assert result.state == "success", "the validated cited report should be accepted"
    assert isinstance(result.output, QuillReport), "Quill must return a QuillReport"
    assert result.output.chart_paths, "the report must carry at least one saved chart"
    assert result.output.sources, "a web-backed report must cite at least one source [n]"
    # The manager really delegated: some step's code references the web_researcher call.
    delegated = any(
        "web_researcher" in (getattr(s, "code_action", None) or "")
        for s in agent.memory.steps if isinstance(s, ActionStep)
    )
    assert delegated, "the manager should have called web_researcher(...) in its code"
