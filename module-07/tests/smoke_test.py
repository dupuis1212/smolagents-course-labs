"""Module 7 smoke tests — Quill stops winging it: planning + sharpened prompt/tools.

Live call budget: ~3 real LLM runs (the M7 live bench runs Quill twice on one question;
carried-forward M6 multi-turn adds one more), well under 10 LLM calls (only when
QUILL_LIVE_TESTS=1 and HF_TOKEN is set). Sandbox budget: 1-2 Docker runs (marked `sandbox`,
skipped if Docker is absent; allow ~180s for the first image build). Everything else runs
OFFLINE with no network and no token: the agent-loop tests use the shared FakeModel (see
repo-root conftest.py) — including the planning ones, where the FakeModel scripts BOTH the
plan and the action so a real PlanningStep lands in agent.memory.steps without a token spent.

This file carries the Module 2/3/4/5/6 tests forward (the toolbox, the agent loop,
make_model(), the Monitor cost accessor, the sandbox policy and the callbacks/multi-turn still
work) and adds the Module 7 tests:
- build_quill(planning_interval=1) inserts a PlanningStep at step 1 (visible in
  agent.memory.steps) through a real fake-model run; the default (no planning_interval) inserts
  NONE; the trigger condition (step-1)%interval==0 is asserted on the source predicate;
- build_quill(instructions="...") APPENDS the text to agent.system_prompt (does NOT replace it
  — the tool list / authorized imports are still injected); omitting instructions uses Quill's
  QUILL_INSTRUCTIONS default; instructions="" gives the bare prompt;
- build_quill's signature is EXTENDED by ADD (planning_interval, instructions are keyword-only,
  both default to the smolagents default) — the M2-M6 call sites are unbroken;
- the data tools' docstrings are non-empty and sharpened (frozen signatures unchanged);
- quill.bench: count_steps / run_and_count count ActionSteps and PlanningSteps honestly, and
  format_report renders the reduction line (all offline, fake-model).

Run from the repo root: ``uv run pytest module-07/tests/``
"""
from __future__ import annotations

import inspect
import os
import pathlib
import sys

import pytest
from smolagents import (
    ActionStep,
    CodeAgent,
    InferenceClientModel,
    LiteLLMModel,
    Model,
    Monitor,
    PlanningStep,
    RunResult,
    TokenUsage,
    Tool,
)
from smolagents.local_python_executor import InterpreterError
from smolagents.memory import Timing

# Make THIS module's `quill` package importable when running from the repo root, even in the
# cumulative suite where earlier modules also ship a `quill` package. Every module-NN/ is a
# self-contained snapshot, so several dirs define a top-level `quill`; whichever is imported
# first would otherwise win in sys.modules. We prepend this module's dir and drop any cached
# `quill*` so this file always binds to module-07/quill.
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
    build_task,
    run_multi_turn,
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

CSV = str(MODULE_DIR / "data" / "sales.csv")
OUTPUTS = MODULE_DIR / "outputs"


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
    agent = build_quill(
        model=fake_model([_plan(), "final_answer('done')"]),
        planning_interval=1,
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
    agent = build_quill(model=fake_model(script), planning_interval=3)
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
    )
    agent.run(build_task(CSV, "two-step job"))
    counts = count_steps(agent)
    assert isinstance(counts, StepCount)
    assert counts.action_steps == 2          # print + final_answer
    assert counts.planning_steps == 1        # one plan at step 1 (interval=3 -> none at step 2)
    assert counts.llm_calls == 3             # ActionSteps + PlanningSteps


def test_run_and_count_runs_then_counts(fake_model):
    """run_and_count(agent, task) runs once and returns the StepCount (no planning here)."""
    agent = build_quill(model=fake_model(["print('x')", "final_answer('ok')"]))
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
    agent = build_quill(model=fake_model(script))
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

    agent = build_quill(model=fake_model(["print('hi')", "final_answer('ok')"]))
    agent.step_callbacks._callbacks.setdefault(ActionStep, []).append(record)
    agent.run(build_task(CSV, "anything"))
    assert recorded == [1, 2]


# ======================================================================================
# Module 6 (NEW): multi-turn via reset=False — memory CONTINUES across runs (T6.7)
# ======================================================================================

def test_reset_false_continues_memory_more_steps_than_a_single_run(fake_model):
    """The headline behaviour: run twice on ONE agent with reset=False and memory keeps the
    first run's steps — so the second run leaves MORE steps in memory than a single run would."""
    # First run: 2 ActionSteps (a print, then final_answer).
    agent = build_quill(model=fake_model(["print('turn 1')", "final_answer('a1')"]))
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
    agent = build_quill(model=fake_model(["print('turn 1')", "final_answer('a1')"]))
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
        ])
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
    agent = build_quill(model=fake_model([script]))
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
    import quill.run as run_mod

    monkeypatch.setattr(
        run_mod, "build_quill",
        lambda: build_quill(model=fake_model([_load(CSV) + "\nfinal_answer('Team grew fastest')"])),
    )
    code = run_main([CSV, "Which category grew fastest?"])
    assert code == 0
    out = capsys.readouterr().out
    assert "[Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct" in out
    assert "[Quill] Run cost — input tokens:" in out
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
    agent = build_quill(model=fake_model([script]))
    out = agent.run(build_task(CSV, "Which category has the highest revenue?"))
    assert out == "Team (highest net revenue)"


def test_pandas_actually_imports_in_the_sandbox(fake_model):
    script = _load(CSV) + "\nfinal_answer(int(len(df)))"
    agent = build_quill(model=fake_model([script]))
    out = agent.run(build_task(CSV, "How many rows?"))
    assert out == 108


def test_return_full_result_gives_a_runresult(fake_model):
    script = _load(CSV) + "\nfinal_answer('Team')"
    agent = build_quill(model=fake_model([script]))
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
    agent = build_quill(model=fake_model([bad, good]))
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
    agent = build_quill(model=fake_model([_load(CSV) + "\nfinal_answer('ok')"]))
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
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    names = set(agent.tools)
    assert {"load_dataset", "profile_dataframe", "save_chart"} <= names
    assert "web_search" in names
    assert "visit_webpage" in names
    assert "final_answer" in names
    assert "python_interpreter" not in names


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
    agent = build_quill(model=fake_model([script]))
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
    full toolbox run — see the next test for the real Approach-1 caveat about Quill's tools."""
    monkeypatch.setenv("QUILL_EXECUTOR", "docker")
    with build_quill(model=fake_model(["final_answer('ok')"])) as agent:
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
    the bare-executor docker run above proves the sandbox boundary itself works.)"""
    monkeypatch.setenv("QUILL_EXECUTOR", "docker")
    script = "import numpy as np\nfinal_answer(int(np.array([1, 2, 3, 4]).sum()))"
    with build_quill(model=fake_model([script])) as agent:
        assert agent.executor_type == "docker"
        with pytest.raises(ValueError, match="SimpleTool validation failed"):
            agent.run("Sum 1..4.")


# ======================================================================================
# Live (skipped by default; QUILL_LIVE_TESTS=1 + HF_TOKEN). Budget: ~4 real LLM runs
# (multi-turn = 2 runs on one agent; the M7 live bench = 2 runs, baseline + improved).
# ======================================================================================

@pytest.mark.live
def test_quill_live_planning_inserts_a_real_planning_step():
    """M7 with a REAL model: planning_interval=3 makes Quill emit a real PlanningStep on a
    multi-step question. Budget: ~1 LLM run (a couple of action steps + at least one plan)."""
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    with build_quill(planning_interval=DEFAULT_PLANNING_INTERVAL) as agent:
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

    with build_quill(instructions="") as baseline_agent:
        baseline = bench.run_and_count(baseline_agent, task)
    with build_quill(planning_interval=DEFAULT_PLANNING_INTERVAL) as improved_agent:
        improved = bench.run_and_count(improved_agent, task)

    assert baseline.action_steps >= 1
    assert improved.action_steps >= 1
    assert improved.planning_steps >= 1, "the improved config plans at least once"
    report = bench.format_report(baseline, improved)
    assert "Baseline" in report and "Improved" in report


@pytest.mark.live
def test_quill_live_run_reports_a_real_token_cost():
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    with build_quill() as agent:  # make_model() default: InferenceClientModel(Qwen coder)
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
    with build_quill() as agent:
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
