"""Module 2 smoke tests — Quill v0.

Live call budget: ~1 real LLM run, well under 10 LLM calls (only when QUILL_LIVE_TESTS=1
and HF_TOKEN is set). Everything else runs OFFLINE via the shared FakeModel (see repo-root
conftest.py): the agent loop is functionally verified — pandas over the CSV, final_answer,
self-correction, RunResult, replay()/memory.steps — with no network or token.

Run from the repo root: ``uv run pytest module-02/tests/``
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
from smolagents import ActionStep, CodeAgent, RunResult

# Make THIS module's `quill` package importable when running from the repo root, even in
# the cumulative suite where module-03+ also ship a `quill` package. Every module-NN/ is a
# self-contained snapshot, so several dirs define a top-level `quill`; whichever is imported
# first would otherwise win in sys.modules. We prepend this module's dir and drop any cached
# `quill*` so this file always binds to module-02/quill.
MODULE_DIR = pathlib.Path(__file__).resolve().parents[1]
for _name in [n for n in list(sys.modules) if n == "quill" or n.startswith("quill.")]:
    del sys.modules[_name]
sys.path.insert(0, str(MODULE_DIR))
from quill.agent import (  # noqa: E402
    DEFAULT_MODEL_ID,
    QUILL_IMPORTS,
    build_quill,
    build_task,
)

CSV = str(MODULE_DIR / "data" / "sales.csv")


def _load(path: str) -> str:
    return f'import pandas as pd\ndf = pd.read_csv({path!r})'


def test_build_quill_is_a_codeagent(fake_model):
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert isinstance(agent, CodeAgent)


def test_quill_imports_locked_to_pandas_numpy(fake_model):
    # Frozen contract (06 §2): additional_authorized_imports is a SUBSET of the minimal
    # Quill list and is NEVER the "*" wildcard.
    assert QUILL_IMPORTS == ["pandas", "numpy"]
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert "*" not in agent.additional_authorized_imports
    assert "pandas" in agent.additional_authorized_imports
    assert "numpy" in agent.additional_authorized_imports


def test_max_steps_is_capped_low(fake_model):
    # We cap below the library default (20): a single-CSV analysis never needs 20 loops.
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert agent.max_steps == 8


def test_default_model_id_is_pinned():
    # Explicit coder model_id; never silently rely on the library default.
    assert DEFAULT_MODEL_ID == "Qwen/Qwen2.5-Coder-32B-Instruct"


def test_quill_answers_a_csv_question_offline(fake_model):
    """The agent writes pandas over data/sales.csv and returns the scripted answer."""
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
    """Proves pandas is genuinely authorized: the code reads the CSV and the row count
    (108 data rows) flows through to the final answer."""
    script = (
        _load(CSV)
        + "\nfinal_answer(int(len(df)))"
    )
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
    # RunResult.steps is a list[dict] (serialized), NOT a list of step objects.
    assert isinstance(result.steps, list)
    assert all(isinstance(s, dict) for s in result.steps)
    # token_usage exists on the dataclass; offline (FakeModel) it is None, since usage is
    # only meaningful against a real model — the live test is what exercises real cost.
    assert hasattr(result, "token_usage")
    assert hasattr(result, "timing")


def test_agent_self_corrects_after_an_error(fake_model):
    """The click moment: step 1 hits a KeyError (wrong column), it is CAPTURED in
    ActionStep.error (not a crash), fed back, and step 2 fixes it and answers."""
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
    # First action step recorded an error; the run still succeeded.
    assert action_steps[0].error is not None
    # A later step has no error and produced the final answer.
    assert action_steps[-1].error is None
    assert any(s.is_final_answer for s in action_steps)


def test_max_steps_error_when_final_answer_never_called(fake_model):
    """If the model never calls final_answer, the agent loops to the ceiling and ends in
    state='max_steps_error' — it does not 'decide' it is done on its own."""
    # A harmless step that never terminates; cap the run at 2 steps via the override.
    looping = _load(CSV) + "\nprint('still thinking...')"
    agent = build_quill(model=fake_model([looping]))
    result = agent.run(build_task(CSV, "Loop forever"), max_steps=2, return_full_result=True)
    assert result.state == "max_steps_error"


def test_trajectory_is_readable_via_memory_steps(fake_model):
    """We read the trajectory via agent.memory.steps / replay() — never agent.logs."""
    agent = build_quill(model=fake_model([_load(CSV) + "\nfinal_answer('ok')"]))
    agent.run(build_task(CSV, "anything"))
    assert hasattr(agent, "memory")
    assert not hasattr(agent, "logs")  # removed in 1.21.0
    assert any(isinstance(s, ActionStep) for s in agent.memory.steps)
    agent.replay()  # must not raise


@pytest.mark.live
def test_quill_live_run():
    """One real run against the hosted model. Asserts the harness, not output quality."""
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    agent = build_quill()
    task = build_task(CSV, "Which category has the highest total net revenue across 2025?")
    result = agent.run(task, return_full_result=True)
    assert result.state == "success"
    assert str(result.output).strip()
    assert len(agent.memory.steps) >= 1
    assert any(isinstance(s, ActionStep) for s in agent.memory.steps)
