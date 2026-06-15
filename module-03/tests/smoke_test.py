"""Module 3 smoke tests — Quill gains a toolbox.

Live call budget: ~1 real LLM run, well under 10 LLM calls (only when QUILL_LIVE_TESTS=1
and HF_TOKEN is set). Everything else runs OFFLINE via the shared FakeModel (see repo-root
conftest.py): the agent loop and the new tools are functionally verified — tool
instantiation/validation, profile_dataframe over the CSV, save_chart writing a real PNG,
and Quill calling its tools end to end — with no network or token.

Run from the repo root: ``uv run pytest module-03/tests/``
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
from smolagents import ActionStep, CodeAgent, RunResult, Tool

# Make THIS module's `quill` package importable when running from the repo root, even in
# the cumulative suite where module-02 also ships a `quill` package. Every module-NN/ is a
# self-contained snapshot, so several dirs define a top-level `quill`; whichever is imported
# first would otherwise win in sys.modules. We prepend this module's dir and drop any cached
# `quill*` so this file always binds to module-03/quill.
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
from quill.tools import load_dataset, profile_dataframe, save_chart  # noqa: E402

CSV = str(MODULE_DIR / "data" / "sales.csv")
OUTPUTS = MODULE_DIR / "outputs"


def _load(path: str) -> str:
    return f'import pandas as pd\ndf = pd.read_csv({path!r})'


# --------------------------------------------------------------------------------------
# Module 2 (carried forward): the agent loop still works, contracts unchanged
# --------------------------------------------------------------------------------------

def test_build_quill_is_a_codeagent(fake_model):
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert isinstance(agent, CodeAgent)


def test_quill_imports_locked_no_wildcard(fake_model):
    # Frozen contract (06 §2): additional_authorized_imports is a minimal explicit list and
    # is NEVER the "*" wildcard. M3 adds "matplotlib.*" (a clean superset toward M5's list).
    assert QUILL_IMPORTS == ["pandas", "numpy", "matplotlib.*"]
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert "*" not in agent.additional_authorized_imports
    for imp in ("pandas", "numpy", "matplotlib.*"):
        assert imp in agent.additional_authorized_imports


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
    assert action_steps[0].error is not None
    assert action_steps[-1].error is None
    assert any(s.is_final_answer for s in action_steps)


def test_max_steps_error_when_final_answer_never_called(fake_model):
    """If the model never calls final_answer, the agent loops to the ceiling and ends in
    state='max_steps_error' — it does not 'decide' it is done on its own."""
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


# --------------------------------------------------------------------------------------
# Module 3 (new): the tools themselves — instantiate, validate, and run offline
# --------------------------------------------------------------------------------------

def test_data_tools_are_validated_tool_instances():
    """All three data tools instantiate without raising — i.e. validate_arguments() (run
    at construction via Tool.__init_subclass__) passed: names are valid Python identifiers,
    inputs/output_type are well-formed, and forward() matches the declared inputs."""
    for t in (load_dataset, profile_dataframe, save_chart()):
        assert isinstance(t, Tool)
        t.validate_arguments()  # explicit re-check: must not raise


def test_frozen_tool_signatures_and_schema():
    """Frozen contract (06 §2): canonical names and the @tool-derived schema.
    load_dataset/profile_dataframe take a single string `path` and return a string;
    save_chart's canonical name is exactly 'save_chart' and returns a string."""
    assert load_dataset.name == "load_dataset"
    assert load_dataset.output_type == "string"
    assert set(load_dataset.inputs) == {"path"}
    assert load_dataset.inputs["path"]["type"] == "string"

    assert profile_dataframe.name == "profile_dataframe"
    assert profile_dataframe.output_type == "string"
    assert set(profile_dataframe.inputs) == {"path"}
    assert profile_dataframe.inputs["path"]["type"] == "string"

    sc = save_chart()
    assert sc.name == "save_chart"  # exact canonical name
    assert sc.output_type == "string"
    # Optional filename input, declared nullable.
    assert sc.inputs["filename"]["nullable"] is True


def test_tools_have_descriptions_injected_into_the_prompt():
    """A tool's description IS the interface the model reads — every Quill tool has one,
    and the @tool decorator mapped the first docstring paragraph into it."""
    assert load_dataset.description.strip()
    assert "summary" in load_dataset.description.lower()
    assert profile_dataframe.description.strip()
    assert save_chart().description.strip()


def test_save_chart_lazy_setup_runs_once():
    """save_chart is NOT initialized at construction; setup() runs lazily on the first
    call only (smolagents calls it when not self.is_initialized)."""
    sc = save_chart()
    assert sc.is_initialized is False  # no matplotlib touched at construction time
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot([1, 2, 3], [1, 4, 9])
    sc("setup_probe")
    assert sc.is_initialized is True  # super().setup() flipped the flag


def test_save_chart_writes_a_png_and_returns_the_path():
    """Call save_chart on a real matplotlib figure: it returns a PNG path under outputs/
    and the file exists on disk (Agg backend, no display)."""
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
    os.remove(out)  # keep the workspace clean


def test_save_chart_errors_cleanly_with_no_figure():
    """A good tool fails with an informative ValueError, not a silent crash: calling
    save_chart with nothing drawn tells the agent to draw first."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.close("all")  # ensure there is no current figure with axes
    sc = save_chart()
    with pytest.raises(ValueError, match="draw a chart"):
        sc("nothing_drawn")


def test_profile_dataframe_returns_a_nonempty_summary():
    """Call the tool directly on the canonical CSV — it returns a non-empty profile that
    mentions the schema and the known columns."""
    out = profile_dataframe(CSV)
    assert isinstance(out, str)
    assert out.strip()
    assert "category" in out
    assert "net_rev" in out
    assert "108 rows" in out  # 108 data rows in data/sales.csv


def test_load_dataset_returns_a_summary_string():
    out = load_dataset(CSV)
    assert isinstance(out, str)
    assert "108 rows" in out
    assert "category" in out


def test_data_tools_raise_valueerror_on_bad_path():
    """Informative ValueError (not a bare FileNotFoundError) so the agent can self-correct."""
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
    """build_quill wires the 3 data tools + web tools; agent.tools is a dict keyed by name.
    FinalAnswerTool is always present; for a CodeAgent there is NO python_interpreter tool."""
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    names = set(agent.tools)
    assert {"load_dataset", "profile_dataframe", "save_chart"} <= names
    assert "web_search" in names  # WebSearchTool's shared name
    assert "visit_webpage" in names
    assert "final_answer" in names  # always added, regardless of add_base_tools
    assert "python_interpreter" not in names  # excluded for a CodeAgent


def test_toolbox_is_runtime_mutable_a_dict_keyed_by_name(fake_model):
    """T3.11: agent.tools is a plain dict keyed by `name`; you can add a tool at runtime."""
    from smolagents import VisitWebpageTool

    agent = build_quill(model=fake_model(["final_answer('done')"]))
    extra = VisitWebpageTool()
    agent.tools[extra.name] = extra  # add/replace by name
    assert agent.tools[extra.name] is extra


def test_quill_uses_its_tools_end_to_end_offline(fake_model):
    """Scripted run: Quill calls profile_dataframe, draws + saves a chart via save_chart,
    and returns a final answer that includes the saved chart path. No network, no token."""
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
    # Extract the saved path from the answer and confirm the PNG really exists.
    saved = str(out).split("Chart saved at ")[-1].strip()
    assert os.path.exists(saved), f"expected the saved chart at {saved}"
    os.remove(saved)


# --------------------------------------------------------------------------------------
# Live (skipped by default; QUILL_LIVE_TESTS=1 + HF_TOKEN)
# --------------------------------------------------------------------------------------

@pytest.mark.live
def test_quill_live_run_saves_a_chart():
    """One real run against the hosted model. The model should call its tools and produce a
    saved PNG. Asserts the harness + that a chart file was created, not output quality."""
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    before = set(OUTPUTS.glob("*.png")) if OUTPUTS.exists() else set()
    agent = build_quill()
    task = build_task(
        CSV,
        "Which product category grew fastest from Q1 to Q4 2025? Back it with a saved chart.",
    )
    result = agent.run(task, return_full_result=True)
    assert result.state == "success"
    assert str(result.output).strip()
    after = set(OUTPUTS.glob("*.png")) if OUTPUTS.exists() else set()
    assert after - before, "expected the live run to save at least one new chart PNG"
