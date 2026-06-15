"""Module 4 smoke tests — Quill gains make_model() and prints its run cost.

Live call budget: ~1 real LLM run, well under 10 LLM calls (only when QUILL_LIVE_TESTS=1 and
HF_TOKEN is set). Everything else runs OFFLINE: model FACTORY tests construct the model
objects WITHOUT any network (we only assert the class and model_id — InferenceClientModel and
LiteLLMModel build lazily, the first real HTTP call happens on .generate(), which we never do
offline), and agent-loop tests use the shared FakeModel (see repo-root conftest.py).

This file carries the Module 3 toolbox tests forward (the tools + the agent loop still work)
and adds the Module 4 model-layer tests:
- make_model() returns the right class per QUILL_MODEL_BACKEND (hf/litellm/local), default hf;
- DEFAULT_MODEL_ID is the explicit coder pin (never the library default);
- the swap flows through build_quill() with NO agent-code edit;
- the Monitor / TokenUsage cost accessor exists and aggregates a run.

Run from the repo root: ``uv run pytest module-04/tests/``
"""
from __future__ import annotations

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
    RunResult,
    TokenUsage,
    Tool,
)

# Make THIS module's `quill` package importable when running from the repo root, even in the
# cumulative suite where earlier modules also ship a `quill` package. Every module-NN/ is a
# self-contained snapshot, so several dirs define a top-level `quill`; whichever is imported
# first would otherwise win in sys.modules. We prepend this module's dir and drop any cached
# `quill*` so this file always binds to module-04/quill.
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
from quill.config import (  # noqa: E402
    DEFAULT_LOCAL_MODEL_ID,
    OLLAMA_NUM_CTX,
    Settings,
    make_model,
)
from quill.run import _format_cost  # noqa: E402
from quill.run import main as run_main  # noqa: E402
from quill.tools import load_dataset, profile_dataframe, save_chart  # noqa: E402

CSV = str(MODULE_DIR / "data" / "sales.csv")
OUTPUTS = MODULE_DIR / "outputs"


def _load(path: str) -> str:
    return f'import pandas as pd\ndf = pd.read_csv({path!r})'


@pytest.fixture(autouse=True)
def _clean_model_env(monkeypatch):
    """Every test starts from a clean model-selection environment so order never matters."""
    monkeypatch.delenv("QUILL_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("QUILL_MODEL_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)


# --------------------------------------------------------------------------------------
# Module 4 (new): the model factory — the FROZEN model contract (06 §2)
# --------------------------------------------------------------------------------------

def test_default_model_id_is_the_explicit_coder_pin():
    """Frozen contract: the default model_id is an EXPLICIT coder model, never the library
    default (Qwen/Qwen3-Next-80B-A3B-Thinking, 'subject to change' as of 1.26.0)."""
    assert DEFAULT_MODEL_ID == "Qwen/Qwen2.5-Coder-32B-Instruct"
    assert Settings.DEFAULT_MODEL_ID == DEFAULT_MODEL_ID


def test_make_model_signature_is_frozen():
    """make_model(role='analyst', **overrides) -> Model. The role param exists and defaults
    to 'analyst' (the frozen signature later modules rely on)."""
    import inspect

    sig = inspect.signature(make_model)
    params = sig.parameters
    assert "role" in params
    assert params["role"].default == "analyst"
    # **overrides is accepted (so completion kwargs / provider / requests_per_minute pass through)
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def test_make_model_default_is_inferenceclientmodel():
    """No QUILL_MODEL_BACKEND set -> hf backend -> InferenceClientModel with the coder pin.
    Constructing it does NOT hit the network (no .generate call); we only assert class+id."""
    model = make_model("analyst")
    assert isinstance(model, InferenceClientModel)
    assert isinstance(model, Model)
    assert model.model_id == DEFAULT_MODEL_ID


def test_make_model_default_backend_is_hf():
    assert Settings.MODEL_BACKEND == "hf"
    assert Settings.DEFAULT_BACKEND == "hf"


def test_make_model_swaps_to_litellm(monkeypatch):
    """QUILL_MODEL_BACKEND=litellm + QUILL_MODEL_ID=gpt-4o -> LiteLLMModel(model_id='gpt-4o').
    No agent-code edit, no network."""
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "litellm")
    monkeypatch.setenv("QUILL_MODEL_ID", "gpt-4o")
    model = make_model("analyst")
    assert isinstance(model, LiteLLMModel)
    assert not isinstance(model, InferenceClientModel)
    assert model.model_id == "gpt-4o"


def test_make_model_local_backend_uses_ollama_via_litellm(monkeypatch):
    """QUILL_MODEL_BACKEND=local -> Ollama via LiteLLMModel, with a sensible default model id
    and num_ctx raised above Ollama's 2048 default (which 'fails horribly' for an agent)."""
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "local")
    model = make_model("analyst")
    assert isinstance(model, LiteLLMModel)
    assert model.model_id == DEFAULT_LOCAL_MODEL_ID
    assert model.model_id.startswith("ollama_chat/")
    assert OLLAMA_NUM_CTX >= 8192  # the guard-rail value


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
    """We fail loud on a typo'd backend instead of silently falling back to the wrong one."""
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "bogus")
    with pytest.raises(ValueError, match="Unknown QUILL_MODEL_BACKEND"):
        make_model("analyst")


def test_make_model_forwards_overrides(monkeypatch):
    """Completion kwargs / provider go to the model at init (the uniform smolagents pattern).
    We pass a provider override and confirm the call accepts it without error."""
    # provider is a real InferenceClientModel kwarg; passing it must not raise at construction.
    model = make_model("analyst", provider="together")
    assert isinstance(model, InferenceClientModel)
    assert model.model_id == DEFAULT_MODEL_ID


def test_settings_rereads_env_at_access_time(monkeypatch):
    """Settings.MODEL_BACKEND/MODEL_ID re-read the environment on every access, so the swap
    works even though quill.config was imported before the env var was set."""
    assert Settings.MODEL_BACKEND == "hf"
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "litellm")
    monkeypatch.setenv("QUILL_MODEL_ID", "anthropic/claude-3-5-sonnet-latest")
    assert Settings.MODEL_BACKEND == "litellm"
    assert Settings.MODEL_ID == "anthropic/claude-3-5-sonnet-latest"


# --------------------------------------------------------------------------------------
# Module 4 (new): build_quill uses make_model(); the swap flows through with no agent edit
# --------------------------------------------------------------------------------------

def test_build_quill_uses_make_model_when_no_model_passed():
    """With no model injected, build_quill() builds the InferenceClientModel default via
    make_model() — proving agent.py no longer instantiates a model class itself."""
    agent = build_quill()
    assert isinstance(agent.model, InferenceClientModel)
    assert agent.model.model_id == DEFAULT_MODEL_ID


def test_build_quill_swap_flows_through_env(monkeypatch):
    """The whole point of M4: change QUILL_MODEL_BACKEND and build_quill() gets the new
    backend with ZERO changes to agent code."""
    monkeypatch.setenv("QUILL_MODEL_BACKEND", "litellm")
    monkeypatch.setenv("QUILL_MODEL_ID", "gpt-4o")
    agent = build_quill()
    assert isinstance(agent.model, LiteLLMModel)
    assert agent.model.model_id == "gpt-4o"


def test_build_quill_still_accepts_an_injected_model(fake_model):
    """The injected-model path (used by every offline loop test) is untouched."""
    agent = build_quill(model=fake_model(["final_answer('ok')"]))
    assert isinstance(agent, CodeAgent)
    assert agent.model.model_id == "fake/deterministic"


# --------------------------------------------------------------------------------------
# Module 4 (new): token cost in plain sight via Monitor / TokenUsage
# --------------------------------------------------------------------------------------

def test_agent_has_a_monitor_with_total_token_counts(fake_model):
    """The cost accessor exists: agent.monitor is a Monitor, and get_total_token_counts()
    returns a TokenUsage (NOT agent.logs / legacy token attrs, removed in 1.21.0)."""
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert isinstance(agent.monitor, Monitor)
    usage = agent.monitor.get_total_token_counts()
    assert isinstance(usage, TokenUsage)
    assert hasattr(usage, "input_tokens")
    assert hasattr(usage, "output_tokens")
    assert hasattr(usage, "total_tokens")
    assert not hasattr(agent, "logs")  # removed in 1.21.0


def test_token_usage_total_is_input_plus_output():
    """TokenUsage.total_tokens is computed (input + output) in __post_init__."""
    usage = TokenUsage(input_tokens=8142, output_tokens=1203)
    assert usage.total_tokens == 9345


def test_run_cost_accessor_after_a_run(fake_model):
    """After a run, get_total_token_counts() returns a TokenUsage. The offline FakeModel
    reports no usage, so totals are >= 0; the live test asserts > 0 against a real model."""
    agent = build_quill(model=fake_model([_load(CSV) + "\nfinal_answer('Team')"]))
    agent.run(build_task(CSV, "Top category?"))
    usage = agent.monitor.get_total_token_counts()
    assert isinstance(usage, TokenUsage)
    assert usage.total_tokens >= 0


def test_format_cost_line_shape():
    """run.py's cost line is human-readable with thousands separators."""
    line = _format_cost(TokenUsage(input_tokens=8142, output_tokens=1203))
    assert "input tokens: 8,142" in line
    assert "output tokens: 1,203" in line
    assert "total: 9,345" in line


def test_run_main_prints_backend_model_and_cost(fake_model, monkeypatch, capsys):
    """The CLI prints 'Backend: ... | Model: ...' and a cost line. We inject a FakeModel via
    make_model so no network is touched, then drive run.main on the canonical CSV."""
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


# --------------------------------------------------------------------------------------
# Module 2/3 (carried forward): the agent loop + the toolbox still work, contracts unchanged
# --------------------------------------------------------------------------------------

def test_build_quill_is_a_codeagent(fake_model):
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert isinstance(agent, CodeAgent)


def test_quill_imports_locked_no_wildcard(fake_model):
    # Frozen contract (06 §2): additional_authorized_imports is a minimal explicit list and
    # is NEVER the "*" wildcard. M3 added "matplotlib.*" (a clean superset toward M5's list).
    assert QUILL_IMPORTS == ["pandas", "numpy", "matplotlib.*"]
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert "*" not in agent.additional_authorized_imports
    for imp in ("pandas", "numpy", "matplotlib.*"):
        assert imp in agent.additional_authorized_imports


def test_max_steps_is_capped_low(fake_model):
    agent = build_quill(model=fake_model(["final_answer('done')"]))
    assert agent.max_steps == 8


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
    """Step 1 hits a KeyError (wrong column), it is CAPTURED in ActionStep.error (not a
    crash), fed back, and step 2 fixes it and answers."""
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
    """We read the trajectory via agent.memory.steps / replay() — never agent.logs."""
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
    """Frozen contract (06 §2): canonical names and the @tool-derived schema."""
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
    """Scripted run: Quill calls profile_dataframe, draws + saves a chart via save_chart, and
    returns a final answer that includes the saved chart path. No network, no token."""
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


# --------------------------------------------------------------------------------------
# Live (skipped by default; QUILL_LIVE_TESTS=1 + HF_TOKEN). Budget: ~1 real LLM run.
# --------------------------------------------------------------------------------------

@pytest.mark.live
def test_quill_live_run_reports_a_real_token_cost():
    """One real run against the default hosted model via make_model(). Asserts the run
    succeeds AND the Monitor reports a non-zero token cost — the M4 'cost in plain sight'
    contract end to end (a real model populates chat_message.token_usage)."""
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    agent = build_quill()  # make_model() default: InferenceClientModel(Qwen coder)
    task = build_task(
        CSV,
        "Which product category grew fastest from Q1 to Q4 2025? Back it with a saved chart.",
    )
    result = agent.run(task, return_full_result=True)
    assert result.state == "success"
    assert str(result.output).strip()
    usage = agent.monitor.get_total_token_counts()
    assert usage.total_tokens > 0, "expected a real run to report token usage > 0"


@pytest.mark.live
def test_quill_live_swap_to_litellm():
    """Optional second live run: swap to a LiteLLM provider with NO agent-code change.
    Skips cleanly unless a provider key is present. Budget: ~1 extra LLM run."""
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("needs OPENAI_API_KEY to run the litellm/gpt-4o swap")
    os.environ["QUILL_MODEL_BACKEND"] = "litellm"
    os.environ["QUILL_MODEL_ID"] = "gpt-4o"
    try:
        agent = build_quill()
        assert isinstance(agent.model, LiteLLMModel)
        result = agent.run(build_task(CSV, "How many rows are in the dataset?"),
                           return_full_result=True)
        assert result.state == "success"
        assert agent.monitor.get_total_token_counts().total_tokens > 0
    finally:
        os.environ.pop("QUILL_MODEL_BACKEND", None)
        os.environ.pop("QUILL_MODEL_ID", None)
