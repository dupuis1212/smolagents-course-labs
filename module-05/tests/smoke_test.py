"""Module 5 smoke tests — Quill gains a frozen sandbox policy (quill/sandbox.py).

Live call budget: ~1 real LLM run, well under 10 LLM calls (only when QUILL_LIVE_TESTS=1 and
HF_TOKEN is set). Sandbox budget: 1-2 Docker runs (marked `sandbox`, skipped if Docker is
absent; allow ~180s for the first image build). Everything else runs OFFLINE with no network
and no token: the agent-loop tests use the shared FakeModel (see repo-root conftest.py), and
the sandbox-policy tests poke `resolve_executor()` and the LOCAL executor directly.

This file carries the Module 2/3/4 tests forward (the toolbox, the agent loop, make_model(),
and the Monitor cost accessor still work) and adds the Module 5 sandbox tests:
- resolve_executor() reads QUILL_EXECUTOR in {local, docker, e2b}, defaults to local, and
  raises ValueError on anything else (including a stale "wasm");
- the returned import list is the FROZEN least-privilege list and is NEVER "*";
- build_quill() now wires executor_type + additional_authorized_imports from resolve_executor;
- a dangerous import (`import os`) on a LOCAL-executor agent is BLOCKED (InterpreterError /
  the os result never reaches final_answer), and a runaway loop is cut off by the cap;
- a `sandbox`-marked test runs a trivial calc inside a real Docker container and cleans up.

Run from the repo root: ``uv run pytest module-05/tests/``
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
from smolagents.local_python_executor import InterpreterError

# Make THIS module's `quill` package importable when running from the repo root, even in the
# cumulative suite where earlier modules also ship a `quill` package. Every module-NN/ is a
# self-contained snapshot, so several dirs define a top-level `quill`; whichever is imported
# first would otherwise win in sys.modules. We prepend this module's dir and drop any cached
# `quill*` so this file always binds to module-05/quill.
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
from quill.sandbox import (  # noqa: E402
    DEFAULT_EXECUTOR,
    QUILL_AUTHORIZED_IMPORTS,
    SUPPORTED_EXECUTORS,
    resolve_executor,
)
from quill.tools import load_dataset, profile_dataframe, save_chart  # noqa: E402

CSV = str(MODULE_DIR / "data" / "sales.csv")
OUTPUTS = MODULE_DIR / "outputs"


def _load(path: str) -> str:
    return f'import pandas as pd\ndf = pd.read_csv({path!r})'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Every test starts from a clean model/executor environment so order never matters."""
    monkeypatch.delenv("QUILL_MODEL_BACKEND", raising=False)
    monkeypatch.delenv("QUILL_MODEL_ID", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("QUILL_EXECUTOR", raising=False)


# ======================================================================================
# Module 5 (NEW): the sandbox policy — quill/sandbox.py, the FROZEN contract (06 §2)
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
# Live (skipped by default; QUILL_LIVE_TESTS=1 + HF_TOKEN). Budget: ~1 real LLM run.
# ======================================================================================

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
