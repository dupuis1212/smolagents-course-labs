"""Shared test support for the smolagents-course-labs repo (Quill).

Provides:
- ``FakeModel``: a deterministic, offline ``smolagents.Model`` that returns scripted
  assistant messages (code actions for ``CodeAgent``). It lets the whole agent loop run
  with NO network and NO HF token, so we can functionally verify the *harness*
  (parsing, tool calls, the local sandbox, memory, ``final_answer``, callbacks) without
  paying for or depending on an LLM. It does NOT verify model output *quality* — that is
  what the ``live`` tests are for.
- ``code_step``: wraps raw Python into the ``Thought:`` + ``<code>...</code>`` shape a
  CodeAgent expects (default ``code_block_tags`` are ``<code>``/``</code>`` in smolagents 1.26.0).
- a ``fake_model`` fixture (a factory) used by every module's tests.
- the ``live`` / ``sandbox`` marker policy (06-FIL-ROUGE-SPEC §2): ``live`` tests are
  skipped unless ``QUILL_LIVE_TESTS=1``; ``sandbox`` tests are skipped if Docker is absent.

Run tests from the repo root, e.g. ``uv run pytest module-05/tests/``.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import sys

import pytest
from smolagents import ChatMessage, MessageRole, Model


def code_step(code: str, thought: str = "Let's compute the answer in Python.") -> str:
    """Wrap raw Python into the Thought + <code> action a CodeAgent parses."""
    return f"Thought: {thought}\n<code>\n{code}\n</code>"


class FakeModel(Model):
    """Deterministic offline model. Feed it a list of scripted assistant messages
    (already-formed strings, or raw code that ``code_step`` will wrap). Each call returns
    the next script entry; the last entry repeats if the agent asks for more steps."""

    def __init__(self, scripted: list[str], model_id: str = "fake/deterministic"):
        super().__init__(model_id=model_id)
        self.scripted = list(scripted)
        self.calls = 0

    def generate(self, messages, stop_sequences=None, response_format=None,
                 tools_to_call_from=None, **kwargs) -> ChatMessage:
        text = self.scripted[min(self.calls, len(self.scripted) - 1)]
        self.calls += 1
        return ChatMessage(role=MessageRole.ASSISTANT, content=text)


# Per-module snapshot of the `quill*` module objects each smoke_test.py loads, keyed by the
# absolute module-NN/ path. Filled at COLLECTION time (right after each test module imports),
# so each entry holds exactly that module's own package objects.
_QUILL_MODULE_SNAPSHOTS: dict[str, dict[str, object]] = {}


def _module_dir_for(path) -> str | None:
    """Return the absolute module-NN/ dir for a tests/smoke_test.py path, if it has a quill/."""
    module_dir = pathlib.Path(str(path)).resolve().parent.parent  # tests/ -> module-NN/
    return str(module_dir) if (module_dir / "quill").is_dir() else None


def _snapshot_quill(module_dir: str) -> None:
    """Record the currently-loaded `quill*` modules that live under ``module_dir``."""
    snapshot = {}
    for name, mod in list(sys.modules.items()):
        if name != "quill" and not name.startswith("quill."):
            continue
        mod_file = getattr(mod, "__file__", "") or ""
        if mod_file.startswith(module_dir + os.sep):
            snapshot[name] = mod
    if snapshot:
        _QUILL_MODULE_SNAPSHOTS[module_dir] = snapshot


@pytest.hookimpl(hookwrapper=True)
def pytest_pycollect_makemodule(module_path, parent):
    """After each smoke_test.py is imported for collection, snapshot ITS own `quill*` modules.

    Every module-NN/ ships its own self-contained top-level `quill` package, all sharing that
    name. The cumulative suite imports every smoke_test.py, so once collection finishes
    sys.modules['quill'] points at whichever module was imported LAST. A test that does a
    RUNTIME ``import quill.run`` (the CLI test) would then bind to the WRONG module's package
    and its monkeypatch would miss the ``build_quill`` the collection-time ``run.main`` calls.

    The Module object returned here imports the test file lazily via ``.obj``; touching it now
    forces the import, after which sys.modules holds this module's `quill*` objects. We snapshot
    them so the autouse fixture below can restore the correct per-module package before each
    test, so a runtime import resolves to the same objects the test bound.
    """
    outcome = yield
    module_dir = _module_dir_for(module_path)
    if module_dir is None:
        return
    mod = outcome.get_result()
    if mod is not None:
        try:
            mod.obj  # force the import of the test file (and its module-NN/quill imports)
        except Exception:
            return
    _snapshot_quill(module_dir)


@pytest.fixture(autouse=True)
def _bind_quill_to_this_modules_package(request):
    """Restore this test's own module-NN/ `quill*` objects into sys.modules before it runs.

    Pairs with the ``pytest_pycollect_makemodule`` snapshot above: a runtime ``import
    quill.run`` then resolves to the identical module the test imported at collection time, so
    monkeypatches land on the object actually used. No re-import; the test's collection-time
    bindings stay valid.
    """
    module_dir = _module_dir_for(request.fspath)
    if module_dir is not None and module_dir in _QUILL_MODULE_SNAPSHOTS:
        for name in [n for n in list(sys.modules) if n == "quill" or n.startswith("quill.")]:
            del sys.modules[name]
        sys.modules.update(_QUILL_MODULE_SNAPSHOTS[module_dir])
        if sys.path[:1] != [module_dir]:
            sys.path.insert(0, module_dir)
    yield


@pytest.fixture
def fake_model():
    """Factory: ``fake_model([raw_code_or_full_message, ...]) -> FakeModel``.
    Entries that already start with 'Thought:' are used verbatim; otherwise they are
    treated as raw Python and wrapped with ``code_step``."""
    def _make(scripted: list[str]) -> FakeModel:
        prepared = [s if s.lstrip().startswith("Thought:") else code_step(s) for s in scripted]
        return FakeModel(prepared)
    return _make


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: hits a real LLM/network (needs HF_TOKEN; skipped unless QUILL_LIVE_TESTS=1)",
    )
    config.addinivalue_line(
        "markers", "sandbox: needs Docker/E2B (skipped if unavailable)"
    )


# smolagents' DockerExecutor (the Jupyter Kernel Gateway) ALWAYS binds this fixed host port for
# its kernel websocket (remote_executors.py — "Starting container on 127.0.0.1:8888..."). It is
# part of "is Docker usable right now?": if the port is occupied (a leaked kernel container from
# an earlier run, another Jupyter), every sandbox test would 500 with "port is already allocated"
# or hang on the kernel — i.e. Docker is present but NOT serviceable.
_DOCKER_KERNEL_PORT = 8888


def _sandbox_available() -> bool:
    """Is Docker GENUINELY able to serve a container right now? (the `sandbox`-marker predicate)

    06-FIL-ROUGE-SPEC §2 requires `sandbox` tests to "skip propre si indisponible". A `docker`
    binary on PATH is NOT enough: the daemon may be down, or the fixed kernel port 8888 may be held
    by a leaked container — in either case the DockerExecutor fails (500) or hangs. We treat all of
    those as "unavailable" and skip, instead of failing/hanging the offline suite. Set
    `QUILL_SANDBOX_TESTS=1` to FORCE these tests to run (a healthy CI box with a free port).

    Checks, cheaply and with timeouts (never blocks the suite):
      1. the `docker` binary exists,
      2. the daemon answers `docker info`,
      3. the kernel port 8888 is free (so DockerExecutor can bind it).
    """
    if shutil.which("docker") is None:
        return False
    try:
        import subprocess

        # Daemon reachable? `docker info` returns non-zero (and fast) when the daemon is down.
        if subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
        ).returncode != 0:
            return False
    except Exception:
        return False

    # Is the fixed Jupyter-Kernel-Gateway port free? If something already holds 127.0.0.1:8888
    # (a leaked jupyter-kernel container), the DockerExecutor cannot bind it — skip rather than
    # fail/hang. We probe by trying to bind it ourselves, then immediately release it.
    import socket

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        probe.bind(("127.0.0.1", _DOCKER_KERNEL_PORT))
    except OSError:
        return False  # port already allocated -> Docker not serviceable for the kernel
    finally:
        probe.close()
    return True


def pytest_collection_modifyitems(config, items):
    run_live = os.environ.get("QUILL_LIVE_TESTS") == "1"
    # `sandbox` tests are off by default (06 §2: the offline suite is the guarantee; sandbox is a
    # heavy, environment-dependent extra — like `live`). They run only when EXPLICITLY enabled with
    # QUILL_SANDBOX_TESTS=1 AND Docker is genuinely serviceable (daemon up + kernel port 8888 free).
    # That makes the default offline run deterministic — it never fails or HANGS on a present-but-
    # unusable Docker (a leaked kernel container holding port 8888, the daemon down, the DockerExecutor
    # blocking on the kernel websocket) — and still lets a healthy CI box exercise them on demand.
    run_sandbox = os.environ.get("QUILL_SANDBOX_TESTS") == "1" and _sandbox_available()
    skip_live = pytest.mark.skip(reason="live: set QUILL_LIVE_TESTS=1 and HF_TOKEN to run")
    skip_sandbox = pytest.mark.skip(
        reason="sandbox: set QUILL_SANDBOX_TESTS=1 with a serviceable Docker "
        "(daemon up, port 8888 free) to run"
    )
    for item in items:
        if "live" in item.keywords and not run_live:
            item.add_marker(skip_live)
        if "sandbox" in item.keywords and not run_sandbox:
            item.add_marker(skip_sandbox)
