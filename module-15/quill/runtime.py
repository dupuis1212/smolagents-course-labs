"""Quill's production runtime — Approach 2 + hardening (NEW, Module 15, the capstone).

Module 15 adds **no new feature**. It ASSEMBLES the fourteen pieces Quill already has and
**hardens** them into something you can hand to someone else. This module is the only NEW file in
the capstone; ``build_quill`` STAYS in ``quill/agent.py`` (the frozen construction entry point) and
``runtime.py`` only ORCHESTRATES it. Two ideas live here, and they are the whole module:

1. **Approach 2 — run the WHOLE multi-agent system inside a sandbox.**
   Up to Module 14, Quill's team (manager ``CodeAgent`` + ``web_researcher`` + optional
   ``vision_browser``) ran in ``executor_type="local"`` ON PURPOSE: ``local`` is not a security
   sandbox (the ``LocalPythonExecutor`` is an AST allow-list, "it is not a security sandbox" in the
   library's own words — Module 5). You CANNOT just flip ``QUILL_EXECUTOR=docker`` to fix that,
   because a remote ``executor_type`` PLUS ``managed_agents`` raises (verified, smolagents 1.26.0
   ``agents.py::create_python_executor``)::

       if self.managed_agents:
           raise Exception("Managed agents are not yet supported with remote code execution.")

   That is **Approach 1** (snippet-in-sandbox): the model/agent stay LOCAL, only the generated
   Python snippets go to the container, and secrets (the HF token) are NEVER shipped in — so a
   sub-agent could not authenticate its own LLM from inside the box. Approach 1 therefore cannot do
   multi-agent.

   **Approach 2** is the answer the brief's diagram (06 §1) draws: you create the sandbox YOURSELF
   (a hardened Docker container here; an ``e2b_code_interpreter.Sandbox()`` as the option), copy the
   ``quill`` package + ``data/`` in, pass ``HF_TOKEN`` as a container env var, and run an entrypoint
   that does ``build_quill(...).run(...)`` *inside* the container. The manager AND its sub-agents now
   execute in the sandbox; the only thing crossing the boundary is the question (in) and the
   serialized ``QuillReport`` (out). That is the ONLY way isolation + multi-agent coexist.

   We deliberately keep ``QUILL_EXECUTOR`` (the frozen M5 contract) for the INNER agent at
   ``"local"`` when we run it inside the sandbox: the container already IS the isolation boundary, so
   re-nesting a remote executor inside it would be both pointless and (with the team) illegal. The
   sandbox is the boundary; the inner agent runs ``local`` *within* it.

2. **Hardening (T12.14)** — four levers, each with a defensible number:
   - **Step caps**: ``max_steps`` is bounded on the manager (Quill's ``build_quill`` default is 8)
     AND on each sub-agent (``web_researcher`` at 10, M10). An agent that does not converge raises
     ``AgentMaxStepsError`` instead of looping forever (M8).
   - **Timeouts**: the ``LocalPythonExecutor`` already caps a single execution at
     ``MAX_EXECUTION_TIME_SECONDS = 30`` (M5); the Docker container adds resource limits
     (``mem_limit`` / ``cpu_quota`` / ``pids_limit``) so a pathological pandas job cannot exhaust the
     host.
   - **Bounded retries**: ``run_with_bounded_retries`` retries a failed run at most ``MAX_RETRIES``
     (2) times. Opinion (style guide): cap retries at 2 — beyond that you are burning tokens on a
     broken loop.
   - **Idempotence**: ``run_signature`` / ``idempotent_chart_stem`` derive a deterministic stem from
     (question + dataset) so a re-run writes the SAME chart path instead of accumulating files, and
     an output dir is created idempotently (``exist_ok=True``).

Hardening flags for the Docker container (research-04 §5, used EXACTLY): ``mem_limit="512m"``,
``cpu_quota=50000`` (50% of one CPU), ``pids_limit=100``, ``security_opt=["no-new-privileges"]``,
``cap_drop=["ALL"]``, and the entrypoint drops to ``USER nobody``. "No solution will be 100% safe"
(the doc) — Approach 2 isolates the SYSTEM, but web/tool content (prompt injection, M5) is still a
vector, which is why the ``final_answer_checks`` (M8) and the import lock (M5) stay on inside.

What this module does NOT do (06 §6 / the brief): no new tool, sub-agent, or ``QuillReport`` field;
no Approach 1 for the team (it raises); no ``"*"`` import wildcard; no ``agent.logs`` (removed in
1.21.0 — read a run via ``agent.memory.steps`` / ``agent.replay()`` / ``agent.visualize()``).
"""
from __future__ import annotations

import hashlib
import os
import pathlib

# Quill's two sandbox backends for Approach 2 (the brief recommends Docker for the lab, E2B as the
# option). These are NOT QUILL_EXECUTOR values — Approach 2 creates the sandbox by hand, it does not
# pass executor_type. (executor_type="approach2" does NOT exist; do not let a tutorial tell you it.)
SUPPORTED_SANDBOX_BACKENDS = ("docker", "e2b")
DEFAULT_SANDBOX_BACKEND = "docker"

# Bounded retries (T12.14). 2 is the persona's defensible cap: a third attempt on a run that failed
# twice is almost always a broken loop burning tokens, not a transient blip. Override per call.
MAX_RETRIES = 2

# The Docker hardening flags (research-04 §5 — used EXACTLY). They go to the container's run kwargs:
# half a CPU, half a gig of RAM, a PID cap (fork-bomb guard), no privilege escalation, every Linux
# capability dropped. The entrypoint also drops to USER nobody. None of this is "100% safe" — it is
# defense in depth around an isolation boundary that already exists (the container).
DOCKER_HARDENING = {
    "mem_limit": "512m",            # cap memory (a runaway pandas merge cannot eat the host)
    "cpu_quota": 50000,             # 50% of one CPU (cpu_period defaults to 100000)
    "pids_limit": 100,              # cap processes/threads — a fork bomb hits this wall
    "security_opt": ["no-new-privileges"],  # no setuid escalation inside the container
    "cap_drop": ["ALL"],            # drop every Linux capability the kernel would grant
}

# The non-root user the in-sandbox entrypoint runs as (research-04 §5: `USER nobody`). Code the LLM
# wrote should never run as root, even inside a throwaway container.
SANDBOX_USER = "nobody"


def run_signature(question: str, dataset: str) -> str:
    """A deterministic signature for a (question, dataset) run — the idempotence key (T12.14).

    A re-run with the SAME question and dataset must not double-write charts or double-count cost.
    We hash the normalised pair to a short hex stem; ``idempotent_chart_stem`` turns it into a chart
    filename and a run's output dir is created with ``exist_ok=True``. The hash is stable across
    processes (no PYTHONHASHSEED dependence — we use sha256, not ``hash()``).

    Args:
        question: the analysis question.
        dataset: the dataset path (or comma-joined paths) the run analyses.

    Returns:
        A 12-char hex digest, e.g. ``"a1b2c3d4e5f6"``.
    """
    payload = f"{question.strip()}\x00{dataset.strip()}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def idempotent_chart_stem(question: str, dataset: str) -> str:
    """A deterministic chart base filename for a run — so a re-run overwrites, never accumulates.

    ``save_chart`` (FROZEN M3 signature, unchanged) takes an optional ``filename``. Passing this stem
    makes the saved path a pure function of (question, dataset): the same run produces
    ``outputs/quill-<sig>.png`` every time, so retries and UI re-submits overwrite the same file
    instead of leaving ``chart-<timestamp>.png`` litter. Idempotence by AGREEMENT (the caller passes
    the stem), NOT by changing ``save_chart``'s contract.

    Returns:
        ``"quill-<run_signature>"`` (no extension — ``save_chart`` appends ``.png``).
    """
    return f"quill-{run_signature(question, dataset)}"


def ensure_outputs_dir(path: str = "outputs") -> str:
    """Create the outputs directory idempotently (``exist_ok=True``) and return it (T12.14).

    A second run never fails because the dir already exists, and never wipes a prior run's charts.
    """
    pathlib.Path(path).mkdir(parents=True, exist_ok=True)
    return path


def run_with_bounded_retries(run_fn, *, max_retries: int = MAX_RETRIES):
    """Call ``run_fn()`` and retry on failure at most ``max_retries`` times (T12.14 — the cap).

    Production reality: a transient blip (a flaky network read, a sandbox that did not warm up) is
    worth one retry; a run that fails TWICE is almost always a broken loop, and a third attempt just
    burns tokens. So we cap retries at 2 (override per call) and re-raise the LAST error if every
    attempt fails — we never swallow it (no silent ``try/except``; the caller sees what broke).

    Args:
        run_fn: a 0-arg callable that performs ONE run and returns its result.
        max_retries: how many EXTRA attempts after the first (default ``MAX_RETRIES`` = 2, so up to
            3 total). Must be >= 0.

    Returns:
        Whatever ``run_fn()`` returns on the first attempt that succeeds.

    Raises:
        ValueError: if ``max_retries`` is negative (fail loud — a negative cap is a bug).
        Exception: the LAST attempt's exception, re-raised, if every attempt fails.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0, got {max_retries}.")

    attempts = max_retries + 1  # the first try plus the bounded retries
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return run_fn()
        except Exception as exc:  # noqa: BLE001 — we re-raise the last one; this is a BOUNDED loop
            last_exc = exc
            print(
                f"[quill.runtime] run attempt {attempt}/{attempts} failed: "
                f"{type(exc).__name__}: {exc}"
            )
    # Every attempt failed — re-raise the last error (never mask it; this is not a silent except).
    assert last_exc is not None
    raise last_exc


def resolve_sandbox_backend(backend: str | None = None) -> str:
    """Resolve the Approach-2 sandbox backend: the explicit arg, else ``QUILL_SANDBOX_BACKEND``,
    else ``QUILL_EXECUTOR`` (docker/e2b), else ``docker``.

    Note this is the backend for the SANDBOX we create by hand (Approach 2), distinct from the inner
    agent's ``QUILL_EXECUTOR`` (which stays ``local`` *inside* the sandbox). If ``QUILL_EXECUTOR`` is
    a remote value (docker/e2b) we reuse it as the sandbox backend, so a reader who set
    ``QUILL_EXECUTOR=e2b`` gets an E2B sandbox without a second env var.

    Raises:
        ValueError: on an unknown backend (fail loud — never silently pick docker).
    """
    if backend is None:
        backend = os.environ.get("QUILL_SANDBOX_BACKEND")
    if backend is None:
        executor = os.environ.get("QUILL_EXECUTOR", "").strip().lower()
        backend = executor if executor in SUPPORTED_SANDBOX_BACKENDS else DEFAULT_SANDBOX_BACKEND
    backend = backend.strip().lower()
    if backend not in SUPPORTED_SANDBOX_BACKENDS:
        raise ValueError(
            f"Unknown sandbox backend {backend!r}. "
            f"Approach-2 backends: {', '.join(SUPPORTED_SANDBOX_BACKENDS)}."
        )
    return backend


# The entrypoint that runs INSIDE the sandbox (Approach 2). It is a self-contained Python program:
# the sandbox has the `quill` package + `data/` copied in and HF_TOKEN as an env var, so it can do
# `build_quill(...).run(...)` — the WHOLE manager + sub-agents — locally *within* the container, and
# print the rendered report to stdout. The inner agent runs executor_type="local" because the
# container already IS the boundary (and a remote executor + managed_agents would raise).
#
# This string is what the lab copies into the container (or `Sandbox.run_code`s for E2B). It is kept
# here, beside the orchestration, so the "what runs inside" is reviewable in one place. It never uses
# the removed `agent.logs` (a run is read via agent.memory.steps / replay() — 06 §6).
SANDBOX_ENTRYPOINT = '''\
import os, sys
# Approach 2: the WHOLE system runs here, inside the sandbox. The inner agent stays
# executor_type="local" — the container is the isolation boundary; we do NOT nest a remote
# executor (and a remote executor + managed_agents would raise the M10 exception anyway).
os.environ.setdefault("QUILL_EXECUTOR", "local")
from quill.agent import build_quill, build_report_task
from quill.report import QuillReport

question = sys.argv[1] if len(sys.argv) > 1 else "Summarize data/sales.csv."
dataset = sys.argv[2] if len(sys.argv) > 2 else "data/sales.csv"

# The full team (manager + web_researcher) runs INSIDE the sandbox — the only way multi-agent and
# isolation coexist (Approach 2). HF_TOKEN reached us as a container env var (never hard-coded).
with build_quill() as agent:
    output = agent.run(build_report_task(dataset, question))

print("===== QUILL REPORT (inside the sandbox) =====")
print(output.to_markdown() if isinstance(output, QuillReport) else output)
'''


def build_hardened_container_kwargs() -> dict:
    """The hardened ``container_run_kwargs`` for an Approach-2 Docker sandbox (research-04 §5).

    Returns a fresh dict of the EXACT hardening flags (mem/cpu/pids caps, no-new-privileges,
    cap_drop ALL, user nobody) to pass to ``docker``'s ``containers.run`` (or smolagents'
    ``DockerExecutor(container_run_kwargs=...)`` if you sandbox via that path). A fresh copy so a
    caller can tweak one flag without mutating the module constant.
    """
    kwargs = dict(DOCKER_HARDENING)
    kwargs["user"] = SANDBOX_USER
    return kwargs


def run_quill_in_docker_sandbox(
    question: str,
    dataset: str = "data/sales.csv",
    *,
    image: str = "python:3.11-slim",
    quill_package_dir: str | None = None,
    timeout: int = 600,
):
    """Approach 2 via Docker: create a HARDENED container by hand and run the WHOLE team inside it.

    This is the capstone's headline. We do NOT use ``executor_type="docker"`` (Approach 1, which
    raises with ``managed_agents``). Instead we:

    1. start a hardened ``python:3.11-slim`` container (the flags from
       :func:`build_hardened_container_kwargs`: ``mem_limit``/``cpu_quota``/``pids_limit``/
       ``no-new-privileges``/``cap_drop=ALL``/``USER nobody``),
    2. copy the ``quill`` package + the dataset in (Approach 2 ships the CODE, so the agent and its
       sub-agents run inside),
    3. pass ``HF_TOKEN`` as a container ENV var (never hard-coded — secrets in env, M5/checklist),
    4. ``pip install`` smolagents + the data stack, then run :data:`SANDBOX_ENTRYPOINT` inside,
    5. capture the rendered ``QuillReport`` markdown from the container's stdout,
    6. tear the container down in a ``finally`` (resource management taught explicitly, not a silent
       except — 03 §3): no dangling container even if the run raises.

    Requires Docker on the host and the ``[docker]`` extra (``docker>=7.1.0``). This function is
    exercised by the ``sandbox``-marked test (and the documented ``python -m quill --sandboxed`` CLI
    path); it is heavy (an image pull + a pip install + a live LLM run) so it is NEVER in the offline
    suite.

    Args:
        question: the analysis question.
        dataset: the dataset path (relative to the package dir copied into the sandbox).
        image: the base image to harden (a small Python image; pinned by the caller).
        quill_package_dir: the dir whose ``quill/`` + ``data/`` are copied in (defaults to THIS
            module's package parent — i.e. ``module-15/``).
        timeout: a hard wall-clock cap (seconds) on the in-sandbox run — a timeout (T12.14).

    Returns:
        The container's stdout (the rendered report markdown plus any trajectory the entrypoint
        prints). The caller asserts on the ``QuillReport`` markdown.

    Raises:
        RuntimeError: with a fix-it message if the ``[docker]`` extra / a daemon is missing.
    """
    try:
        import docker
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Approach 2 (Docker) needs the [docker] extra: "
            "uv pip install 'smolagents[docker]==1.26.0' (docker>=7.1.0)."
        ) from exc

    import io
    import tarfile

    package_root = pathlib.Path(quill_package_dir or pathlib.Path(__file__).resolve().parents[1])
    client = docker.from_env()

    # Start the hardened container. It sleeps so we can copy code in and exec the entrypoint; the
    # caps mean even hostile generated code cannot exhaust the host (mem/cpu/pids) or escalate.
    hardening = build_hardened_container_kwargs()
    container = client.containers.run(
        image,
        command=["sleep", str(timeout)],
        detach=True,
        environment={
            # Secrets cross the boundary as ENV, never baked into the image (M5/checklist). Approach
            # 2 NEEDS this: the sub-agents authenticate their LLM calls from inside the sandbox.
            "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
            "QUILL_EXECUTOR": "local",  # the container is the boundary; inner agent runs local
            "QUILL_MODEL_ID": os.environ.get("QUILL_MODEL_ID", ""),
        },
        **hardening,
    )
    try:
        # Copy quill/ + data/ into /quill-app inside the container (Approach 2 ships the CODE).
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(str(package_root / "quill"), arcname="quill")
            data_dir = package_root / "data"
            if data_dir.exists():
                tar.add(str(data_dir), arcname="data")
        tar_stream.seek(0)
        container.exec_run("mkdir -p /quill-app", user="root")
        container.put_archive("/quill-app", tar_stream.getvalue())
        # Write the entrypoint program inside the container.
        entry_b64 = _b64(SANDBOX_ENTRYPOINT)
        container.exec_run(
            ["sh", "-c", f"echo {entry_b64} | base64 -d > /quill-app/_run.py"], user="root"
        )
        # Install the runtime deps inside the sandbox, then run the WHOLE team as `nobody`.
        container.exec_run(
            ["sh", "-c", "pip install --no-cache-dir 'smolagents[toolkit]==1.26.0' "
             "'huggingface_hub>=1.0,<2' 'pandas>=2.2.3' matplotlib rank-bm25 >/dev/null 2>&1"],
            user="root",
        )
        result = container.exec_run(
            ["python", "/quill-app/_run.py", question, dataset],
            workdir="/quill-app",
            user=SANDBOX_USER,  # drop to nobody to RUN the LLM-generated code (research-04 §5)
        )
        return result.output.decode("utf-8", errors="replace")
    finally:
        # Resource management taught explicitly (03 §3 — NOT a silent except): always tear the
        # container down, even if the run raised, so there is no dangling container.
        container.remove(force=True)


def _b64(text: str) -> str:
    """base64-encode text for safe shell transport into the container."""
    import base64

    return base64.b64encode(text.encode("utf-8")).decode("ascii")


def run_quill_in_e2b_sandbox(
    question: str,
    dataset: str = "data/sales.csv",
    *,
    quill_package_dir: str | None = None,
):
    """Approach 2 via E2B (the OPTION): create an ``e2b_code_interpreter.Sandbox`` and run the team
    inside it.

    The same shape as the Docker path — create the sandbox, upload the ``quill`` package + dataset,
    pass ``HF_TOKEN`` to the sandbox env, and run :data:`SANDBOX_ENTRYPOINT` inside it — but the
    boundary is an E2B cloud microVM instead of a local container. Swapping Docker <-> E2B here is
    the lab's first "Try it yourself". Requires the ``[e2b]`` extra (``e2b-code-interpreter>=1.0.3``)
    and ``E2B_API_KEY``.

    Raises:
        RuntimeError: with a fix-it message if the ``[e2b]`` extra / the key is missing.
    """
    try:
        from e2b_code_interpreter import Sandbox
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Approach 2 (E2B) needs the [e2b] extra: "
            "uv pip install 'smolagents[e2b]==1.26.0' (e2b-code-interpreter>=1.0.3) and E2B_API_KEY."
        ) from exc

    package_root = pathlib.Path(quill_package_dir or pathlib.Path(__file__).resolve().parents[1])
    sandbox = Sandbox()  # the E2B sandbox IS the isolation boundary (Approach 2)
    try:
        # Upload the quill package + dataset into the sandbox, then run the whole team inside it.
        for path in (package_root / "quill").rglob("*.py"):
            rel = path.relative_to(package_root)
            sandbox.files.write(str(rel), path.read_text(encoding="utf-8"))
        dataset_path = package_root / dataset
        if dataset_path.exists():
            sandbox.files.write(dataset, dataset_path.read_text(encoding="utf-8"))
        sandbox.commands.run(
            "pip install 'smolagents[toolkit]==1.26.0' 'huggingface_hub>=1.0,<2' "
            "'pandas>=2.2.3' matplotlib rank-bm25",
        )
        sandbox.files.write("_run.py", SANDBOX_ENTRYPOINT)
        execution = sandbox.commands.run(
            f"HF_TOKEN={os.environ.get('HF_TOKEN', '')} QUILL_EXECUTOR=local "
            f"python _run.py {question!r} {dataset!r}",
        )
        return execution.stdout
    finally:
        # Resource management taught explicitly: always close the sandbox (no leaked microVM).
        sandbox.kill()


def build_quill_app(
    *,
    model=None,
    planning_interval: int | None = None,
    telemetry: bool = True,
):
    """Assemble the production Quill — the capstone's single in-process entry point.

    The brief's "assemble a single entrypoint wiring manager + web_researcher + retriever +
    QuillReport + telemetry + planning; hardening". It does, in ORDER, the things a production run
    needs and that the fourteen modules each own a piece of:

    1. **Telemetry FIRST** (M14, 06 §2): ``instrument()`` runs BEFORE the agent is built, so no early
       span is lost. With ``QUILL_TELEMETRY=none`` (default) it is a clean no-op — never broken.
    2. **Planning ON** (M7): default to ``DEFAULT_PLANNING_INTERVAL`` so a multi-step run re-centres
       on the goal (one extra LLM call, worth it on the longer capstone jobs).
    3. **The frozen team + RAG + report contract** come for free from ``build_quill`` (M10 manager +
       ``web_researcher``, M12 ``RetrieverTool``, M8 ``final_answer_checks`` -> ``QuillReport``).
    4. **Bounded steps** are already set inside ``build_quill`` (manager ``max_steps=8``,
       ``web_researcher`` ``max_steps=10``) — hardening (T12.14).

    ``build_quill`` STAYS the construction owner (06 §2); this only wraps it with the production
    wiring. The returned agent is a context manager — use ``with build_quill_app() as agent:`` so any
    sandbox is torn down. For the FULLY isolated multi-agent run, use :func:`run_quill_sandboxed`
    (Approach 2) instead — this in-process app runs the team on the chosen ``QUILL_EXECUTOR``.

    Args:
        model: a ``smolagents.Model``; ``None`` -> ``make_model`` (tests pass a fake model).
        planning_interval: the manager's planning cadence; ``None`` -> ``DEFAULT_PLANNING_INTERVAL``.
        telemetry: call ``instrument()`` before building (default ``True``; no-op unless a backend is
            configured). Pass ``False`` in tests that assert ordering themselves.

    Returns:
        The assembled, hardened Quill ``CodeAgent`` (manager over its team), ready to ``.run(...)``.
    """
    # Import here so quill.runtime imports cheaply and tests can patch these on this module.
    from .agent import DEFAULT_PLANNING_INTERVAL, build_quill
    from .telemetry import instrument

    if telemetry:
        # 06 §2 ORDERING: instrument BEFORE building the agent (no-op when QUILL_TELEMETRY=none).
        instrument()

    cadence = DEFAULT_PLANNING_INTERVAL if planning_interval is None else planning_interval
    # build_quill is the FROZEN construction owner — runtime only adds the production wiring.
    return build_quill(model=model, planning_interval=cadence)


def run_quill_sandboxed(question: str, dataset: str = "data/sales.csv", *, backend: str | None = None):
    """Run the WHOLE Quill team inside a sandbox (Approach 2), dispatching on the backend.

    The single entry point the ``--sandboxed`` CLI calls. Resolves the backend
    (:func:`resolve_sandbox_backend`: docker by default, e2b as the option) and runs the team inside
    it via :func:`run_quill_in_docker_sandbox` / :func:`run_quill_in_e2b_sandbox`, wrapped in
    :func:`run_with_bounded_retries` so a transient sandbox blip is retried at most twice.

    Returns:
        The sandboxed run's stdout (the rendered ``QuillReport`` markdown).
    """
    resolved = resolve_sandbox_backend(backend)
    runner = run_quill_in_docker_sandbox if resolved == "docker" else run_quill_in_e2b_sandbox
    return run_with_bounded_retries(lambda: runner(question, dataset))


__all__ = [
    "SUPPORTED_SANDBOX_BACKENDS",
    "DEFAULT_SANDBOX_BACKEND",
    "MAX_RETRIES",
    "DOCKER_HARDENING",
    "SANDBOX_USER",
    "SANDBOX_ENTRYPOINT",
    "run_signature",
    "idempotent_chart_stem",
    "ensure_outputs_dir",
    "run_with_bounded_retries",
    "resolve_sandbox_backend",
    "build_hardened_container_kwargs",
    "build_quill_app",
    "run_quill_in_docker_sandbox",
    "run_quill_in_e2b_sandbox",
    "run_quill_sandboxed",
]
