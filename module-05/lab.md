# Lab 5 — Run Quill in a sandbox: locked imports, a blocked import, a loop cap, clean teardown

**Goal:** give Quill `quill/sandbox.py` with the **frozen** `resolve_executor()` — the one
place that decides **where** Quill's generated Python runs (`QUILL_EXECUTOR` in
`{local, docker, e2b}`) and **what** it may import (locked to a least-privilege list, **never
`"*"`**) — wire it into `build_quill()`, run Quill inside a Docker container, and watch the
first layer of defense **block a dangerous import** and **cut off a runaway loop**.

**You'll see:** Quill's import lock refuse `import os` with an `InterpreterError` (offline, no
Docker or token), the same for a `while True` loop, and — with `QUILL_EXECUTOR=docker` — Quill
answer a question with its Python running inside a container that is then cleaned up (no
dangling container).

**Observable result:**

```bash
uv run python -m quill.demo_sandbox --attack
```

```text
[demo] executor='local'  authorized_imports=['pandas', 'numpy', 'matplotlib.*', 'json', 'statistics']
--- Attack 1: import os and run a shell command (data-exfil / arbitrary code) ---
import os
os.system('echo pwned > /tmp/quill_pwned.txt')
>>> BLOCKED by the executor: InterpreterError: ... Import of os is not allowed. ...
--- Attack 2: a runaway loop (resource abuse / denial of service) ---
x = 0
while True:
    x += 1
>>> BLOCKED by the executor: InterpreterError: ... Maximum number of 1000000 iterations in While loop exceeded
```

## Steps

1. **Setup** — copy the Module 4 state (the cumulative rule: M5 code must still pass the M1–M5
   smoke tests), then sync the pins. Module 5 adds the `[docker]` extra
   (`DockerExecutor`: `docker>=7.1.0` + `websocket-client`); `[e2b]` is optional for the swap.
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
     "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
   cp module-05/.env.example module-05/.env   # then add your HF token; never commit .env
   docker info                                # confirm the Docker daemon is running
   ```
   `data/sales.csv` is inherited from Module 2 unchanged. `pandas` and `matplotlib` are already
   there.

2. **Write `quill/sandbox.py` (the FROZEN sandbox contract).** It does ONE thing — decide the
   executor and the import lock — and it does **not** build the agent:
   ```python
   import os

   DEFAULT_EXECUTOR = "local"
   SUPPORTED_EXECUTORS = ("local", "docker", "e2b")  # NOTE: no "wasm" (removed in 1.26.0)

   # Quill's frozen least-privilege import list — NEVER the "*" wildcard.
   QUILL_AUTHORIZED_IMPORTS = ["pandas", "numpy", "matplotlib.*", "json", "statistics"]


   def resolve_executor() -> tuple[str, list[str]]:
       executor_type = os.environ.get("QUILL_EXECUTOR", DEFAULT_EXECUTOR).strip().lower()
       if executor_type not in SUPPORTED_EXECUTORS:
           raise ValueError(
               f"Unknown QUILL_EXECUTOR {executor_type!r}. "
               f"Supported executors: {', '.join(SUPPORTED_EXECUTORS)}. "
               "(Note: 'wasm' was removed from smolagents in 1.26.0 — it is not a valid value.)"
           )
       return executor_type, list(QUILL_AUTHORIZED_IMPORTS)  # a copy, so the constant can't drift
   ```
   `"matplotlib.*"` authorizes the package **and** its submodules (`matplotlib.pyplot`) — the
   same wildcard form as `"numpy.*"`. The effective `agent.authorized_imports` is this list
   **unioned with** smolagents' 11 always-on `BASE_BUILTIN_MODULES` (`collections`, `datetime`,
   `itertools`, `math`, `queue`, `random`, `re`, `stat`, `statistics`, `time`, `unicodedata` —
   they live in `utils.py`, not `local_python_executor.py`). `os`, `subprocess`, `socket` are in
   **neither** — so they are blocked.

3. **Wire `quill/agent.py`** — `build_quill` **stays here** (frozen contract, 06 §2) and now
   calls `resolve_executor()`:
   ```python
   from .sandbox import resolve_executor

   def build_quill(model=None):
       executor_type, authorized_imports = resolve_executor()
       return CodeAgent(
           tools=[load_dataset, profile_dataframe, save_chart(),
                  WebSearchTool(), VisitWebpageTool()],
           model=model or make_model(role="analyst"),
           executor_type=executor_type,                       # WHERE the code runs (M5)
           additional_authorized_imports=authorized_imports,  # WHAT it can import (M5)
           max_steps=8,
       )
   ```
   Do **not** duplicate the import list anywhere else, and do **not** move `build_quill` into
   `sandbox.py`. `CodeAgent` already supports `__enter__`/`__exit__`/`cleanup()`, so the agent
   it returns is a context manager (Step 6).

   > A remote executor (`docker`/`e2b`) **plus** `managed_agents` raises
   > `Exception("Managed agents are not yet supported with remote code execution.")`. That is
   > why multi-agents (Module 10) cannot run in Approach 1 — it needs **Approach 2** (the whole
   > agent inside one sandbox), the capstone (Module 15). We do not pass `managed_agents` here.

4. **Demo the isolation** (`quill/demo_sandbox.py`). `--attack` feeds two hostile snippets
   **straight to the agent's executor** (no model call, so it runs anywhere):
   ```python
   from smolagents.local_python_executor import InterpreterError

   def _run_snippet(agent, label, code):
       try:
           agent.python_executor(code)             # the SAME executor the agent uses each step
           print(">>> NOT blocked (unexpected).")
       except InterpreterError as exc:
           print(f">>> BLOCKED by the executor: InterpreterError: {exc}")

   # Attack 1 — blocked: 'os' is not in authorized imports.
   _run_snippet(agent, "import os", "import os\nos.system('echo pwned')\n")
   # Attack 2 — blocked: the executor caps While-loop iterations (MAX_WHILE_ITERATIONS = 1_000_000).
   _run_snippet(agent, "runaway loop", "x = 0\nwhile True:\n    x += 1\n")
   ```
   The error is the same `InterpreterError` (a `ValueError` subclass) the agent **captures**
   into the failing `ActionStep` mid-run — it never crashes your process, and `os.getcwd()`
   never reaches `final_answer`.

5. **Hardening (shown, optional to run).** For a remote deployment, pass hardening kwargs to the
   `DockerExecutor` and drop privileges in the image. Illustrative — Quill's default path does
   not require it:
   ```python
   from smolagents import CodeAgent
   with CodeAgent(
       tools=[], model=model, executor_type="docker",
       executor_kwargs={"container_run_kwargs": {
           "mem_limit": "512m",
           "cpu_quota": 50000,
           "pids_limit": 100,
           "security_opt": ["no-new-privileges"],
           "cap_drop": ["ALL"],
       }},
   ) as agent:
       ...
   ```
   In the Dockerfile, run as a non-root user (`USER nobody`). The four best-practice axes:
   **resource management** (mem/CPU/PIDs limits, timeouts), **security** (least privilege,
   network off when not needed, secrets via env never hard-coded), **environment** (minimal
   deps, pinned + patched base image), and **cleanup** (Step 6).

6. **Run it in Docker, then clean up.** Everything goes through the context manager so a remote
   sandbox is torn down even if the run raises:
   ```bash
   QUILL_EXECUTOR=docker uv run python -m quill.demo_sandbox
   docker ps -a    # no dangling jupyter-kernel container afterward
   ```
   ```python
   with build_quill() as agent:          # cleanup() runs on exit (stops + removes the container)
       result = agent.run(build_task("data/sales.csv", "How many rows?"))
   ```
   > **Real Approach-1 caveat you'll hit here.** Under a remote executor, smolagents *sends*
   > the agent's custom tools into the sandbox via `send_tools()`. Quill's `@tool` data tools
   > reference a module-level helper (`_read_table`), which the remote-serialization path
   > (`SimpleTool.to_dict()`) rejects — `agent.run()` then raises `ValueError: SimpleTool
   > validation failed for load_dataset: Name '_read_table' is undefined`. The container and
   > the import lock work; a **bare** `CodeAgent` (no custom tools) runs fine in Docker. Making
   > the data tools self-contained for remote sending is out of scope here.

7. **Tests** (`tests/smoke_test.py`). Offline (no token, no Docker): `resolve_executor()`
   defaults to `local`, validates `QUILL_EXECUTOR` (and rejects `wasm`), and returns the frozen
   import list; a `import os` on a local-executor agent is blocked (the run does not end in
   `success` and never returns the `os` value); and a `while True` loop hits the cap. The
   `sandbox`-marked tests run a snippet inside a real Docker container and clean up (skip if
   Docker is absent); `live`-marked tests skip without `HF_TOKEN`.
   ```bash
   uv run pytest module-05/tests/                                  # offline
   QUILL_EXECUTOR=docker uv run pytest -m sandbox module-05/tests/ # the Docker runs
   ```

## Try it yourself (not graded)

1. **Swap to E2B.** Sign up at [e2b.dev](https://e2b.dev) (free tier), put `E2B_API_KEY` in
   `.env`, `uv pip install "smolagents[e2b]==1.26.0"`, then run with `QUILL_EXECUTOR=e2b` and
   compare the spin-up time against Docker.
2. **Starve a snippet of memory.** Add `executor_kwargs={"container_run_kwargs": {"mem_limit":
   "256m", "cap_drop": ["ALL"]}}` to a `docker` run and watch a deliberately memory-hungry
   snippet get killed by the limit.

Verified against **smolagents 1.26.0**. Note: `executor_type="wasm"` does not exist in 1.26.0.
