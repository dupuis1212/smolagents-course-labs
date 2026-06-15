# Module 5 — Running Untrusted Code Safely: Sandboxing smolagents

A `CodeAgent` writes and runs LLM-generated Python every step. Until now Quill ran that code
**in your own process** via the `LocalPythonExecutor` — an AST allow-list interpreter that, in
the library's own words, **"is not a security sandbox."** This module gives Quill a **frozen
sandbox policy**: `quill/sandbox.py` with `resolve_executor()` — the **one place** that decides
**where** Quill's Python runs (`QUILL_EXECUTOR` in `{local, docker, e2b}`) and **what** it may
import (locked to a least-privilege list, **never `"*"`**). `build_quill()` now calls it, and
the returned agent is a context manager so a remote sandbox is torn down deterministically.

This is **Approach 1** (snippet-in-sandbox): the model and tools stay local; only the generated
snippets go to the container. (The whole-agent-in-sandbox **Approach 2** — the only topology
that also supports multi-agents — is the capstone, Module 15.)

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
  "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-05/.env.example module-05/.env   # then put your HF token in it
docker info                                # make sure Docker is running (for the docker path)
```

**See the import lock block a dangerous import — offline, no Docker or token needed:**

```bash
uv run python -m quill.demo_sandbox --attack
```

prints:

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

**Run Quill inside a real Docker sandbox** (Approach 1; makes a model call, so set `HF_TOKEN`):

```bash
QUILL_EXECUTOR=docker uv run python -m quill.demo_sandbox
```

The first run builds the `jupyter-kernel` image (~once, up to a few minutes), starts a
container, then cleans it up via `with build_quill(...) as agent:` — `docker ps -a` shows **no
dangling container** afterward. One **real Approach-1 caveat** surfaces here: under a remote
executor smolagents must *send* the agent's custom tools into the sandbox, and Quill's `@tool`
data tools reference a module-level helper that the remote-serialization path
(`SimpleTool.to_dict()`) rejects — so the docker normal-run reports that caveat rather than a
full toolbox answer. The sandbox boundary and the import lock work regardless (see `--attack`),
and a bare `CodeAgent` with no custom tools runs cleanly in Docker (the `sandbox` test proves
it). Making the data tools self-contained for remote sending is a later concern.

Run from inside `module-05/` so `data/sales.csv`, `outputs/`, and the `quill` package resolve.

## Test it

```bash
uv run pytest module-05/tests/                                  # offline (no token, no Docker)
QUILL_EXECUTOR=docker uv run pytest -m sandbox module-05/tests/ # also the real Docker runs
QUILL_LIVE_TESTS=1 uv run pytest module-05/tests/               # also the real LLM run (needs HF_TOKEN)
```

The offline tests run with no network. They prove the **frozen sandbox contract**:
`resolve_executor()` reads `QUILL_EXECUTOR`, defaults to `local`, and raises `ValueError` on
anything else (including a stale `"wasm"`); the import list is exactly
`["pandas", "numpy", "matplotlib.*", "json", "statistics"]` and never `"*"`; a `import os` on a
local-executor agent is **blocked** (`InterpreterError`, the `os` result never reaches
`final_answer`); and a `while True` loop is cut off by the iteration cap. The `sandbox`-marked
tests build a real Docker container, run a snippet inside it, and clean up — skipped cleanly if
Docker is absent. Every Module 2/3/4 test (the toolbox, the agent loop, `make_model()`, the
`Monitor` cost accessor) still passes here.

## The sandbox policy

| `QUILL_EXECUTOR` | Where Quill's Python runs | Isolation | Needs |
|---|---|---|---|
| `local` (default) | in-process `LocalPythonExecutor` (AST allow-list) | surface-area reduction — **NOT a security sandbox** | nothing |
| `docker` | a local Docker container (Jupyter Kernel Gateway) | real OS isolation | `[docker]` extra + a running daemon |
| `e2b` | an E2B cloud microVM (Firecracker) | real VM isolation | `[e2b]` extra + `E2B_API_KEY` |

`additional_authorized_imports` is locked to Quill's minimal list **regardless of executor** —
`["pandas", "numpy", "matplotlib.*", "json", "statistics"]`, extended only by explicit addition,
**never the `"*"` wildcard** (which authorizes every import; the library warns *"Use this at
your own risk!"*). Least privilege beats convenience: a data snippet has no business importing
`os`, `socket`, or `subprocess`. `resolve_executor() -> (executor_type, additional_authorized_imports)`
is the **frozen contract** every later module reuses.

## What this module deliberately does NOT do

- **No multi-agents** (Module 10). Quill stays a single `CodeAgent`. We only state the
  constraint: a remote executor **+** `managed_agents` raises
  `Exception("Managed agents are not yet supported with remote code execution.")` — so
  multi-agents needs **Approach 2** (the whole agent inside one sandbox), which is **Module 15**.
- **No Approach 2** (manual sandbox wrapping the whole agent). We do Approach 1
  (`executor_type=`). Approach 2 is the capstone.
- No `step_callbacks` / multi-turn memory (Module 6), no telemetry (Module 14), no `GradioUI`
  (Module 13).
- **No `executor_type="wasm"`** — it was removed from smolagents in 1.26.0 (PR #2321). Many
  stale tutorials still show it; the valid set is `{local, docker, e2b, modal, blaxel}`.

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
