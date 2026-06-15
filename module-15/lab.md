# Lab 15 — Capstone: ship Quill v1.0 (Approach 2 + hardening + release)

**Goal:** run the **whole multi-agent Quill inside a sandbox (Approach 2)**, harden it (bounded
`max_steps`, execution timeouts, retries ≤ 2, idempotent outputs), keep telemetry on and the eval
gate green, expose it as a Gradio Space, and tag the repo **`v1.0`**. This module **assembles and
hardens** — it adds no new feature.

**You'll see:** why `executor_type="docker"` + `managed_agents` raises (and how Approach 2 solves
it); the whole team running inside a hardened Docker container; a deterministic, idempotent
`save_chart` path; bounded retries that re-raise instead of looping; the eval gate as a release
blocker; the production checklist as a shareable asset; and a `git tag v1.0`.

**Observable result:**

```bash
QUILL_EXECUTOR=docker uv run python -m quill --sandboxed \
  "Analyze data/sales.csv vs data/customers.csv and tell me which segment is churning fastest, with a chart and sources."
```

→ Quill starts a **hardened Docker container**, runs the manager + sub-agents **inside it** (Approach
2), delegates web research to `web_researcher`, writes/saves a chart via `save_chart`, and returns a
Markdown report with `[n]` citations mapped to `QuillReport.sources`. Telemetry emits nested spans
(manager → sub-agent). Then:

```bash
uv run python -m quill.eval.run_evals --out eval/results/run-v1.0.json
```

→ the **eval gate** passes (TSR ≥ the golden-set floor) — exit code **0**; otherwise non-zero
("green or no ship"). Finally `git tag v1.0`.

---

## Step 0 — Setup (copy the cumulative state, install the full environment)

This module starts from the Module 14 state (`module-14/quill/` copied to `module-15/quill/`). The
code of M15 must still pass the smoke tests of every module ≤ 15 (06 §5.3). **No new feature is
created** — only Approach-2 orchestration, hardening, the checklist, and the tag.

```bash
cp -r module-14 module-15        # already done in this repo

# the full course environment (Approach 2 needs [docker] = docker>=7.1.0, or [e2b])
uv pip install "smolagents[toolkit,litellm,openai,e2b,docker,mcp,telemetry,gradio,vision]==1.26.0" \
  "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib rank-bm25
```

Make sure Docker is installed and running (or have an E2B account). Run the system once in `local`
to confirm the Module 14 baseline still works:

```bash
uv run python -m quill "Which category grew fastest?" --data data/sales.csv
```

---

## Step 1 — Why you cannot just flip `QUILL_EXECUTOR=docker` (the constraint)

`local` is an AST allow-list, **not a sandbox**. But a remote executor + `managed_agents` raises:

```python
# smolagents/agents.py::create_python_executor (1.26.0)
if self.managed_agents:
    raise Exception("Managed agents are not yet supported with remote code execution.")
```

This is **Approach 1**: the model/agent stay local, only generated snippets go to the container, and
**secrets never enter the box** — so a sub-agent could not authenticate its LLM from inside. Approach
1 cannot do multi-agent. Prove the exception offline:

```python
import os
os.environ["QUILL_EXECUTOR"] = "docker"
from quill.agent import build_quill
build_quill()   # default team + remote executor -> raises the exact exception (no daemon touched)
```

---

## Step 2 — Approach 2: run the whole system inside the sandbox (`quill/runtime.py`)

`quill/runtime.py` (the only NEW module) creates the sandbox **by hand** and runs
`build_quill(...).run(...)` inside it. It does **not** set `executor_type` for the team (there is no
`executor_type="approach2"`). The Docker path (recommended for the lab):

```python
# quill/runtime.py (abridged)
container = client.containers.run(
    "python:3.11-slim", command=["sleep", "600"], detach=True,
    environment={"HF_TOKEN": os.environ.get("HF_TOKEN", ""),   # secrets cross as ENV, never hard-coded
                 "QUILL_EXECUTOR": "local"},                    # the container IS the boundary
    **build_hardened_container_kwargs(),                        # the hardening flags (Step 3)
)
try:
    # copy quill/ + data/ in, pip install the deps, then run the entrypoint as `nobody`
    container.put_archive("/quill-app", tar_of_quill_and_data)
    container.exec_run(["python", "/quill-app/_run.py", question, dataset], user="nobody")
finally:
    container.remove(force=True)   # resource mgmt taught explicitly (03 §3), not a silent except
```

The entrypoint that runs **inside** the box (`runtime.SANDBOX_ENTRYPOINT`) is just:

```python
os.environ.setdefault("QUILL_EXECUTOR", "local")
from quill.agent import build_quill, build_report_task
with build_quill() as agent:                 # the WHOLE team runs here, inside the sandbox
    output = agent.run(build_report_task(dataset, question))
print(output.to_markdown())
```

**Why not `executor_type="docker"` + `managed_agents`?** It raises (Step 1). Approach 1 ships no
secrets into the sandbox, so a sub-agent cannot authenticate — we run the whole system inside the
sandbox = **Approach 2**, the only way isolation + multi-agent coexist.

E2B is the option (one backend swap — the first "Try it yourself"): `from e2b_code_interpreter import
Sandbox`, create it, upload `quill/` + `data/`, run the same entrypoint. `run_quill_sandboxed`
dispatches on `resolve_sandbox_backend()` (docker by default, e2b as the option).

---

## Step 3 — Hardening (T12.14)

Four levers in `quill/runtime.py` + `quill/agent.py`:

- **Step caps** — `max_steps` bounded on the manager (**8**) and `web_researcher` (**10**). A
  non-converging agent raises `AgentMaxStepsError` (M8) instead of looping.
- **Timeouts / resource limits** — the `LocalPythonExecutor` already caps a single execution at
  `MAX_EXECUTION_TIME_SECONDS = 30` (M5); the Docker container adds the hardening flags
  (research-04 §5, used exactly):

  ```python
  DOCKER_HARDENING = {
      "mem_limit": "512m", "cpu_quota": 50000, "pids_limit": 100,
      "security_opt": ["no-new-privileges"], "cap_drop": ["ALL"],
  }   # + the entrypoint runs as USER nobody
  ```

- **Bounded retries** — `run_with_bounded_retries(fn, max_retries=2)`: up to 3 attempts, then it
  **re-raises the last error** (never a silent except). *Cap retries at 2 — beyond that you're
  burning tokens on a broken loop.*
- **Idempotence** — `run_quill_report` sets a deterministic `run_signature(question, dataset)` around
  the run; an **un-named** `save_chart` then writes `outputs/quill-<sig>.png`, so a re-run overwrites
  the same file (no `chart-<timestamp>.png` litter, no double-count). `save_chart`'s frozen M3
  signature is untouched. The outputs dir is created with `exist_ok=True`.

Read a run without a backend: `agent.memory.steps`, `agent.replay()`, `agent.visualize()` — **never**
`agent.logs` (removed in 1.21.0).

---

## Step 4 — Telemetry + evals (recall M14, integrate)

`SmolagentsInstrumentor().instrument()` runs **before** the agent is built (frozen M14,
`quill/telemetry.py`; `build_quill_app` enforces the order). Run the eval harness as the **release
gate** — it exits non-zero if the TSR drops below the floor:

```bash
uv run python -m quill.eval.run_evals --out eval/results/run-v1.0.json   # exit 0 = green or no ship
```

---

## Step 5 — UI / Space (recall M13)

`GradioUI` wraps the hardened Quill (`make_ui(build_quill())`, `reset_agent_memory=False`,
`file_upload_folder="uploads"`). `push_to_hub` exports the Space artefacts (`agent.json`,
`prompts.yaml`, `tools/`, `managed_agents/`, `app.py`, `requirements.txt`). The push itself stays
optional (it needs an HF write token).

---

## Step 6 — The production checklist

`PRODUCTION-CHECKLIST.md` (the shareable asset) distils the whole course into checkable items across
**Security/sandbox**, **Reliability**, **Cost/perf**, **Observability/quality**, **Deploy** — each
item tied to its module. Keep it next to the code so a reviewer can run down it before a release.

---

## Step 7 — Demo end to end + tag `v1.0`

```bash
# Approach 2: the whole team inside a hardened container
QUILL_EXECUTOR=docker uv run python -m quill --sandboxed \
  "Which segment is churning fastest? Use the web for the SaaS benchmark, with a chart and sources."

# read the trajectory without a backend (NEVER agent.logs)
uv run python -m quill.agent data/sales.csv "Which category grew fastest?"   # prints agent.replay()

# gate green, then tag the release
uv run python -m quill.eval.run_evals --out eval/results/run-v1.0.json
git tag v1.0
```

---

## Step 8 — Test it

```bash
uv run pytest module-15/tests/                                       # offline (no token, no network, no Docker)
QUILL_LIVE_TESTS=1 uv run pytest module-15/tests/ -m "sandbox and live"  # Approach 2 in a real container
```

`tests/smoke_test.py` asserts (offline): a remote executor + `managed_agents` raises the exact
exception (Approach 1 can't do multi-agent → Approach 2); `save_chart` paths are idempotent (same
question ⇒ same path); retries are bounded (≤ 2, re-raise); `max_steps` is bounded on manager and
sub-agent; the eval gate exits non-zero below the threshold; the Docker hardening flags match
research-04 §5; and no `agent.logs` anywhere. The `sandbox`+`live` test runs the whole team inside a
real hardened container (Approach 2). **Budget for the live/sandbox run:** one container + a handful
of LLM calls + a pip install inside the box; a full multi-agent + vision + RAG run can be **dozens**
of LLM calls — the HF free tier is ~$0.10/month (as of smolagents 1.26.0), so cap it.

---

## Try it yourself

1. **Swap the Approach-2 backend from Docker to E2B** in `runtime.py`
   (`QUILL_SANDBOX_BACKEND=e2b` / `QUILL_EXECUTOR=e2b`) and compare startup / cost / isolation.
2. **Add an item to `PRODUCTION-CHECKLIST.md`** (e.g. "rate-limit per user") **and write the
   `smoke_test.py` assertion that verifies it** — close the loop between the checklist and the tests.

## What this lab does NOT do

- **No new feature** (no new tool / sub-agent / `QuillReport` field) — assembly and hardening only.
- **No Approach 1 for the team** (it raises) — only Approach 2.
- **No mandatory paid deploy** (the Space `push_to_hub` is documented but optional; E2B optional,
  Docker local by default).
- **No `"*"`** in `additional_authorized_imports` (never — 06 §2).
- **No re-teaching** of telemetry / eval / RAG / vision (recall + integration only — M11–M14 own it).

## Where to go next

- **Open Deep Research** (`examples/open_deep_research`) — fork and adapt it; it is Quill pushed further.
- **The Hugging Face AI Agents Course** — free and certifying.
- **Course 1 (LangGraph)** for fine-grained stateful workflows; **course 2 (Strands / Bedrock)** for managed AWS agents.
- **CrewAI / PydanticAI / OpenAI Agents SDK / Google ADK / LlamaIndex** — pick one when its shape fits your problem.
- And always: **don't reach for an agent** when a deterministic workflow would do.

Verified against **smolagents 1.26.0** (latest at build time).
