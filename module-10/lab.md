# Lab 10 — Quill gets a research team: a manager over a `web_researcher` sub-agent

**Goal:** turn Quill from a one-person band into a **manager `CodeAgent`** over a specialised
**`web_researcher` `ToolCallingAgent`**. The manager plans and analyses the dataset; the sub-agent
searches the web, visits pages, and returns a short summary with source URLs. The web tools LEAVE
the manager and move into the sub-agent — that is **context isolation**. The run still ends in a
cited `QuillReport`.

**You'll see:** the current multi-agent wiring (`name` + `description` + `managed_agents=[...]`, and
that **`ManagedAgent` no longer exists** in 1.26.0); the web tools gone from the manager and present
on the sub-agent; the manager calling `web_researcher("...")` from its generated code; and (live)
Quill answering a churn-vs-industry question by delegating to the researcher and returning a
validated, cited `QuillReport`. You'll also meet the hard constraint: a remote executor +
`managed_agents` raises — so Quill stays `local`.

**Observable result:**

```bash
uv run python -m quill "Is our Q3 churn (in data/customers.csv) high vs the SaaS industry average?" --data data/customers.csv
```

```text
[Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct
...
 ─ Executing: summary = web_researcher("What is the SaaS industry average annual churn rate?")
Here is the final answer from your managed agent 'web_researcher':
SaaS median annual churn is ~5% per year [https://...].
...                        # ← the manager writes pandas on data/customers.csv, draws a chart, save_chart
===== REPORT =====
# Is our Q3 churn high vs the SaaS industry average?

## Findings
- Our churn is above the ~5% SaaS median [1].

## Charts
- `outputs/churn_vs_industry.png`

## Sources
[1] [SaaS Churn Benchmarks](https://...)

[Quill] Run cost — input tokens: ... | output tokens: ... | total: ...
```

> The exact trajectory varies — LLMs are non-deterministic. What is guaranteed: the manager
> *delegates to `web_researcher`* and returns a *validated, cited `QuillReport`*.

Verified against **smolagents 1.26.0**.

---

## Step 1 — Setup

Start from the cumulative Module 9 state (this `module-10/` already carries it forward). **No new
extra** is needed — `[toolkit]` already provides `WebSearchTool`/`VisitWebpageTool` (`ddgs` +
`markdownify`):

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
```

`HF_TOKEN` is never committed — copy `.env.example` to `.env` and put your token there. M10 seeds
a new dataset, `data/customers.csv` (with a `churned` column), alongside the fil-rouge
`data/sales.csv` (unchanged).

---

## Step 2 — Build the `web_researcher` sub-agent (`quill/team.py`)

`build_web_researcher` returns a **`ToolCallingAgent`** (a "dispatcher": JSON tool calls, single
timeline) with the web tools, the canonical **`name="web_researcher"`** and a focused
`description`. The two attributes — `name` + `description` — are what make it **callable by a
manager**. The model ALWAYS comes from `make_model` (the M4 frozen contract):

```python
def build_web_researcher(model=None, *, max_steps=10, provide_run_summary=False):
    return ToolCallingAgent(
        tools=[WebSearchTool(), VisitWebpageTool()],   # name="web_search" — exactly ONE web-search tool
        model=model or make_model(role="researcher"),  # never HfApiModel; always make_model
        name="web_researcher",                          # canonical name (not web_agent/researcher)
        description="Searches the web and visits pages to fetch missing context...",
        max_steps=10,                                   # the per-delegation cost ceiling
        provide_run_summary=provide_run_summary,
    )
```

Why a `ToolCallingAgent` and not a `CodeAgent`? Web navigation is a single timeline of validated
tool calls — it does not need to *compose Python*. The manager, which plans + does arithmetic, is the
`CodeAgent`. (Decision matrix from Module 3.)

---

## Step 3 — Make Quill the manager (`quill/agent.py`)

`build_quill` is **extended by addition** with `managed_agents`. By default it builds the
`web_researcher` and registers it; the web tools are **removed** from the manager's own `tools=`:

```python
# The manager's toolbox: DATA tools only. The web tools moved to the sub-agent (context isolation).
tools = [load_dataset, profile_dataframe, save_chart()]      # NO WebSearchTool / VisitWebpageTool here

if managed_agents is _DEFAULT_TEAM:
    resolved_managed_agents = [build_web_researcher(model=model)]   # Quill's default team
else:
    resolved_managed_agents = managed_agents                       # [] = solo; or your own list

agent = CodeAgent(
    tools=tools,
    model=model or make_model(role="analyst"),
    executor_type=executor_type,                 # MUST stay "local" with managed_agents — see Step 6
    managed_agents=resolved_managed_agents,       # the team — NEVER ManagedAgent
    final_answer_checks=resolved_checks,          # still the M8 cited-report contract
    max_steps=8,
)
```

The manager reaches the team by **writing code**: smolagents exposes each managed agent inside the
sandbox as a function, so Quill literally writes `summary = web_researcher("...")`. The
`additional_args` channel on that function is how rich objects (images, DataFrames) flow down — the
hook Module 11 reuses.

> **`ManagedAgent` is gone** (deprecated 1.8.0, dropped from docs 1.21.0, absent from `agents.py` in
> 1.26.0). A sub-agent is "managed" purely by `name` + `description` + `managed_agents=[...]`. Any
> tutorial that does `from smolagents import ManagedAgent` is dead.

---

## Step 4 — (Optional) let the manager see the sub-agent's reasoning

`provide_run_summary=True` surfaces the sub-agent's full reasoning trace to the manager, not only its
final answer:

```python
researcher = build_web_researcher(provide_run_summary=True)
agent = build_quill(managed_agents=[researcher])
```

Off (the default), the manager gets only the summary string — cleaner memory, less context. On, the
manager can audit *how* the researcher reached its answer. Flip it and compare (the "Try it yourself"
below).

---

## Step 5 — Run it on a real question + read the trajectory

```bash
uv run python -m quill "Is our Q3 churn (in data/customers.csv) high vs the SaaS industry average?" --data data/customers.csv
```

Watch the trajectory: the manager delegates to `web_researcher`, gets a summary back as the
observation, then writes pandas on `data/customers.csv`, draws a chart, calls `save_chart`, and
returns a cited `QuillReport`. For the full step-by-step replay use the trajectory printer:

```bash
uv run python -m quill.agent data/customers.csv "Is our churn high vs the SaaS average?"
```

Read the trajectory the supported way — `agent.replay()` / `agent.memory.steps` (never the removed
`agent.logs`).

---

## Step 6 — The constraint: remote executor + managed agents

A remote `executor_type` (docker/e2b) **plus** `managed_agents` raises:

```
Exception: Managed agents are not yet supported with remote code execution.
```

`create_python_executor` refuses the combination *at construction* (before any container starts).
The reason: in **Approach 1** (snippet-in-sandbox) the model stays local and the HF token is NOT
shipped into the sandbox, so a sub-agent could not authenticate its own LLM from inside. So this lab
keeps the manager `executor_type="local"`:

```python
# QUILL_EXECUTOR=docker uv run python -m quill ...   # -> raises (Approach 1 can't do multi-agent)
```

Running the WHOLE team *inside* a sandbox — **Approach 2** — is the capstone, Module 15. A SOLO
manager (`managed_agents=[]`) is fine under a remote executor; it is the *combination* that fails.

---

## Step 7 — Tests

```bash
uv run pytest module-10/tests/                    # offline: no token, no network, no LLM
QUILL_LIVE_TESTS=1 uv run pytest module-10/tests/ # also the real team run (needs HF_TOKEN)
```

**Live budget:** a manager + sub-agent run makes **5–15 LLM calls** (the manager's loop PLUS the
researcher's own loop, capped at `max_steps=10`). The new M10 live test does one such run; every
live test skips unless `QUILL_LIVE_TESTS=1` and `HF_TOKEN` is set.

The offline tests build BOTH the manager (a fake-model `CodeAgent`) AND the `web_researcher`
sub-agent (a fake tool-call model that emits `final_answer`), so the whole team loop runs with **zero
LLM calls**. They prove the sub-agent is a `ToolCallingAgent` named exactly `web_researcher`, that
`ManagedAgent` does not exist, that `build_quill` registers the team in `agent.managed_agents`, that
the web tools left the manager, that an end-to-end run finishes in a **cited `QuillReport`**, and
that a remote executor + `managed_agents` raises.

---

## Try it yourself (not graded)

1. **See the reasoning.** Set `provide_run_summary=True` on the `web_researcher` and re-run — observe
   what the manager now sees (the sub-agent's steps, not just its final summary). Decide whether the
   extra context is worth the memory it costs.
2. **Force the empty-handed failure.** Lower the sub-agent's `max_steps` to `3`
   (`build_web_researcher(max_steps=3)`) and ask a question that needs deeper digging. Watch the
   researcher run out of steps and how the manager handles a thin / empty summary.

---

## What this lab does NOT do

No `vision_browser` sub-agent / `run(images=...)` (M11). No Approach 2 — the team stays `local`
(M15). No telemetry / nested-span traces (M14). No evaluation / LLM-as-judge — the evaluator-optimizer
stays a *described* pattern (M14). No real parallelization (`max_tool_threads`, fan-out) — mentioned
only. No hierarchy deeper than one level. No hard-coded key (`os.environ["HF_TOKEN"]` + `.env`).
