# Module 10 — Multi-Agent Systems: Quill Gets a Research Team

Quill from Module 9 is a one-person band: a single `CodeAgent` that plans, writes the pandas,
saves charts, AND goes to the web itself — every page it visits lands *whole* in its own memory.
The smolagents docs put the complaint bluntly: *"why fill the memory of the code-generating agent
with all the content of the webpages visited?"*. This module splits the work. Quill becomes a
**manager** `CodeAgent` over a specialised sub-agent, the **`web_researcher`**:

- the **manager** plans and analyses the dataset on its OWN clean memory;
- the **`web_researcher`** (a `ToolCallingAgent`) does the searching and page-visiting, then
  returns only a short summary with source URLs;
- the run still ends in a cited **`QuillReport`** (the Module 8 contract, unchanged).

This is **orchestrator-workers**: a central LLM dynamically breaks down a task and delegates to
worker LLMs. It is the one Anthropic agent pattern smolagents gives you built-in.

## Why a second agent (and when one is still enough)

The real reason for multi-agent is **context isolation**: each agent sees only what it needs, so
the manager never has to digest raw web pages. Specialization (a focused prompt + toolset per role)
is the runner-up. The counterweight is **coordination cost**: each delegation is at least one extra
LLM call (the sub-agent runs its own ReAct loop, up to `max_steps`), more latency, and a new failure
mode (the worker comes back empty-handed). A manager + sub-agent run can **triple** the LLM calls of
a single agent — pay it only when isolation or specialization earns more than it costs.

> ⚠️ **Common misconception: "More agents = better results."** False. Every agent you add multiplies
> tokens, latency, and points of failure. The reason to reach for multi-agent is context isolation,
> not raw power — smolagents itself says to *"regularize towards not using agentic behaviour"*.

## The current way to wire a team (smolagents 1.26.0)

There is **no `ManagedAgent`** — deprecated in 1.8.0, dropped from the docs in 1.21.0, and ABSENT
from `agents.py` in 1.26.0. Any tutorial that does `from smolagents import ManagedAgent` is dead.
The current mechanism: ANY agent becomes callable by a manager the moment it has a **`name`** and a
**`description`**, then you pass it via **`managed_agents=[...]`**.

```python
from smolagents import CodeAgent, ToolCallingAgent, WebSearchTool, VisitWebpageTool

web_researcher = ToolCallingAgent(
    tools=[WebSearchTool(), VisitWebpageTool()],
    model=model,                       # always via make_model() in Quill
    name="web_researcher",             # name + description make it manager-callable
    description="Searches the web and visits pages to fetch missing context...",
    max_steps=10,
)
manager = CodeAgent(tools=[...], model=model, managed_agents=[web_researcher])
```

### How the manager calls the sub-agent (the internals)

For a **`CodeAgent` manager**, smolagents exposes each managed agent inside the Python sandbox as a
callable function whose signature is auto-generated from the system-prompt template:

```python
def web_researcher(task: str, additional_args: dict[str, Any]) -> str:
    """<the sub-agent's description>
    Args:
        task: Long detailed description of the task.
        additional_args: Dictionary of extra inputs (e.g. images, dataframes).
    """
```

So Quill literally writes `summary = web_researcher("Find the SaaS average churn")` in its code,
like calling any tool. The `additional_args` channel is how rich objects (images, DataFrames) flow
DOWN to a sub-agent — the hook Module 11 reuses for vision. (A **`ToolCallingAgent` manager** would
instead dispatch via `execute_tool_call(tool_name, arguments)` and list the team in its prompt.)

### Why the worker is a `ToolCallingAgent` and the manager a `CodeAgent`

| | Manager (Quill) | Worker (`web_researcher`) |
|---|---|---|
| Agent type | **`CodeAgent`** ("problem solver") | **`ToolCallingAgent`** ("dispatcher") |
| Role | plan + analyse the data | search + visit + summarise |
| Action format | Python code | JSON tool calls (single timeline) |
| Toolset | `load_dataset`, `profile_dataframe`, `save_chart` + `managed_agents` | `WebSearchTool` + `VisitWebpageTool` |
| Why | composing pandas + arithmetic | step-through web nav, schema-validated, no Python needed |

Web navigation does not need to compose Python; the JSON tool-call is safer and sufficient. (A
`ToolCallingAgent` can fan out parallel calls via `max_tool_threads` — a real capability we do NOT
use here: the researcher is a single timeline.)

## The five Anthropic patterns → smolagents

| Anthropic pattern | smolagents |
|---|---|
| Prompt chaining | deterministic Python ("write all the code yourself") |
| Routing | an `if llm_decision(): path_a() else: path_b()` you write — **pattern, not API** |
| Parallelization (sectioning/voting) | a loop / several `agent.run` you aggregate — **pattern, not API** |
| **Orchestrator-workers** | **`managed_agents=[...]` — built-in (this module)** |
| Evaluator-optimizer | a second agent / a judge tool in a hand-written loop — **pattern, not API** |

Only orchestrator-workers is built in. Routing, parallelization, and evaluator-optimizer are
patterns you compose — don't go looking for a class that doesn't exist. (The evaluator/judge is the
heart of Module 14's eval.)

> As of the smolagents launch material, a multi-agent smolagents system topped the **GAIA**
> leaderboard, and Open Deep Research (HF's repro) reported **55.15%** on GAIA — partly thanks to
> the code-action format using ~30% fewer tokens than JSON tool-calling. Treat these as *reported*
> figures, not guarantees.

## The constraint nobody warns you about: remote sandbox + multi-agent

A remote `executor_type` (docker/e2b) **plus** `managed_agents` raises an exception:

```
Exception: Managed agents are not yet supported with remote code execution.
```

`create_python_executor` refuses the combination. The reason: in **Approach 1** (snippet-in-sandbox)
the model stays local and secrets (the HF token) are NOT shipped into the sandbox, so a sub-agent
could not authenticate its own LLM from inside. So Quill's team runs in `executor_type="local"` in
this module. Running the WHOLE team *inside* a sandbox (**Approach 2**) is the capstone, Module 15.

| | Approach 1 (snippet in sandbox) | Approach 2 (whole agent in sandbox) |
|---|---|---|
| What runs in the sandbox | each generated Python snippet | the entire agent system |
| Multi-agent | **NOT supported** (raises) | **OK** (more manual; secrets passed in) |
| Knob | one parameter (`executor_type`) | hand-built (Module 15) |

## What Module 10 adds to Quill

`quill/team.py` (**NEW**):

```python
build_web_researcher(model=None, *, max_steps=10, provide_run_summary=False) -> ToolCallingAgent
```

builds the `web_researcher` (name + description + the web tools), model always via `make_model`.

`quill/agent.py` (**MODIFIED**): `build_quill` is **extended by addition** with
`managed_agents: list | None = None`:

- omitted → Quill's **default team** (just the `web_researcher`);
- `[]` → a **solo manager** (no team, pre-M10 shape minus the web tools);
- a list → your own sub-agents.

The web tools **left** the manager's `tools=` list (they live in the sub-agent now — context
isolation). The manager stays `executor_type="local"` (T10.7). `make_model`, `QuillReport`,
`save_chart` and the import lock are untouched.

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-10/.env.example module-10/.env   # then put your HF token in it
```

Run from inside `module-10/` so `data/`, `outputs/` and the `quill` package resolve.

```bash
uv run python -m quill "Is our Q3 churn (in data/customers.csv) high vs the SaaS industry average?" --data data/customers.csv
```

The trajectory shows the manager delegating to `web_researcher` (search + page visits), then a
Markdown report with `[n]` citations mapped onto the `QuillReport`'s `sources` (at least one web
source). The step-by-step replay (`agent.replay()`) is at `python -m quill.agent`.

## Test it

```bash
uv run pytest module-10/tests/                    # offline (no token, no network, no LLM)
QUILL_LIVE_TESTS=1 uv run pytest module-10/tests/ # also the real team run (needs HF_TOKEN)
```

The offline tests build BOTH the manager (a fake-model `CodeAgent`) AND the `web_researcher`
sub-agent (a fake tool-call model), so the whole team loop runs deterministically with zero LLM
calls. They prove:

- `build_web_researcher` returns a `ToolCallingAgent` named EXACTLY `web_researcher`, with the web
  tools and `max_steps=10`; **`ManagedAgent` does not exist** in 1.26.0;
- `build_quill(managed_agents=...)` is extended by addition: default team / `[]` solo / custom; the
  manager registers the sub-agent in `agent.managed_agents` (a name-keyed dict);
- the web tools are NOT on the manager (context isolation) — they are one level down on the sub-agent;
- an end-to-end offline run delegates to the (fake-model) sub-agent and finishes in a **cited
  `QuillReport`** (a `[1]` marker resolving to a `Source`);
- the constraint: a remote executor + `managed_agents` raises the documented exception, so Quill
  stays `local`.

Every Module 2–9 test still passes here (the cumulative suite). Two carried-forward assertions were
*updated* (not deleted) for the M10 reality: the manager's toolbox no longer holds
`web_search`/`visit_webpage`, and the M8 web-source check is now exercised through a
`web_researcher(...)` delegation.

## What this module deliberately does NOT do

- **No `vision_browser` sub-agent / `run(images=...)`** — Module 11.
- **No Approach 2** (multi-agent system INSIDE a sandbox) — stays `local`; Module 15.
- **No telemetry** / nested-span traces of the manager→sub-agent call — Module 14.
- **No evaluation / LLM-as-judge** — the evaluator-optimizer stays a *described* pattern; Module 14.
- **No real parallelization** (`max_tool_threads`, fan-out) — mentioned only.
- **No hierarchy deeper than 1** (sub-agents of sub-agents) — mentioned only.

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
