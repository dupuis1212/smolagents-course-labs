# Module 7 — Planning and Building Good Agents

Quill from Module 6 *works*, but it **wings it**: it attacks every question step by step, re-runs
`load_dataset` and `profile_dataframe` on a file it already loaded, chases a column that does not
exist, corrects itself, and repeats. Ten steps for what needed five — and every step is one LLM
call, i.e. latency and money. This module makes Quill **reliable and frugal**, not by giving it
*more* power, but by **planning its work, sharpening its prompt and tools**, and knowing when an
agent is the wrong tool entirely.

The thesis is contrarian: **less agency is usually better**. Adding tools, freedom, or a
sub-agent multiplies LLM calls, latency, and failure modes. The engineering skill is to
*regularize towards not using agentic behaviour* — hard-code what is deterministic, and leave the
model only what needs judgement.

This module **extends `build_quill()` by addition** (no signature breaks) with two keyword-only
arguments, and sharpens the data tools' docstrings:

- **`planning_interval: int | None = None`** — when set (Quill ships `DEFAULT_PLANNING_INTERVAL =
  3`), the agent inserts a **`PlanningStep`** at step 1 and then every `planning_interval` steps
  (trigger `step_number == 1 or (step_number - 1) % planning_interval == 0`, as of smolagents
  1.26.0). A planning step is a tool-free LLM call where Quill (re)states its facts and plan; it
  re-centres the agent on the goal and cuts redundant exploration. It is itself **one extra LLM
  call**, so it is a trade-off: worth it on long jobs, wasteful on short ones.
- **`instructions: str | None`** — text **APPENDED to the system prompt** (smolagents' own words:
  *"Custom instructions for the agent, will be inserted in the system prompt"*). Omit it for
  Quill's default `QUILL_INSTRUCTIONS` (a data-analyst brief whose headline rule is *profile
  once*); pass `""` for the bare prompt; pass your own string to specialise. It does **not**
  replace the system prompt — we never touch `prompt_templates["system_prompt"]`, which the docs
  call *"generally not advised"* (you would lose the Jinja2 injection of the tool list, authorized
  imports, and code-block tags).

It also adds **`quill/bench.py`**: run Quill twice on one question (baseline vs improved) and
print the **drop in `ActionStep`s** — the module's observable result.

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
  "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-07/.env.example module-07/.env   # then put your HF token in it
```

**Measure the step drop** (makes real model calls, so set `HF_TOKEN`). Run from inside
`module-07/` so `data/sales.csv`, `outputs/`, and the `quill` package resolve:

```bash
uv run python -m quill.bench \
  --dataset data/sales.csv \
  --question "Which category grew fastest from Q1 to Q4 2025, and is that growth statistically meaningful?"
```

prints (your numbers will vary — LLMs are non-deterministic):

```text
===== STEP COMPARISON =====
Baseline  (no planning, bare instructions)        : 11 ActionSteps, ~11 LLM calls
Improved  (planning_interval=3, sharpened)        :  6 ActionSteps, ~8 LLM calls (2 of them planning)
Step reduction: ~45%   (your numbers will vary — LLMs are non-deterministic)
```

> **Count `ActionStep`s, not "all steps".** A `PlanningStep` is itself one LLM call, so the
> improved run's *total* step count includes the planning calls. The bench reports `ActionStep`s
> (the work) and tracks `PlanningStep`s separately — measuring honestly.

## Test it

```bash
uv run pytest module-07/tests/                    # offline (no token, no Docker)
QUILL_LIVE_TESTS=1 uv run pytest module-07/tests/ # also the real planning + bench runs (needs HF_TOKEN)
```

The offline tests run with **no network and zero tokens**: the planning tests use the shared
`FakeModel` (repo-root `conftest.py`) to script *both* the plan and the action, so a real
`PlanningStep` lands in `agent.memory.steps` for free. They prove:

- `build_quill(planning_interval=1)` inserts a `PlanningStep` at step 1 (visible in
  `agent.memory.steps`); the default (no `planning_interval`) inserts **none**; the trigger
  `step_number == 1 or (step_number - 1) % interval == 0` fires at steps 1, 4, 7… for interval 3;
- `build_quill(instructions="...")` **appends** the text to `agent.system_prompt` (and the
  injected tool list is still there — proof it was appended, not replaced); omitting
  `instructions` uses `QUILL_INSTRUCTIONS`; `instructions=""` gives the bare prompt; we never edit
  the raw `prompt_templates["system_prompt"]` (its Jinja2 placeholders survive);
- the data tools' docstrings are non-empty and sharpened while their **frozen M3 signatures**
  (name, inputs, output_type) are unchanged;
- `quill.bench`'s `count_steps` / `run_and_count` count `ActionStep`s and `PlanningStep`s
  separately, and `format_report` renders the reduction line with the variance caveat.

Every Module 2/3/4/5/6 test (the toolbox, the agent loop, `make_model()`, the `Monitor` cost
accessor, the sandbox policy, the callbacks, and multi-turn) still passes here.

## The planning trigger (`planning_interval=3`)

| `step_number` | `(step - 1) % 3` | planning? |
|---|---|---|
| 1 | 0 | ✅ yes (step 1 always) |
| 2 | 1 | no |
| 3 | 2 | no |
| 4 | 0 | ✅ yes |
| 7 | 0 | ✅ yes |

The model generates the plan up to the `<end_plan>` stop sequence (which bounds the plan length
and separates the plan from the action).

## `instructions=` vs editing the system prompt

| | what it does | risk | survives `save`/`from_hub`? | when to use | smolagents verdict |
|---|---|---|---|---|---|
| **`instructions=`** | **appends** to the system prompt | low — injections preserved | yes (it is an agent attribute) | specialising an agent | **the recommended default** |
| Editing `prompt_templates["system_prompt"]` | **replaces** the prompt | high — drops the injected tool list / authorized imports / code-block tags; breaks on lib update | only your edited string | last resort, knowing the cost | **"generally not advised"** |

## The six "building good agents" principles

| # | Principle | Why | Quill application |
|---|---|---|---|
| 1 | **Reduce the number of LLM calls** (the master rule) | each step is one LLM call | `profile_dataframe` returns schema + dtypes + stats + missing in ONE call, not column-by-column |
| 2 | Improve the information flow to the LLM | failures must reach the next `Observation:` | `load_dataset` prints a readable summary, not a raw dump |
| 3 | Write better tools | the docstring IS the interface the model reads | sharpened `load_dataset`/`profile_dataframe`/`save_chart` docstrings: formats, examples, return contract |
| 4 | Pass extra objects with `additional_args` | hand the agent images/DataFrames/URLs directly | a pre-loaded DataFrame can be passed in rather than reloaded |
| 5 | Debugging | a stronger model; more instructions | `instructions=` at init (appended), task-specific detail in the task string, tool-specific detail in its `description` |
| 6 | Extra planning | re-centre on the goal periodically | `planning_interval=3` — planning IS one of the six principles |

## `code_block_tags` (volatile — as of smolagents 1.26.0)

| value | code-block delimiters |
|---|---|
| default (`<code>` / `</code>`) | `<code>` … `</code>` |
| `'markdown'` (opt-in) | ` ```python ` … ` ``` ` |

⚠️ Many tutorials still show the markdown ` ```py ` fences. That is an **opt-in**, not the default
as of smolagents 1.26.0 — Quill uses the `<code>` default.

## What this module deliberately does NOT do

- **No structured output / `final_answer_checks` / `response_format` /
  `use_structured_outputs_internally`** (Module 8). We count steps with `return_full_result`; we
  do **not** validate the answer's content. `structured_code_agent.yaml` exists but is Module 8.
- **No multi-agents / `managed_agents` / sub-agent** (Module 10).
- **No editing of `prompt_templates["system_prompt"]`** — shown only as a commented anti-pattern.
- **No new extra, no Docker/E2B required, no paid service, no GPU.** Quill runs `local` for the
  bench (`QUILL_EXECUTOR` stays configurable).
- **Never `agent.logs`.** Always `agent.memory.steps` / `agent.replay()`.

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
