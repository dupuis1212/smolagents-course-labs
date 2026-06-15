# Module 6 — Memory, State, and Inspecting Runs

Quill from Module 5 is sandboxed but **amnesiac**: every `run()` starts from scratch
(`reset=True` is the default), and every step drags the full `print(df.head(50))` dump from two
steps ago back into the prompt — you pay for the same DataFrame on every later LLM call. This
module fixes both.

A smolagents agent's memory is **not a separate state object** — it is just a list of typed
**steps** (`agent.memory.steps`). Every step the agent turns that list back into chat messages
with `write_memory_to_messages()` and sends them to the model. So memory **is** the state, it is
a mutable Python list, and you can read it, continue it, and prune it yourself.

This module adds **`quill/callbacks.py`**: two **step callbacks** with the frozen smolagents
signature `(memory_step, agent)`, wired into `build_quill()` via `step_callbacks=`:

- **`prune_old_observations`** — after each step, null out the big DataFrame dumps in stale
  `ActionStep`s (replace them with a small marker), so they stop being re-sent to the model.
- **`log_step_cost`** — print one `step N: <in>+<out> tokens` line per step from
  `ActionStep.token_usage` (the minimal cost-observability brick before any telemetry).

And it makes Quill **multi-turn** with `run(..., reset=False)`, so a follow-up question keeps the
loaded DataFrame and prior findings instead of reloading everything.

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
  "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-06/.env.example module-06/.env   # then put your HF token in it
```

**Multi-turn: two questions on one agent, the second `reset=False`** (makes real model calls, so
set `HF_TOKEN`). Run from inside `module-06/` so `data/sales.csv`, `outputs/`, and the `quill`
package resolve:

```bash
uv run python -m quill.agent --multi-turn data/sales.csv \
  "Which category grew fastest from Q1 to Q4 2025?" \
  "Now answer the same question but exclude any rows from 2020."
```

prints (truncated):

```text
===== TURN 1 (reset=True, the default) =====
[Quill] step 1: 2143+98 tokens (total 2241)
[Quill] step 2: 2562+74 tokens (total 2636)
...
Turn 1 answer: Team grew fastest ...

===== TURN 2 (reset=False — keep turn 1 in memory) =====
[Quill] step 1: 3110+61 tokens (total 3171)   # the loaded df is still there — no reload
...
Turn 2 answer: Excluding 2020, Team still ...

===== MEMORY GREW (reset=False did not wipe turn 1) =====
steps after turn 1     : 3
steps after turn 2     : 5  (turn 1's steps are still here)

===== FULL TRAJECTORY (agent.replay) =====
... # the old big observation now shows: [pruned: large DataFrame dump removed to save tokens]
```

`reset=False` keeps the conversation **in RAM inside the agent object** — it persists nothing to
disk (that is Module 13). Kill the process and the memory is gone.

## Test it

```bash
uv run pytest module-06/tests/                    # offline (no token, no Docker)
QUILL_LIVE_TESTS=1 uv run pytest module-06/tests/ # also the real multi-turn LLM run (needs HF_TOKEN)
```

The offline tests run with no network and **spend zero tokens** — that is the whole point of the
module: the callback tests build `ActionStep`s by hand and call `prune_old_observations` /
`log_step_cost` directly, so we verify pruning + cost logging without an LLM. They prove:

- `prune_old_observations((memory_step, agent))` prunes a big observation on a step older than
  `KEEP_LAST`, leaves recent/small ones alone, ignores non-`ActionStep` steps, and is idempotent;
- `log_step_cost` does **not** raise when `token_usage is None` and prints `<in>+<out> tokens`
  when it is set;
- `build_quill()` wires `step_callbacks`; the callbacks **fire** during a real (fake-model) run
  and prune the old dump in `agent.memory.steps`;
- `reset=False` **continues** memory (a second run leaves more steps than a single run), while
  the default `reset=True` wipes it.

Every Module 2/3/4/5 test (the toolbox, the agent loop, `make_model()`, the `Monitor` cost
accessor, the sandbox policy) still passes here.

## Reading and inspecting a run (never `agent.logs`)

| You want | Use | Notes |
|---|---|---|
| The list of step objects | `agent.memory.steps` | `SystemPromptStep`, `TaskStep`, `ActionStep`, `PlanningStep` |
| A pretty replay of the trajectory | `agent.replay()` | `detailed=True` for the full dump (debugging only) |
| The agent's structure as a tree | `agent.visualize()` | rich tree of tools / sub-agents |
| Steps as dicts (succinct / full) | `agent.memory.get_succinct_steps()` / `get_full_steps()` | succinct drops `model_input_messages` |
| Every line of Python the agent ran | `agent.memory.return_full_code()` | concatenated `code_action`s |
| Per-step token cost | `step.token_usage` on each `ActionStep` | aggregate via `agent.monitor.get_total_token_counts()` |

> **`agent.logs` is dead** — it was removed in smolagents 1.21.0. Use `agent.memory.steps` /
> `agent.replay()`. Most tutorials still show `.logs`; do not copy them.

## `reset=True` vs `reset=False`

| | memory before run | what the model sees | typical use | gotcha |
|---|---|---|---|---|
| `reset=True` (default) | wiped | only this task | one-shot question | "it forgot everything" surprises chat builders |
| `reset=False` | kept | prior steps + this task | a follow-up turn / chat | nothing is saved to disk — RAM only (Module 13 persists) |

## What this module deliberately does NOT do

- **No `planning_interval` / prompt edits** (Module 7). We only *name* the `PlanningStep` type;
  the callbacks ignore it.
- **No `stream_to_gradio` / `GradioUI`** (Module 13). `GradioUI` uses `reset=False` under the
  hood, but the UI itself is Module 13.
- **No OpenTelemetry / telemetry backends** (Module 14). `log_step_cost` is the minimal brick
  before that.
- **No disk/Hub persistence of memory** (Module 13). `reset=False` is RAM only.
- **No `AgentError` hierarchy / auto-correction mechanics** (Module 8). `error` is only ever a
  field we read.
- **No image input / `images=` / `additional_args`** (Module 11). We only prune the
  `observations_images` field in passing — Module 11 reuses this exact callback for screenshots.
- **Never `agent.logs`.** Always `agent.memory.steps` / `agent.replay()`.

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
