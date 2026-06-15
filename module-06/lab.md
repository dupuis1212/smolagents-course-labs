# Lab 6 — Give Quill a memory: read it, continue it (`reset=False`), prune it in a callback

**Goal:** make Quill **multi-turn** with `run(..., reset=False)` and add `quill/callbacks.py` —
two **step callbacks** with the frozen smolagents signature `(memory_step, agent)` that
**prune** the big DataFrame dumps from stale observations (token saving) and **log** per-step
token cost — then inspect the whole trajectory with `agent.replay()`. Never `agent.logs`.

**You'll see:** a second question on the same agent reuse the already-loaded DataFrame (no
reload, no re-profile) because memory carried turn 1 forward; one `step N: <in>+<out> tokens`
line per step from the cost callback; and a `replay()` where the old big observation now reads
`[pruned: large DataFrame dump removed to save tokens]`.

**Observable result:**

```bash
uv run python -m quill.agent --multi-turn data/sales.csv \
  "Which category grew fastest from Q1 to Q4 2025?" \
  "Now answer the same question but exclude any rows from 2020."
```

```text
===== TURN 1 (reset=True, the default) =====
[Quill] step 1: 2143+98 tokens (total 2241)
...
Turn 1 answer: Team grew fastest ...

===== TURN 2 (reset=False — keep turn 1 in memory) =====
[Quill] step 1: 3110+61 tokens (total 3171)   # turn 1 is still in memory: no df reload
...
Turn 2 answer: Excluding 2020, Team still ...

===== MEMORY GREW (reset=False did not wipe turn 1) =====
steps after turn 1     : 3
steps after turn 2     : 5  (turn 1's steps are still here)

===== FULL TRAJECTORY (agent.replay) =====
... # an old ActionStep observation: [pruned: large DataFrame dump removed to save tokens]
```

## Steps

1. **Setup** — copy the Module 5 state (the cumulative rule: M6 code must still pass the M1–M6
   smoke tests), then sync the pins. **No new extra** — `step_callbacks` and `reset=False` are
   core `smolagents`, already installed since M3/M5.
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
     "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
   cp module-06/.env.example module-06/.env   # then add your HF token; never commit .env
   ```
   `data/sales.csv` is inherited from Module 2 unchanged.

2. **Read the memory first.** Memory is just a list of typed steps. Do a simple run, then walk
   `agent.memory.steps`, filter the `ActionStep`s, and print what each one holds — and call
   `agent.replay()`. (Use `agent.memory.steps` / `agent.replay()`, **never** the removed
   `agent.logs`.)
   ```python
   from smolagents import ActionStep
   from quill.agent import build_quill, build_task

   with build_quill() as agent:
       agent.run(build_task("data/sales.csv", "Which category has the highest net_rev?"))
       for step in agent.memory.steps:
           if isinstance(step, ActionStep):
               usage = step.token_usage
               obs_len = len(step.observations or "")
               print(f"step {step.step_number}: tokens={usage} | observations={obs_len} chars")
       agent.replay()                       # pretty replay of the trajectory
       print(agent.memory.return_full_code())  # every line of Python Quill ran, concatenated
   ```
   `get_succinct_steps()` (drops `model_input_messages`) and `get_full_steps()` (keeps them, as
   dicts) are the dict views; `return_full_code()` is gold for a `CodeAgent`.

3. **Write `quill/callbacks.py`** — two functions with the frozen `(memory_step, agent)`
   signature. The callback receives the `agent`, so it can read **and mutate**
   `agent.memory.steps` — that is the whole lever (`write_memory_to_messages()` re-reads that
   list on the next step, so editing it changes what the model sees):
   ```python
   from smolagents import ActionStep

   KEEP_LAST = 2            # keep the last 2 steps verbatim
   MAX_OBS_CHARS = 1000     # only prune observations bigger than this
   PRUNE_MARKER = "[pruned: large DataFrame dump removed to save tokens]"

   def prune_old_observations(memory_step: ActionStep, agent) -> None:
       if not isinstance(memory_step, ActionStep):
           return
       current = memory_step.step_number
       for step in agent.memory.steps:          # the just-finished step is NOT in here yet
           if not isinstance(step, ActionStep) or step is memory_step:
               continue
           if current - step.step_number < KEEP_LAST:
               continue                          # too recent — the model still needs it
           obs = step.observations
           if obs is not None and len(obs) > MAX_OBS_CHARS and obs != PRUNE_MARKER:
               step.observations = PRUNE_MARKER  # mutate memory in place

   def log_step_cost(memory_step: ActionStep, agent) -> None:
       if not isinstance(memory_step, ActionStep):
           return
       usage = memory_step.token_usage
       if usage is None:                          # legit None offline / pre-model error
           return                                  # no silent try/except — test `is not None`
       print(f"[Quill] step {memory_step.step_number}: "
             f"{usage.input_tokens}+{usage.output_tokens} tokens (total {usage.total_tokens})")

   def quill_callbacks():
       return [prune_old_observations, log_step_cost]  # prune BEFORE logging
   ```
   This is the canonical smolagents pruning pattern — the vision browser's `save_screenshot`
   callback prunes `observations_images` the exact same way (Module 11 reuses this file).

4. **Wire the callbacks** in `build_quill()` (`quill/agent.py`). One line — do **not** touch
   `make_model()`, the tool signatures, `QuillReport`, or the sandbox policy:
   ```python
   from .callbacks import quill_callbacks

   def build_quill(model=None):
       executor_type, authorized_imports = resolve_executor()
       return CodeAgent(
           tools=[load_dataset, profile_dataframe, save_chart(),
                  WebSearchTool(), VisitWebpageTool()],
           model=model or make_model(role="analyst"),
           executor_type=executor_type,
           additional_authorized_imports=authorized_imports,
           step_callbacks=quill_callbacks(),   # NEW (M6): prune + log, after every step
           max_steps=8,
       )
   ```
   `step_callbacks` takes a **list** (run on every step type) or a **dict by step type**
   (`{ActionStep: [...]}`, run only for that type). We pass the list; the callbacks themselves
   short-circuit on non-`ActionStep` steps. Internally they fire from `_finalize_step()` via a
   `CallbackRegistry`, which calls a two-arg callback as `cb(memory_step, agent=agent)`.

5. **Go multi-turn.** One agent, two questions, the second `reset=False`:
   ```python
   with build_quill() as agent:
       agent.run(build_task("data/sales.csv", "Which category grew fastest?"))
       # reset=False keeps the loaded DataFrame and prior findings in memory:
       answer = agent.run(
           build_task("data/sales.csv", "Now exclude any rows from 2020."),
           reset=False,
       )
   ```
   `reset=True` (the default) wipes memory before each run; `reset=False` continues the
   conversation. This is exactly what `GradioUI` does under the hood (Module 13). It saves
   **nothing to disk** — memory lives in RAM in the agent object; kill the process and it's gone.
   The repo ships this as `run_multi_turn(...)` and the `--multi-turn` CLI flag.

6. **Inspect the effect.** Re-walk `agent.memory.steps` and confirm the old big observations are
   now `PRUNE_MARKER`; compare the total context size before vs after:
   ```python
   total = sum(len(s.observations or "") for s in agent.memory.steps
               if isinstance(s, ActionStep))
   print(f"total observation chars now in memory: {total}")  # far smaller than the raw dumps
   ```

7. **Tests** (`tests/smoke_test.py`). The callback tests are **offline and spend zero tokens**:
   build an `ActionStep` by hand with a big `observations`, call `prune_old_observations`, and
   assert it's pruned past `KEEP_LAST`; assert `log_step_cost` does not blow up when
   `token_usage is None`; then a fake-model run proves the callbacks fire in the real loop, and a
   `reset=False` run proves memory continues. A `live`-marked multi-turn test runs the real thing
   (budget: ~2 LLM runs; skipped without `HF_TOKEN`).
   ```bash
   uv run pytest module-06/tests/                    # offline
   QUILL_LIVE_TESTS=1 uv run pytest module-06/tests/ # also the real multi-turn run
   ```

## Try it yourself (not graded)

1. **Use the dict-by-type form.** Pass `step_callbacks={ActionStep: [prune_old_observations,
   log_step_cost]}` and confirm nothing fires on a `PlanningStep` (you'll see planning steps once
   you enable `planning_interval` in Module 7).
2. **Summarize instead of delete.** Add a third callback that *replaces* an old big observation
   with a one-line heuristic summary (its first + last line) instead of the marker, and compare
   how much context you keep vs. how much you save.

Verified against **smolagents 1.26.0**. Never use `agent.logs` (removed in 1.21.0) — always
`agent.memory.steps` / `agent.replay()`.
