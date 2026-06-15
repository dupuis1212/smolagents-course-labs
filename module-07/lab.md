# Lab 7 — Make Quill reliable and frugal: `planning_interval`, sharper `instructions`, fewer steps

**Goal:** make Quill *reliable and economical* on a multi-step analysis by turning on
`planning_interval=3`, adding business-specific `instructions=` (appended to the system prompt),
and sharpening its tool docstrings — then **measure the drop in `ActionStep`s** (baseline vs
improved). Same task, fewer LLM calls.

**You'll see:** a real `PlanningStep` appear in `agent.memory.steps` at step 1 (then step 4); the
default Quill `instructions` show up inside `agent.system_prompt` *alongside* the still-injected
tool list (proof they were **appended**, not pasted over it); and a bench that prints a concrete
step reduction.

**Observable result:**

```bash
uv run python -m quill.bench \
  --dataset data/sales.csv \
  --question "Which category grew fastest from Q1 to Q4 2025, and is that growth statistically meaningful?"
```

```text
===== STEP COMPARISON =====
Baseline  (no planning, bare instructions)        : 11 ActionSteps, ~11 LLM calls
Improved  (planning_interval=3, sharpened)        :  6 ActionSteps, ~8 LLM calls (2 of them planning)
Step reduction: ~45%   (your numbers will vary — LLMs are non-deterministic)
```

> The numbers vary every run — LLMs are non-deterministic. The reduction is a **trend**, not a
> guarantee. The blog's "~30% fewer steps" figure is *reported* (measured against JSON tool
> calling), so never attribute it to planning alone.

## Steps

1. **Setup** — copy the Module 6 state (the cumulative rule: M7 code must still pass the M1–M6
   smoke tests), then sync the pins. **No new extra** — `planning_interval` and `instructions=`
   are core `smolagents`, already installed since M3/M5.
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
     "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
   cp module-07/.env.example module-07/.env   # then add your HF token; never commit .env
   ```
   `data/sales.csv` is inherited from Module 2 unchanged. Quill runs `local` for the bench;
   `QUILL_EXECUTOR` stays configurable (`local`/`docker`/`e2b`).

2. **Measure the baseline.** Run the *bare* Quill (no planning, `instructions=""`) on a multi-step
   question, and count the `ActionStep`s with `agent.memory.steps` (M6 — never `agent.logs`):
   ```python
   from smolagents import ActionStep
   from quill.agent import build_quill, build_task

   task = build_task("data/sales.csv",
                     "Which category grew fastest from Q1 to Q4 2025, "
                     "and is that growth statistically meaningful?")
   with build_quill(instructions="") as agent:          # bare system prompt, no planning
       agent.run(task)
       baseline = len([s for s in agent.memory.steps if isinstance(s, ActionStep)])
   print("baseline ActionSteps:", baseline)
   ```
   Watch the baseline trajectory with `agent.replay()` (M6): you'll often see it re-`load_dataset`,
   re-`profile_dataframe`, or chase a column that doesn't exist. That redundancy is the cost.

3. **Turn on planning.** Pass `planning_interval=3` to `build_quill()` (the signature is
   **extended by addition** — every prior call site still works), re-run the same task, and find
   the `PlanningStep`s:
   ```python
   from smolagents import PlanningStep

   with build_quill(planning_interval=3) as agent:
       agent.run(task)
       agent.replay()                                   # see the plan at step 1, then step 4
       plans = [s for s in agent.memory.steps if isinstance(s, PlanningStep)]
       print("planning steps:", len(plans))
   ```
   The trigger is `step_number == 1 or (step_number - 1) % planning_interval == 0` (as of
   smolagents 1.26.0): for interval 3 that fires at steps **1, 4, 7…**. The model writes the plan
   up to the `<end_plan>` stop sequence. Remember: each `PlanningStep` is **one extra LLM call** —
   it is a trade-off, good on long jobs, wasteful on short ones (the persona's rule of thumb: off
   for 3–4 steps, `planning_interval=3` from ~6 steps up).

4. **Sharpen the `instructions`.** Add a data-analyst brief. It is **APPENDED to the system
   prompt** — it does *not* replace it:
   ```python
   QUILL_INSTRUCTIONS = (
       "You are Quill, a meticulous data analyst. Work in this order:\n"
       "1. Profile the dataset ONCE with profile_dataframe before writing any analysis code. "
       "Never reload or re-profile a file you have already loaded ...\n"
       "2. State a short plan, then execute it step by step with pandas/numpy.\n"
       "3. Only use column names you have actually seen in the profile; do not guess columns.\n"
       "4. Back a quantitative claim with a chart: draw it, then call save_chart.\n"
       "5. Finish with final_answer, naming the columns you used and the chart path."
   )

   with build_quill(instructions=QUILL_INSTRUCTIONS) as agent:
       # instructions are APPENDED to the system prompt — they don't replace it
       assert "meticulous data analyst" in agent.system_prompt   # our brief landed
       assert "load_dataset" in agent.system_prompt              # the injected tool list survives
   ```
   Do **NOT** edit `agent.prompt_templates["system_prompt"]` directly — the docs call that
   "generally not advised": you would drop the Jinja2-injected tool list, authorized imports, and
   code-block tags, and break on the next library update. (In the repo, omitting `instructions`
   gives this `QUILL_INSTRUCTIONS` default; `instructions=""` gives the bare prompt for the bench
   baseline.)

5. **Sharpen the tool docstrings** (principle 3 — *write better tools*, T3.12). In
   `quill/tools/data.py`, rewrite the docstrings of `load_dataset` / `profile_dataframe` /
   `save_chart` to state the supported formats, the date format for this dataset (`'%Y-%m'`), an
   example, and **exactly what the tool returns** — while keeping their **frozen M3 signatures**
   (name, inputs, output_type) byte-for-byte the same. Each tool already `print()`s a useful
   one-line summary and raises an informative `ValueError` on a bad path; verify both. The "before
   vs after" for `save_chart`:
   ```text
   before: "Save a chart."
   after : "Save the CURRENT matplotlib figure ... and RETURN its saved file path as a string,
            e.g. 'outputs/category_revenue.png'. Draw your chart FIRST ... do NOT use plt.show()
            ... Raises ValueError if no figure has been drawn yet."
   ```
   The docstring **is** the interface the model reads (it is injected into the system prompt), so a
   precise one is the cheapest reliability win there is.

6. **Re-measure and compare.** `quill/bench.py` runs baseline vs improved and prints the drop.
   Count `ActionStep`s (the work), and track `PlanningStep`s separately so planning is never
   counted as "free":
   ```python
   from quill.bench import run_and_count, format_report

   with build_quill(instructions="") as base:           # baseline
       baseline = run_and_count(base, task)
   with build_quill(planning_interval=3) as impr:        # improved (default sharpened instructions)
       improved = run_and_count(impr, task)
   print(format_report(baseline, improved))
   ```
   Be honest about it: the numbers vary run to run, and a planned run has MORE *total* steps than
   its `ActionStep` count (the plans are extra LLM calls). The fair comparison is on `ActionStep`s.

7. **Tests** (`tests/smoke_test.py`). The planning and instructions tests are **offline and spend
   zero tokens**: the shared `FakeModel` scripts both the plan (ending in `<end_plan>`) and the
   action, so a real `PlanningStep` lands in `agent.memory.steps`. They assert: a `PlanningStep`
   appears when `planning_interval` is set and none when it isn't; `instructions=` is appended to
   `agent.system_prompt` (and the tool list survives); the tool docstrings are non-empty; the
   frozen signatures are unchanged; and the bench counts steps honestly. The real analysis +
   bench run is `live`-marked (budget: ~4 LLM runs; skipped without `HF_TOKEN`).
   ```bash
   uv run pytest module-07/tests/                    # offline
   QUILL_LIVE_TESTS=1 uv run pytest module-07/tests/ # also the real planning + bench runs
   ```

## Try it yourself (not graded)

1. **Find the planning break-even.** Vary `planning_interval` over `1`, `3`, `5`, `None` on the
   same question and chart `ActionStep`s vs total LLM calls. Find the point where planning stops
   paying for itself (hint: on a short task, `planning_interval=1` adds a plan per step for no
   gain).
2. **Constrain the order with `instructions`.** Rewrite the `instructions` to forbid any web
   search until the local analysis is done, and watch the effect on the trajectory in
   `agent.replay()`.

Verified against **smolagents 1.26.0**. `instructions=` is *appended* to the system prompt — never
edit `prompt_templates["system_prompt"]` ("generally not advised"). Never use `agent.logs`
(removed in 1.21.0) — always `agent.memory.steps` / `agent.replay()`.
