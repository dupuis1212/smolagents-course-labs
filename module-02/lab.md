# Lab 2 — Build Quill v0: a CodeAgent that answers a question about a CSV

**Goal:** build **Quill v0** — a `CodeAgent` that, given a CSV and a question, writes and
runs pandas to answer it, calls `final_answer`, and whose full trajectory you read with
`agent.replay()`.

**You'll see:** the agent print its *Thought*, the pandas *Code* it writes, the
*Observation* that comes back — and at least one step where it gets a column name wrong,
hits a `KeyError`, and **fixes itself on the next step** without you touching the code.

**Observable result:**

```bash
uv run python -m quill.agent data/sales.csv "Which category grew fastest from the first to the last quarter of 2025?"
```

prints (1) the trajectory (Thought/Code/Observation per step, via `replay()`), then (2) the
final answer (e.g. `Team`), and (3) a `RunResult` recap: `state=success`, the number of
steps, and the token usage.

## Steps

1. **Setup** — `uv` + Python 3.11/3.12. `pandas` is *not* a smolagents dependency, so add it
   explicitly:
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3"
   cp module-02/.env.example module-02/.env   # then add your HF token; never commit .env
   ```

2. **The data** — `data/sales.csv` is the canonical Quill dataset (it is inherited unchanged
   by every later module — never renamed). It holds **monthly 2025** SaaS figures with the
   columns:

   | column | meaning |
   |---|---|
   | `month` | `YYYY-MM` (2025-01 … 2025-12) |
   | `region_code` | `NA` / `EU` / `APAC` |
   | `category` | `Free` / `Pro` / `Team` |
   | `units` | units sold that month |
   | `net_rev` | net revenue that month |
   | `churn_flag` | monthly churn rate |

   "Which category grew fastest?" is non-trivial: it needs a group-by plus a growth ratio
   (e.g. compare each category's first-quarter total to its last-quarter total, or use
   `pct_change`).

3. **`quill/agent.py`** — `build_quill(model=None) -> CodeAgent`:
   ```python
   from smolagents import CodeAgent, InferenceClientModel

   def build_quill(model=None):
       return CodeAgent(
           tools=[],                                   # final_answer is always there; M3 adds tools
           model=model or InferenceClientModel(
               model_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # explicit: default is "subject to change"
               token=os.environ.get("HF_TOKEN"),
           ),
           additional_authorized_imports=["pandas", "numpy"],  # the sandbox blocks these by default
           max_steps=8,                                # the default is 20; a single-CSV job needs far fewer
       )
   ```
   `model=None` is deliberate: it lets tests pass a fake model, and it leaves a clean seam
   for Module 4 to swap in `make_model()`.

4. **The CLI** (`python -m quill.agent <csv> <question>`) — read the CSV path and question
   from `sys.argv`, phrase the task so the agent loads the file itself, and run with the full
   result:
   ```python
   task = f"The CSV is at {csv_path}. Question: {question}\nLoad it with pandas, inspect the columns first, then call final_answer."
   result = agent.run(task, return_full_result=True)
   ```

5. **Inspect the trajectory** — after the run, read it the supported way:
   ```python
   from smolagents import ActionStep

   agent.replay()                       # pretty-prints every step: Thought / Code / Observation
   print(result.state, len(result.steps), result.token_usage)
   for step in agent.memory.steps:      # NEVER agent.logs (removed in 1.21.0)
       if isinstance(step, ActionStep):
           print(step.step_number, step.error, (step.observations or "")[:80])
   ```
   `result.steps` is a `list[dict]` (serialized); for the step *objects* use
   `agent.memory.steps`.

6. **The self-correction moment** — on a real run you'll often see something like this. The
   model writes a misspelled column name, the interpreter raises, the error is **captured in
   `ActionStep.error`** (not raised up to crash the run), it's fed back to the model, and the
   next step fixes it:
   ```text
   ─ Step 1 ─
   Thought: I'll group by category and compare quarters.
   <code>
   import pandas as pd
   df = pd.read_csv("data/sales.csv")
   print(df["Catgory"].unique())          # typo
   </code>
   Observation: ... KeyError: 'Catgory' ...

   ─ Step 2 ─
   Thought: There's no "Catgory" column; the column is "category". Let me check the columns.
   <code>
   import pandas as pd
   df = pd.read_csv("data/sales.csv")
   print(df.columns.tolist())
   q = df.assign(quarter=pd.to_datetime(df["month"]).dt.quarter)
   g = q.groupby(["category", "quarter"])["net_rev"].sum().unstack()
   growth = (g[4] - g[1]) / g[1].replace(0, pd.NA)
   final_answer(growth.idxmax())
   </code>
   Observation: Final answer: Team
   ```
   That is the heart of Module 2: a step error is an *opportunity to self-correct*, not a
   run failure. The only thing that truly ends the run is calling `final_answer` (or hitting
   `max_steps`). If your data/model doesn't trigger a mistake, swap in a question with a
   trap-prone column name to *see* the correction happen.

7. **Smoke tests** (`tests/smoke_test.py`) — the offline tests use the shared `FakeModel`
   (see the repo-root `conftest.py`) to drive the loop with no network: pandas over the CSV,
   `final_answer`, the `KeyError`-then-fix path, `RunResult`, and `memory.steps`/`replay()`.
   The `live` test (one real run) is skipped unless `QUILL_LIVE_TESTS=1` and skips cleanly
   without `HF_TOKEN`.
   ```bash
   uv run pytest module-02/tests/                       # offline
   QUILL_LIVE_TESTS=1 uv run pytest module-02/tests/    # + 1 real run
   ```

## Try it yourself

1. Run the same task with `agent.run(task, stream=True)` and iterate the generator to watch
   each `ActionStep` / `FinalAnswerStep` arrive live (streaming internals are Module 6).
2. Drop `max_steps` to `2` on a question that needs several steps and watch
   `result.state == "max_steps_error"` plus the fallback answer.

## What this lab does NOT do (yet)

No custom tool or `@tool`/`Tool` class — `tools=[]` (Module 3). No `make_model()` / LiteLLM
swap / token cost via `Monitor` (Module 4). No Docker/E2B sandbox, no threat model, no `"*"`
wildcard (Module 5). No multi-turn `reset=False` or `step_callbacks` (Module 6). No planning
(Module 7). No `QuillReport` / `final_answer_checks` (Module 8). Quill v0 is deliberately
bare — `tools=[]`, local executor, one explicit model. Every module from here adds exactly
one capability.
