# Lab 14 — Observability and evaluation: is Quill any good?

**Goal:** make Quill **observable** (OpenTelemetry traces → Langfuse or Phoenix) and **measurable**
(an eval harness — a frozen golden set + an LLM-as-judge over `QuillReport` — that reports Task
Success Rate + cost/run behind a regression gate).

**You'll see:** `SmolagentsInstrumentor().instrument()` turning tracing on in one line (imported from
`openinference.instrumentation.smolagents`, called **BEFORE** the agent is built); a trace where the
`web_researcher` span nests under the manager span; an **LLM-as-judge** that scores a `QuillReport`
with a numeric rubric, evidence-before-score, and structured JSON — never grading itself; a
**deterministic** `citations` check (no LLM); a **regression gate** that exits non-zero when TSR
drops or cost/run jumps; and cost/run read via `Monitor.get_total_token_counts()` (M4), steps via
`agent.memory.steps` (M6) — never the removed `agent.logs`.

**Observable result:**

```bash
uv run python -m quill.eval.run_evals --out eval/results/run-baseline.json
```

→ runs Quill on each item of `quill/eval/golden_set.json`, scores each run (judge + deterministic
checks), writes `eval/results/run-baseline.json`, and prints:

```text
Golden set: 5 tasks · model=Qwen/Qwen2.5-Coder-32B-Instruct
TSR: 0.80 (4/5)  ·  avg report_quality: 4.6/6  ·  avg steps: 6.2  ·  cost/run: ~12.4k tokens
Regression gate: PASS (TSR>=0.70, cost/run no cap)
```

(exit code **0** if the gate passes, **non-zero** otherwise). In parallel, a normal Quill run with
`QUILL_TELEMETRY=phoenix` makes the trace appear in Phoenix (nested manager → `web_researcher`
spans).

---

## Step 0 — Setup (copy the cumulative state, install the telemetry extra)

This module starts from the Module 13 state (`module-13/quill/` copied to `module-14/quill/`). The
code of M14 must still pass the smoke tests of every module ≤ 14 (06 §5.3).

```bash
uv pip install 'smolagents[telemetry]==1.26.0'
# pulls arize-phoenix + opentelemetry-sdk + opentelemetry-exporter-otlp +
# openinference-instrumentation-smolagents>=0.1.15.
# Langfuse-only (skip arize-phoenix): pip install openinference-instrumentation-smolagents langfuse
```

Copy `.env.example` to `.env` and (optionally) set `QUILL_TELEMETRY` + the backend keys. The eval
harness itself needs **no** telemetry backend — it scores via `Monitor` + `agent.memory.steps`.

> Note (telemetry semantic conventions are still evolving as of smolagents 1.26.0): the span
> attribute names may move. Re-verify the OpenInference instrumentation version at build time.

## Step 1 — Telemetry (`quill/telemetry.py`, NEW)

One function, `instrument(backend=None)`, that calls `SmolagentsInstrumentor().instrument()` **BEFORE
the agent is built** (06 §2). Backend chosen by `QUILL_TELEMETRY ∈ {none, langfuse, phoenix}` (default
`none` = a clean no-op, so a run with no backend is never broken):

```python
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

def instrument(backend=None) -> bool:
    resolved = resolve_backend(backend)         # QUILL_TELEMETRY, default "none"
    if resolved == "none":
        return False                            # no-op: tracing stays OFF
    if resolved == "phoenix":
        from phoenix.otel import register; register()           # local collector
    elif resolved == "langfuse":
        from langfuse import get_client; get_client().auth_check()  # verify keys first
    SmolagentsInstrumentor().instrument()        # the ONE backend-agnostic call
    return True
```

`quill/__main__.py` calls `instrument()` at the top of `_entrypoint()` — BEFORE `build_quill(...)`.
The import is `openinference.instrumentation.smolagents`, **not** `from smolagents import ...`.

For Phoenix, start the collector and run Quill:

```bash
python -m phoenix.server.main serve                  # UI at http://0.0.0.0:6006/projects/
QUILL_TELEMETRY=phoenix uv run python -m quill "Which category grew fastest?" --data data/sales.csv
```

Open the trace: a root span (the run), an LLM span per call, a tool span per call, and the
`web_researcher` span **nested** under the manager span.

## Step 2 — The golden set (`quill/eval/golden_set.json`, NEW — FROZEN format)

A list of `{id, question, dataset, expected_points[], min_sources}` over `data/sales.csv` (inherited
M2, never renamed). `expected_points[]` are the facts the report MUST contain (the judge scores them);
`min_sources` is the minimum number of cited sources expected. One item is pure analysis
(`min_sources: 0`); another needs a web source (`min_sources >= 1`, exercising `web_researcher`).

```json
{
  "id": "net-rev-definition",
  "question": "What does the net_rev column mean, and what is Team's total net_rev?",
  "dataset": "data/sales.csv",
  "expected_points": ["net_rev is net revenue: gross revenue minus refunds",
                      "Team has the highest total net_rev"],
  "min_sources": 1
}
```

## Step 3 — The judge (`quill/eval/judge.py`, NEW)

`judge_report(report: QuillReport, item: dict, model) -> dict` imports `QuillReport` from
`quill/report.py` (FROZEN M8 — reused, never modified), builds a **rubric** prompt (coverage 0-2,
grounding 0-2, citations 0-2), demands **evidence-before-score** (rationale BEFORE scores) and
**structured output** (JSON `{rationale, scores{...}, verdict}`). The judge goes through `make_model`
with a SEPARATE `model_id` (`QUILL_JUDGE_MODEL_ID`) — ideally a different/stronger model, **never
self-grading**. Calibration (`calibration_correlation`) is documented + optional: compare the judge's
scores to your human labels (~0.80 as an order of magnitude), re-run periodically.

## Step 4 — The harness (`quill/eval/run_evals.py`, NEW — FROZEN format)

CLI `--out eval/results/run-<name>.json`. For each item: `build_quill(...)`,
`agent.run(build_report_task(dataset, question))` → `QuillReport`; then

- `task_success` — the judge (does it cover the `expected_points`?),
- `report_quality` — the judge's rubric total (/6),
- `citations` — `len(report.sources) >= min_sources` (**deterministic, no LLM**),
- `steps` — `len([s for s in agent.memory.steps if isinstance(s, ActionStep)])` (M6),
- `cost` — `agent.monitor.get_total_token_counts().total_tokens` (M4 — never the removed attrs).

It aggregates `{run_name, model, scores: [...], aggregate: {TSR, avg_report_quality, avg_steps,
cost_per_run}}`, then applies the **regression gate**: compare the aggregate to `QUILL_EVAL_MIN_TSR`
(default 0.70) and `QUILL_EVAL_MAX_COST` → `sys.exit(1)` if either fails. Writes the JSON to
`eval/results/`.

## Step 5 — Baseline, then catch a regression

```bash
uv run python -m quill.eval.run_evals --out eval/results/run-baseline.json   # the baseline
# ... tweak an instruction "to improve it" ...
uv run python -m quill.eval.run_evals --out eval/results/run-candidate.json  # the candidate
```

Compare `run-baseline.json` vs `run-candidate.json`: a "small prompt tweak" that drops TSR from 0.8
to 0.6 or bumps cost/run 40% fails the gate (non-zero exit) — caught BEFORE you deploy. With no
telemetry backend, read the trajectory in-process with `agent.replay()` (M6).

> Cost (06 §2): one eval run = N Quill runs + N judge calls, and the judge ALSO costs tokens. With
> the HF free tier at ~$0.10/month (as of smolagents 1.26.0, subject to change), a 5-item multi-step
> golden set can eat a real slice — the cost/run column is not cosmetic.

## Step 6 — Test it

```bash
uv run pytest module-14/tests/                            # offline (no token, no network, no backend)
QUILL_LIVE_TESTS=1 uv run pytest module-14/tests/ -m live # ONE real golden-set item (Quill + judge)
```

The offline tests assert: `run_evals` produces a JSON with the EXACT frozen keys; the judge takes a
`QuillReport` and returns parsable structured output; `citations` is deterministic (no LLM); the gate
exits non-zero when TSR < threshold; `telemetry.instrument()` is a clean no-op with
`QUILL_TELEMETRY=none`; the entry point instruments BEFORE building the agent. The `live` test
(skipped by default, skips cleanly without `HF_TOKEN`) runs ONE real golden item end to end.

## Try it yourself

1. Run the harness with **two different `model_id`s** for Quill on the SAME golden set; compare
   `run-modelA.json` vs `run-modelB.json` (TSR, report_quality, cost/run) — is the cheaper model
   worth its TSR?
2. Add a **trap question** (`min_sources >= 1`, needs a web source), then run the harness once WITH
   `web_researcher` and once with `build_quill(managed_agents=[])` — verify the gate (via `citations`)
   catches the degradation.

## What this lab does NOT do (deferred)

- **No hardening / Approach 2** (timeouts, caps, retries, the whole agent in a remote sandbox) — the
  eval runs on the current `QUILL_EXECUTOR`; that is the capstone (Module 15).
- **No re-deployment** (`push_to_hub` / `GradioUI` / CLI — Module 13); **no Agent-as-a-judge**
  implemented (concept only); **no public benchmark** executed (GAIA/SWE-bench/WebArena are
  references); **no change** to `QuillReport` / `quill/report.py` / `build_quill`.

Verified against **smolagents 1.26.0**.
