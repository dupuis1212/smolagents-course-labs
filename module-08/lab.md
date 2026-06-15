# Lab 8 — Make Quill reliable: a validated `QuillReport` + `final_answer_checks`

**Goal:** turn Quill's free-form answer into a **contract**. Quill now returns a structured
`QuillReport` and is *refused* — via `final_answer_checks` — if the report has no saved chart, or
makes a web-backed claim with no source. A rejected report does **not** crash: the agent sees why
and self-corrects on the next step.

**You'll see:** the two 3-arg checks reject an empty/bad report and accept a complete one (asserted
directly, no LLM); a fake-model Quill build a `QuillReport`, call `final_answer(report)`, and pass;
a bare-string answer get rejected and the `AgentError` land in `ActionStep.error` (the
self-correction loop); and `QuillReport.to_markdown()` render numbered `[n]` citations.

**Observable result:**

```bash
uv run python -m quill "Which category grew fastest, and is that consistent with the public trend?" --data data/sales.csv
```

```text
[Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct
...
Final answer rejected: a report must include at least one saved chart. Draw a matplotlib
chart, call save_chart to save it, add the returned path to QuillReport.chart_paths, then
call final_answer again.
...                       # ← Quill self-corrects: draws + saves a chart, answers again
===== REPORT =====
# Which category grew fastest, and is that consistent with the public trend?

## Findings
- Team grew fastest, +38% from Q1 to Q4 [1].

## Charts
- `outputs/category_growth.png`

## Sources
[1] [SaaS Trends 2025](https://example.com/saas-2025)
[Quill] Run cost — input tokens: 12,431 | output tokens: 1,118 | total: 13,549
```

> The exact trajectory varies — LLMs are non-deterministic. What is guaranteed is the *shape*: a
> `QuillReport` that the checks accepted (a saved chart; a source for any web-backed claim).

## Steps

1. **Setup** — copy the Module 7 state (the cumulative rule: M8 code must still pass the M1–M7
   smoke tests), then sync the pins. **No new extra** — `final_answer_checks` is core `smolagents`
   and `QuillReport` is a plain `dataclass`.
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
     "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
   cp module-08/.env.example module-08/.env   # then add your HF token; never commit .env
   ```
   `data/sales.csv` is inherited from Module 2 unchanged. Quill runs `local` by default;
   `QUILL_EXECUTOR` stays configurable (`local`/`docker`/`e2b`).

2. **Freeze the report schema** in `quill/report.py` (this is the FROZEN M8 contract — M12 cites it,
   M14 scores it, so the five/two fields cannot change without updating `06-FIL-ROUGE-SPEC §2`):
   ```python
   from dataclasses import dataclass, field

   @dataclass
   class Source:
       url: str
       title: str

   @dataclass
   class QuillReport:
       question: str
       findings: list[str] = field(default_factory=list)
       chart_paths: list[str] = field(default_factory=list)
       sources: list[Source] = field(default_factory=list)
       caveats: list[str] = field(default_factory=list)
   ```
   Add `to_markdown()` so each finding keeps its `[n]` markers and the Sources section renders
   `[n] [title](url)` — the numbers line up. A one-source report contains `[1]` and
   `[1] [<title>](<url>)`.

3. **Write the two `final_answer_checks`** — 3-arg `(final_answer, memory, agent)`. (The
   guided-tour 2-arg form only "works" because smolagents passes the 3rd as a keyword; write the
   3-arg form.) A check that returns `False` *or raises* rejects the answer; we **raise** so the
   model reads the reason:
   ```python
   def check_has_chart(final_answer, memory, agent) -> bool:
       if not isinstance(final_answer, QuillReport):
           raise ValueError("Final answer rejected: the answer must be a QuillReport ...")
       if not final_answer.chart_paths:
           raise ValueError("Final answer rejected: a report must include at least one saved "
                            "chart. Draw a chart, call save_chart, add the path to chart_paths ...")
       return True

   def check_has_source_for_web_claims(final_answer, memory, agent) -> bool:
       if _used_a_web_tool(memory) and not final_answer.sources:
           raise ValueError("Final answer rejected: this analysis used the web but sources is "
                            "empty. Add a Source(url=..., title=...) and cite it as [n] ...")
       return True
   ```
   `_used_a_web_tool(memory)` scans `memory.steps` for `web_search`/`visit_webpage` in the code each
   step ran (`code_action` and the `python_interpreter` `ToolCall.arguments`). Read `memory.steps` —
   **never** the removed `agent.logs`. Timing note: the validator runs *before* the current step is
   appended to memory, so a web call in the *same* step as `final_answer` isn't visible yet — in
   practice Quill searches earlier and answers later.

4. **Wire the checks into `build_quill`.** The signature is **extended by addition** (every prior
   call site still works); omitting `final_answer_checks` gives Quill's defaults, `[]` opts out:
   ```python
   from quill.agent import build_quill
   agent = build_quill()                              # default checks ON (chart + web-source)
   assert agent.final_answer_checks == [check_has_chart, check_has_source_for_web_claims]
   ```
   Keep the model as `make_model()` (M4). **Do NOT** turn on `response_format` for the default
   `InferenceClientModel` — it only supports it with `provider in {"cerebras","fireworks-ai"}`
   (`STRUCTURED_GENERATION_PROVIDERS`, as of smolagents 1.26.0). The *form* of the report is
   guaranteed by the code Quill writes; the *business rules* live in `final_answer_checks`.
   `build_quill` also injects `QuillReport`/`Source` into the sandbox (via `send_variables`) so the
   agent's code can build a report **without** importing `quill.report` — the frozen import lock
   (M5) deliberately forbids that import, and we do not widen it.

5. **Observe the self-correction.** Ask a question that pushes Quill to conclude without a chart;
   watch the check return reject, the message re-feed, and the corrective step that calls
   `save_chart`. With a real model:
   ```python
   from smolagents import ActionStep
   from quill.agent import build_quill, build_report_task

   with build_quill() as agent:                       # checks ON
       result = agent.run(build_report_task("data/sales.csv",
                          "Which category grew fastest from Q1 to Q4 2025?"),
                          return_full_result=True)
       agent.replay()                                  # see the rejection, then the fix
       rejected = [s for s in agent.memory.steps
                   if isinstance(s, ActionStep) and s.error is not None]
       print("rejections captured:", len(rejected))    # AgentError in step.error, NOT a crash
   print(result.output.to_markdown())                 # a validated QuillReport
   ```
   A rejection is captured in `ActionStep.error` and re-injected to the model — the **same**
   mechanism that recovers from a `KeyError` in the agent's own code. Only `max_steps` ends a run.

6. **(Optional extra, off the mandatory path)** turn on per-step structured outputs *if* your
   backend supports `response_format`:
   ```python
   from quill.config import make_model
   # InferenceClientModel needs provider in cerebras/fireworks-ai for response_format;
   # otherwise use a LiteLLM/OpenAI backend.
   agent = build_quill(model=make_model(backend="litellm"),   # a response_format-capable backend
                       use_structured_outputs_internally=True)  # loads structured_code_agent.yaml
   ```
   This makes each step's Thought+code parse reliably — it does **not** validate the final answer's
   *content* (that is `final_answer_checks`). Form ≠ validity.

7. **Tests** (`tests/smoke_test.py`). The M8 tests are **offline and spend zero tokens**: the
   checks are asserted directly as functions, and a `FakeModel` scripts the report code so a real
   agent loop runs with no network. They assert: the frozen schema (5/2 fields); `to_markdown()`
   renders `[1]` for one source; the 3-arg checks reject a bad answer and accept a complete
   `QuillReport`; a fake-model agent that builds a report and calls `final_answer(report)` passes;
   a bare string is rejected and the `AgentError` lands in `ActionStep.error` (loop, not crash);
   and the self-correction demo (no-chart → rejected → chart → accepted). The carried-forward M2–M7
   agent-loop tests pass `final_answer_checks=[]` to opt out of the new contract (they assert
   pre-M8 mechanics). The real report run is `live`-marked (skipped without `HF_TOKEN`).
   ```bash
   uv run pytest module-08/tests/                    # offline (live/sandbox auto-skip)
   QUILL_LIVE_TESTS=1 uv run pytest module-08/tests/ # also the real validated-report run
   ```

## Try it yourself (not graded)

1. **A third check.** Add a check that refuses a report with `findings=[]` (an analysis with no
   conclusion) or more than N `caveats` (a report drowning in hedges). Keep the 3-arg signature and
   a message the model can act on.
2. **Real structured outputs.** Wire `make_model(backend="litellm")` on a provider that supports
   `response_format`, turn on `use_structured_outputs_internally=True`, and compare the per-step
   parsing success rate against the default. Note it still doesn't validate content — the checks do.

Verified against **smolagents 1.26.0**. `final_answer_checks` are 3-arg `(final_answer, memory,
agent)`; a `False`/raise loops the agent (the error lands in `ActionStep.error`), it does not crash.
`grammar=` is dead (removed in 1.21.0) — use `response_format=`. Never use `agent.logs` (removed in
1.21.0) — always `agent.memory.steps` / `agent.replay()`.
