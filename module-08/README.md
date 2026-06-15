# Module 8 — Reliable Agents: Structured Output, Validation, and Errors

Quill from Module 7 is smart and frugal, but it is **not reliable**: ask the same question three
times and you get three shapes — a paragraph, a half-built dict, "growth is strong" with no number
and no saved chart — and it "cites" the web without ever giving a URL. A report no one can audit,
compare, or score. This module turns Quill's answer into a **contract**: a structured `QuillReport`
that is **validated** before it is accepted, and that **self-corrects** when a validation fails.

The contrarian point: **structured output does not validate your answer.** smolagents has two
"structured output" mechanisms, neither of which checks *content* — they only constrain *form*. The
business rules ("an analysis without a chart isn't a report"; "a web-backed claim needs a source")
live in `final_answer_checks`. Form ≠ validity.

This module **extends `build_quill()` by addition** (no signature breaks) with two keyword-only
arguments, and adds the FROZEN `quill/report.py`:

- **`final_answer_checks: list | None`** — when omitted, Quill ships
  `quill_final_answer_checks()`: two **3-arg** `(final_answer, memory, agent) -> bool` validators
  wired into `CodeAgent(final_answer_checks=...)`. `check_has_chart` refuses an answer that is not a
  `QuillReport` or has no saved chart; `check_has_source_for_web_claims` refuses an empty `sources`
  list **when** the run actually called a web tool. A check that returns `False` (or raises) does
  **not** crash — smolagents stores the `AgentError` in `ActionStep.error` and loops, so Quill
  self-corrects. Pass `final_answer_checks=[]` to opt out.
- **`use_structured_outputs_internally: bool = False`** (the OPTIONAL extra, off the mandatory
  path) — when `True`, `CodeAgent` loads `structured_code_agent.yaml` and sets `response_format` on
  every step to make each step's Thought+code parse reliably. It only works with a backend that
  supports `response_format` (Quill's default `InferenceClientModel` does so **only** with
  `provider in {"cerebras","fireworks-ai"}` — `STRUCTURED_GENERATION_PROVIDERS`, as of smolagents
  1.26.0). It does **not** validate the final answer's content.

The new **`quill/report.py`** freezes Quill's output schema (`06-FIL-ROUGE-SPEC §2` — M12 cites it,
M14 scores it):

```python
QuillReport{ question: str, findings: list[str], chart_paths: list[str],
             sources: list[Source], caveats: list[str] }
Source{ url: str, title: str }
```

`to_markdown()` renders the findings with numbered `[n]` citations into `sources`.

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit,litellm,openai,docker]==1.26.0" \
  "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-08/.env.example module-08/.env   # then put your HF token in it
```

**Get a validated report** (makes real model calls, so set `HF_TOKEN`). Run from inside
`module-08/` so `data/sales.csv`, `outputs/`, and the `quill` package resolve. The question is
positional; the CSV comes from `--data`:

```bash
uv run python -m quill "Which category grew fastest, and is that consistent with the public trend?" --data data/sales.csv
```

prints a rejection-then-correction trajectory and a rendered report (your exact run varies — LLMs
are non-deterministic):

```text
Final answer rejected: a report must include at least one saved chart. ...
...                       # ← Quill self-corrects, then answers again
===== REPORT =====
# Which category grew fastest, and is that consistent with the public trend?

## Findings
- Team grew fastest, +38% from Q1 to Q4 [1].

## Charts
- `outputs/category_growth.png`

## Sources
[1] [SaaS Trends 2025](https://example.com/saas-2025)
```

## Test it

```bash
uv run pytest module-08/tests/                    # offline (no token, no Docker)
QUILL_LIVE_TESTS=1 uv run pytest module-08/tests/ # also the real validated-report run (needs HF_TOKEN)
```

The offline tests run with **no network and zero tokens**: the checks are asserted directly as
functions, and the shared `FakeModel` (repo-root `conftest.py`) scripts the report code so the agent
loop runs offline. They prove:

- `QuillReport`/`Source` are the FROZEN schema (exactly 5 / 2 fields); `to_markdown()` renders the
  `[1]` marker and a `[1] [title](url)` line for one source;
- the two checks are 3-arg `(final_answer, memory, agent)`; `check_has_chart` rejects a
  non-`QuillReport` and an empty `chart_paths`; `check_has_source_for_web_claims` rejects empty
  `sources` **only** when a web tool was used (heuristic on the memory's code);
- `build_quill` wires Quill's default checks into `agent.final_answer_checks`, exposes
  `QuillReport`/`Source` to the sandbox **without** widening the frozen import lock, and keeps
  `use_structured_outputs_internally=False` by default;
- a fake-model Quill that builds a complete `QuillReport` and calls `final_answer(report)` **passes**
  and the run returns the `QuillReport`; a bare-string answer is **rejected**, the `AgentError`
  lands in `ActionStep.error`, and the run loops (it does not crash);
- the self-correction demo: no-chart → rejected → draw + save chart → accepted.

Every Module 2/3/4/5/6/7 test still passes here (the carried-forward agent-loop tests pass
`final_answer_checks=[]` to opt out of the new contract, since they assert pre-M8 mechanics).

## The two "structured outputs" (and why neither validates)

| | `response_format` | `use_structured_outputs_internally` |
|---|---|---|
| **Lives on** | `Model.generate(...)` (a model arg) | `CodeAgent.__init__` (an agent arg) |
| **What it controls** | the JSON shape of one model call | the Thought+code format of *every* step |
| **Backend constraint** | `InferenceClientModel`: only `cerebras`/`fireworks-ai` (`STRUCTURED_GENERATION_PROVIDERS`, as of 1.26.0) | inherits the same constraint |
| **YAML loaded** | — | `structured_code_agent.yaml` |
| **Validates the final answer?** | **No** | **No** |

> ⚠️ **Common misconception: "structured output validates my answer."** False. Both mechanisms
> guarantee a *form* (well-typed JSON), never a *content*: a perfectly-typed `QuillReport` can still
> have `chart_paths=[]` and `sources=[]`. Only a `final_answer_check` enforces the business rule.
> `grammar=` (the old TGI path) was **removed in 1.21.0** — use `response_format=`.

## `final_answer_checks`: turning the answer into a contract

The call site in smolagents 1.26.0 (`agents.py` `_validate_final_answer`) is:

```python
assert check_function(final_answer, self.memory, agent=self)   # 3-arg, the 3rd is keyword
```

So a check that returns `False` (or raises) makes that `assert` fail; smolagents wraps it in an
`AgentError` that is **caught and stored in `ActionStep.error`** — the run loops and the model
self-corrects on the next step. The guided-tour 2-arg example only "works" because the 3rd arg is
passed as a keyword; write the **3-arg** form.

## The `AgentError` hierarchy (as of smolagents 1.26.0)

```text
AgentError
├── AgentParsingError        (malformed code/JSON)
├── AgentExecutionError      (failed during execution)
│   ├── AgentToolCallError       (bad arguments passed to a tool)
│   └── AgentToolExecutionError  (the tool itself raised at runtime)
├── AgentGenerationError     (model generation failed — re-raised, not auto-corrected)
└── AgentMaxStepsError       (max_steps reached — the only hard stop)
```

An error in a step is **captured in `ActionStep.error`** rather than crashing the run; it is
re-injected to the model via `write_memory_to_messages()`, so "LLM dumb" errors auto-correct on the
next step. A `final_answer_check` that returns `False` uses **exactly** this mechanism — which is
why an actionable check message is precious. Only `max_steps` ends a run (`state="max_steps_error"`).

## Where reliability lives

| concern | lives in |
|---|---|
| output *shape* | the code Quill writes (a `QuillReport`); optionally `use_structured_outputs_internally` |
| business *rules* | `final_answer_checks` |
| transient failures | auto-correction via `ActionStep.error` + step retries |
| hard stop | `max_steps` |

## What this module deliberately does NOT do

- **No MCP `output_schema` / `structured_output`** (Module 9) — that is a *different* structured
  output, and it is purely informational (no validation).
- **No eval harness / LLM-as-judge** (Module 14) — the checks are runtime validators, not scoring.
  These same `final_answer_checks` become the seed of M14's automated eval.
- **No RAG / retriever to populate `sources`** (Module 12) — here sources come from the web tools.
- **No `response_format` on the default `InferenceClientModel`** (incompatible); **no `grammar=`**
  (removed in 1.21.0).
- **No multi-agents / vision / telemetry / UI.**
- **Never `agent.logs`.** Always `agent.memory.steps` / `agent.replay()`.

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
