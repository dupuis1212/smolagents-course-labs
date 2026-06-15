# Module 4 — Models and Providers: What Powers Your Agent

Until now Quill's model was hard-wired in `agent.py`. This module introduces the **frozen model
contract**: `quill/config.py` with `Settings` and `make_model(role="analyst", **overrides)` —
the **one place** that decides what powers Quill. The default is `InferenceClientModel` with an
explicit coder `model_id`; flip two environment variables to swap to LiteLLM (100+ hosted
providers) or a local model, with **no edit to the agent code**. And `quill/run.py` prints the
**token cost of every run** via `agent.monitor.get_total_token_counts()`.

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit,litellm,openai]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-04/.env.example module-04/.env   # then put your HF token in it

# Default: Hugging Face Inference Providers (free tier, one HF_TOKEN, provider="auto")
uv run python -m quill.run data/sales.csv "Which product category grew fastest last quarter?"
```

prints Quill's answer, then:

```text
[Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct
[Quill] Run cost — input tokens: 8,142 | output tokens: 1,203 | total: 9,345
```

Swap the model in one place — **no agent-code change**:

```bash
# A hosted frontier model via LiteLLM (needs that provider's key, e.g. OPENAI_API_KEY)
QUILL_MODEL_BACKEND=litellm QUILL_MODEL_ID="gpt-4o" \
  uv run python -m quill.run data/sales.csv "Which category grew fastest last quarter?"

# A local model via Ollama (served at localhost:11434; num_ctx raised to 8192 for you)
QUILL_MODEL_BACKEND=local QUILL_MODEL_ID="ollama_chat/qwen2.5-coder" \
  uv run python -m quill.run data/sales.csv "Which category grew fastest last quarter?"
```

Run it from inside `module-04/` so `data/sales.csv`, `outputs/`, and the `quill` package
resolve.

## Test it

```bash
uv run pytest module-04/tests/                       # offline (no token needed)
QUILL_LIVE_TESTS=1 uv run pytest module-04/tests/    # also runs the real LLM runs (needs HF_TOKEN)
```

The offline tests run with no network. The model-**factory** tests construct
`InferenceClientModel` / `LiteLLMModel` objects and assert only their class and `model_id`
(these classes build lazily — the first HTTP call happens on `.generate()`, which the offline
tests never call), monkeypatching `QUILL_MODEL_BACKEND`/`QUILL_MODEL_ID` to prove the swap.
The agent-loop and tool tests use the shared `FakeModel`. The `Monitor`/`TokenUsage` cost
accessor is exercised (it returns a `TokenUsage`; the offline fake reports `0`, the live test
asserts `> 0`). Every Module 2/3 test (pandas, self-correction, `RunResult`, the toolbox)
still passes here.

## The model layer

| Backend (`QUILL_MODEL_BACKEND`) | Class built by `make_model()` | Notes |
|---|---|---|
| `hf` (default) | `InferenceClientModel(model_id=...)` | HF Inference Providers; `provider="auto"` routes via the HF router; token from `HF_TOKEN`. The **default** `model_id` we pin is `Qwen/Qwen2.5-Coder-32B-Instruct` — explicit, never the library default (which is "subject to change"). |
| `litellm` | `LiteLLMModel(model_id="<provider>/<model>")` | 100+ providers (`gpt-4o`, `anthropic/claude-3-5-sonnet-latest`, `gemini/...`). Needs `[litellm]` and the provider's key. |
| `local` | Ollama via `LiteLLMModel("ollama_chat/<model>", api_base="http://localhost:11434", num_ctx=8192)` | Easy laptop path. `num_ctx` is raised above Ollama's 2048 default, which "fails horribly" for a multi-step agent. |

`make_model(role="analyst", **overrides)` is the **frozen contract** every later module reuses.
`role` is part of the signature now (M4 uses the default for every role); `**overrides` forward
completion kwargs (`temperature`, `max_tokens`), a `provider=`, or `requests_per_minute=` (the
`ApiModel` client-side rate-limit guard) straight to the constructor.

## What this module deliberately does NOT do

We wire the **model** layer now; the executor that runs Quill's Python safely is **Module 5**
(no `executor_type`, no sandbox, no `"*"` wildcard — Quill still runs in the local executor).
No structured outputs / `response_format` / `QuillReport` (Module 8) — `response_format` is just
a parameter on `generate` here, not driven. No streaming / `step_callbacks` (Module 6), no
multi-agent (Module 10), no OpenTelemetry tracing (Module 14) — we measure tokens via `Monitor`
only. `LiteLLMRouterModel`, `bill_to=`, and `VLLMModel` are described in the article but are not
on Quill's required path.

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
