# Lab 4 — One place to swap the model: `make_model()` and token cost in plain sight

**Goal:** give Quill `quill/config.py` with the **frozen** `make_model(role)` factory — an
`InferenceClientModel` by default (explicit coder `model_id`), one env var to swap to LiteLLM
or a local model — and print the token cost of every run. After this lab, nothing in
`agent.py` knows about a provider, and "change the model in all of Quill" is a one-line change.

**You'll see:** Quill answer a question on `data/sales.csv`, then a line naming the backend and
model, then a line with the run's token cost — and the same command swap to `gpt-4o` with no
edit to the agent.

**Observable result:**

```bash
uv run python -m quill.run data/sales.csv "Which product category grew fastest last quarter?"
```

```text
...
[Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct
[Quill] Run cost — input tokens: 8,142 | output tokens: 1,203 | total: 9,345
```

## Steps

1. **Setup** — copy the Module 3 state (the cumulative rule: M4 code must still pass the M1–M4
   smoke tests), then sync the pins. Module 4 adds the `[litellm]` extra (the swap backend) and
   the optional `[openai]` extra:
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit,litellm,openai]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
   cp module-04/.env.example module-04/.env   # then add your HF token; never commit .env
   ```
   Create a **fine-grained** HF token with the "Make calls to Inference Providers" scope (the
   `.env.example` reminds you). Provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …) are
   only needed for the `litellm` backend.

2. **`Settings` (`quill/config.py`)** — a tiny config object that reads the environment, so
   there is exactly ONE place that knows which backend and model Quill uses:
   ```python
   DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"   # EXPLICIT coder pin

   class Settings:
       # MODEL_BACKEND ∈ {hf, litellm, local}; MODEL_ID is the backend-specific model.
       # (In the repo these are class properties that re-read the env on each access, so the
       #  swap works even if config was imported before the env var was set.)
       MODEL_BACKEND = os.environ.get("QUILL_MODEL_BACKEND", "hf").lower()
       MODEL_ID = os.environ.get("QUILL_MODEL_ID", DEFAULT_MODEL_ID)
   ```
   The default `MODEL_ID` is **explicit**. We do **not** rely on `InferenceClientModel`'s own
   default `model_id` (`Qwen/Qwen3-Next-80B-A3B-Thinking` as of smolagents 1.26.0) — it is
   documented as "subject to change", and a CodeAgent that writes and runs Python every step
   wants a code-capable model you chose on purpose.

3. **`make_model(role="analyst", **overrides)` (`quill/config.py`)** — the **frozen** factory.
   It dispatches on `Settings.MODEL_BACKEND` and returns a `smolagents.Model`:
   ```python
   from smolagents import InferenceClientModel, LiteLLMModel, Model

   def make_model(role: str = "analyst", **overrides) -> Model:
       backend = Settings.MODEL_BACKEND
       if backend == "hf":
           # token comes from HF_TOKEN in the env; provider defaults to "auto" (HF router).
           return InferenceClientModel(model_id=Settings.MODEL_ID, **overrides)
       if backend == "litellm":
           # id format "<provider>/<model>": "gpt-4o", "anthropic/claude-3-5-sonnet-latest", …
           return LiteLLMModel(model_id=Settings.MODEL_ID, **overrides)
       if backend == "local":
           # Ollama via LiteLLM; num_ctx raised above the 2048 default (it "fails horribly").
           model_id = Settings.MODEL_ID if Settings.MODEL_ID != DEFAULT_MODEL_ID else "ollama_chat/llama3.2"
           return LiteLLMModel(model_id=model_id, api_base="http://localhost:11434",
                               num_ctx=8192, **overrides)
       raise ValueError(f"Unknown QUILL_MODEL_BACKEND {backend!r} (hf, litellm, local).")
   ```
   Notes that matter:
   - `role` is part of the **frozen signature** now (later modules can pick a cheaper model for
     a researcher sub-agent vs the analyst manager); M4 uses the default for every role.
   - `**overrides` is the uniform smolagents pattern: completion kwargs (`temperature`,
     `max_tokens`, `top_p`) are passed at **init** and forwarded on every call. You can also
     pass `provider="together"` or `requests_per_minute=` (an `ApiModel` rate-limit guard).
   - We fail **loud** on an unknown backend — a typo should never silently send you elsewhere.
   - The **token is never hard-coded**: `InferenceClientModel` reads `HF_TOKEN`, `LiteLLMModel`
     reads the provider's key, all from the environment.

4. **`build_quill` calls `make_model()` (`quill/agent.py`)** — delete the hard-wired model;
   `agent.py` no longer imports any model class:
   ```python
   from .config import make_model

   def build_quill(model=None):
       return CodeAgent(
           tools=[load_dataset, profile_dataframe, save_chart(), WebSearchTool(), VisitWebpageTool()],
           model=model or make_model(role="analyst"),     # ONE place to choose what powers Quill
           additional_authorized_imports=QUILL_IMPORTS,
           max_steps=8,
       )
   ```
   The optional `model=` is still how tests inject an offline fake model; otherwise
   `make_model()` owns the default.

5. **`quill/run.py` — the cost-aware CLI** — run Quill, then print the backend, model, and the
   run's token cost from `agent.monitor.get_total_token_counts()` (a `TokenUsage` aggregated
   across every step). Read the cost via the `Monitor` — **never** `agent.logs` or old token
   attributes (removed in 1.21.0):
   ```python
   from .agent import build_quill, build_task
   from .config import Settings

   def main(argv=None):
       args = sys.argv[1:] if argv is None else argv
       csv_path = args[0] if args else "data/sales.csv"
       question = args[1] if len(args) > 1 else "Which category grew fastest last quarter?"

       agent = build_quill()
       print(f"[Quill] Backend: {Settings.MODEL_BACKEND} | Model: {Settings.MODEL_ID}")
       answer = agent.run(build_task(csv_path, question))
       print(answer)

       u = agent.monitor.get_total_token_counts()   # TokenUsage(input, output, total)
       print(f"[Quill] Run cost — input tokens: {u.input_tokens:,} | "
             f"output tokens: {u.output_tokens:,} | total: {u.total_tokens:,}")
       return 0
   ```
   No silent `try/except`: if a run errors, you see it.

6. **Run it, then swap with no agent edit:**
   ```bash
   uv run python -m quill.run data/sales.csv "Which product category grew fastest last quarter?"
   # [Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct
   # [Quill] Run cost — input tokens: 8,142 | output tokens: 1,203 | total: 9,345

   QUILL_MODEL_BACKEND=litellm QUILL_MODEL_ID="gpt-4o" \
     uv run python -m quill.run data/sales.csv "Which product category grew fastest last quarter?"
   # [Quill] Backend: litellm | Model: gpt-4o
   ```
   A multi-step CodeAgent burns tokens fast — the HF free tier is ~$0.10/mo of routed credits
   (as of smolagents 1.26.0, subject to change), so that cost line is your guard-rail.

7. **Smoke tests** (`tests/smoke_test.py`) — offline, no network:
   - `make_model("analyst")` returns an `InferenceClientModel` with the coder `model_id` when
     `QUILL_MODEL_BACKEND` is unset (constructing it touches no network — we assert class +
     `model_id` only; the first HTTP call would be on `.generate()`, which we never call).
   - monkeypatch `QUILL_MODEL_BACKEND=litellm` → `make_model()` returns a `LiteLLMModel`;
     `QUILL_MODEL_BACKEND=local` → an Ollama `LiteLLMModel`; a bogus backend raises `ValueError`.
   - the swap flows through `build_quill()` with no agent edit; `agent.monitor` is a `Monitor`
     and `get_total_token_counts()` returns a `TokenUsage`.
   - The **live** tests (one real run that reports `total_tokens > 0`, plus an optional LiteLLM
     swap) are `@pytest.mark.live`, skipped unless `QUILL_LIVE_TESTS=1`, and skip cleanly
     without `HF_TOKEN`. Live budget: ~1–2 LLM runs.
   ```bash
   uv run pytest module-04/tests/                       # offline
   QUILL_LIVE_TESTS=1 uv run pytest module-04/tests/    # + the real runs
   ```

## Try it yourself

1. Set `QUILL_MODEL_BACKEND=litellm QUILL_MODEL_ID="anthropic/claude-3-5-sonnet-latest"` (needs
   `ANTHROPIC_API_KEY`) and compare the number of steps and the token cost against the default
   on the same question.
2. Add `requests_per_minute=` to a `make_model()` override and watch the behavior when Quill
   chains calls — it is the client-side rate-limit guard inherited from `ApiModel`.

## What this lab does NOT do (yet)

We wire the **model** layer now; the executor that runs Quill's Python safely is **Module 5**
(no `executor_type`, no sandbox, no `"*"` wildcard — Quill still runs in the local executor).
No structured outputs / `response_format` / `QuillReport` / `final_answer_checks` (Module 8) —
`response_format` is only a parameter on `generate` here. No streaming / `step_callbacks`
(Module 6), no multi-agent (Module 10), no OpenTelemetry tracing (Module 14) — we measure
tokens via `Monitor` only. `LiteLLMRouterModel`, `bill_to=`, and `VLLMModel` are described in
the article but are not on Quill's required path. Verified against **smolagents 1.26.0**.
