# Module 2 — Your First CodeAgent: The ReAct Loop, Step by Step

**Quill v0** is born here: `quill/agent.py` (`build_quill()`) returns a `CodeAgent` that,
given a CSV and a question, writes and runs pandas to answer it, then calls `final_answer`.
You read exactly how it got there with `agent.replay()` and `agent.memory.steps`.

## Run it

```bash
uv venv --python 3.11 && uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3"
cp module-02/.env.example module-02/.env   # then put your HF token in it

uv run python -m quill.agent data/sales.csv "Which category grew fastest from the first to the last quarter of 2025?"
# -> the agent's trajectory (Thought/Code/Observation per step), the final answer,
#    then a RunResult recap (state=success, step count, token usage).
```

Run it from inside `module-02/` so `data/sales.csv` and the `quill` package resolve.

## Test it

```bash
uv run pytest module-02/tests/                       # offline (FakeModel) — no token needed
QUILL_LIVE_TESTS=1 uv run pytest module-02/tests/    # also runs the 1 real LLM run (needs HF_TOKEN)
```

The offline tests exercise the whole loop with no network: pandas over the CSV,
`final_answer`, the self-correction moment (a `KeyError` captured in `ActionStep.error`,
then fixed on the next step), `RunResult`, and reading the trajectory via
`agent.memory.steps` / `agent.replay()`.

## What this module deliberately does NOT do

`tools=[]` (no custom tools — Module 3), one explicit model_id (no `make_model()` — Module 4),
local executor only (no Docker/E2B sandbox or threat model — Module 5), no multi-turn memory
or deep step inspection (Module 6), no `QuillReport` (Module 8). Every module from here adds
exactly one capability.

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
