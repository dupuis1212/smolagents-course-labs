# Module 1 — What Are Code Agents? From LLM Calls to smolagents

Your dev environment plus your **first `CodeAgent`** — an empty-toolset agent that writes
and runs Python to solve a calculation. No Quill yet: that starts in Module 2, on purpose.

## Run it

```bash
uv venv --python 3.11 && uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2"
cp module-01/.env.example module-01/.env   # then put your HF token in it
uv run python -m quill_intro "Calculate the sum of all integers from 1 to 100"
# -> the agent prints the Python it writes, then: 5050
```

## Test it

```bash
uv run pytest module-01/tests/            # offline (FakeModel) — no token needed
QUILL_LIVE_TESTS=1 uv run pytest module-01/tests/   # also runs the 1 real LLM call (needs HF_TOKEN)
```

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
