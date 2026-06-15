# Lab 1 — Setup + your first CodeAgent

**Goal:** get a working dev environment and run a bare `CodeAgent` that writes and runs
Python to compute an answer.

**You'll see:** the agent prints its *Thought*, the Python *code* it writes, the
*observation*, and the final answer `5050`.

## Steps

1. **Get a Hugging Face token.** Create one at
   <https://huggingface.co/settings/tokens> (fine-grained, scope *"Make calls to
   Inference Providers"*). Put it in `module-01/.env` (copy `.env.example`). Never commit
   `.env`. Heads-up on cost: the free HF Inference credit allowance is tiny and subject to
   change (as of smolagents 1.26.0) — Module 4 shows how to swap to a free provider key or
   a local model.

2. **Install with uv** (Python 3.11/3.12):
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2"
   ```

3. **Write the agent** (`quill_intro/first_agent.py`). Note we pin an explicit
   `model_id` — the library default is "subject to change":
   ```python
   import os
   from smolagents import CodeAgent, InferenceClientModel

   model = InferenceClientModel(
       model_id="Qwen/Qwen2.5-Coder-32B-Instruct",  # explicit: default is "subject to change"
       token=os.environ["HF_TOKEN"],
   )
   agent = CodeAgent(tools=[], model=model)          # an empty toolset is valid and deliberate
   print(agent.run("Calculate the sum of all integers from 1 to 100"))
   ```

4. **Run it:** `uv run python -m quill_intro "Calculate the sum of all integers from 1 to 100"`.
   Watch the agent *write Python* to compute the answer — your first taste of code-as-action.

## Try it yourself

1. Change the task to *"Find the 10th Fibonacci number and tell me if it's prime"* and read
   the code the agent writes.
2. Pass `verbosity_level=2` to the `CodeAgent` to see more of the trajectory.

## What this lab does NOT do (yet)

No tools (`tools=[]`), no CSV/pandas (that's Quill v0 in Module 2), no `make_model()` or
provider swap (Module 4), no sandbox (Module 5), no multi-turn memory or `replay()`
(Module 6), no `QuillReport` (Module 8), no multi-agents (Module 10).
