# smolagents-course-labs — build **Quill**, a code-first data-analyst agent

Companion code for the free course **[smolagents Mastery](https://d2apczqz24upf4.cloudfront.net/courses/smolagents/)**
(part of [aifromzero.dev](https://d2apczqz24upf4.cloudfront.net)). Across 15 modules you build
**Quill**: a code-first data analyst that *writes and runs Python in a sandbox* to clean,
analyze and visualize data, browses the web for context, cross-checks its findings, and returns
a cited report with charts.

Verified against **smolagents 1.26.0** · Python **3.11/3.12** · no GPU required.

## What's here

Each `module-NN/` is a **self-contained, cumulative snapshot**: checking out `module-08/` gives
you everything built through Module 8. The agent lives in `module-NN/quill/` and grows module by
module:

| | Module | What Quill gains |
|---|---|---|
| 01 | First CodeAgent | a bare `CodeAgent` (no Quill yet) |
| 02 | The ReAct loop | Quill v0 — answers questions over a CSV by writing pandas |
| 03 | Tools | `load_dataset`, `profile_dataframe`, `save_chart` + web tools |
| 04 | Models & providers | `make_model()` — one place to swap model/provider |
| 05 | Secure sandboxing | runs in a Docker/E2B sandbox, imports locked |
| 06 | Memory & inspection | multi-turn + a memory-pruning step callback |
| 07 | Planning & good agents | `planning_interval`, sharper instructions |
| 08 | Reliable output | a validated `QuillReport` via `final_answer_checks` |
| 09 | Tool interop | MCP server + Hub-shared tools |
| 10 | Multi-agent | a manager + a `web_researcher` sub-agent |
| 11 | Vision & multimodal | re-reads its own charts; optional vision browser |
| 12 | Agentic RAG | a `RetrieverTool` over a docs corpus → cited answers |
| 13 | Deployment | a Gradio UI + Hub Space + CLI |
| 14 | Observability & eval | OpenTelemetry + an eval harness (golden set + judge) |
| 15 | Capstone | the whole system in a sandbox; release v1.0 |

## Quickstart

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib rank-bm25 pytest
# (later modules add extras: smolagents[litellm,openai,docker,mcp,vision,gradio,telemetry])

cp module-01/.env.example module-01/.env   # add your Hugging Face token
uv run python -m quill_intro "Calculate the sum of all integers from 1 to 100"
```

> The free Hugging Face Inference allowance is small (as of smolagents 1.26.0). Module 4 shows how
> to swap to a free provider key or a local model — no code changes beyond `QUILL_MODEL_BACKEND`.

## Tests

The whole suite runs **offline** with no token or network: a deterministic `FakeModel` (see
[`conftest.py`](conftest.py)) drives the real agent loop, so tools, the local sandbox, memory,
`final_answer`, RAG (BM25) and the eval harness are all exercised for real.

```bash
# offline: every module, no token needed
uv run pytest $(for n in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15; do echo module-$n/tests; done) -q
# -> 1881 passed, 126 skipped

QUILL_LIVE_TESTS=1   uv run pytest ...   # also run real-LLM tests (needs HF_TOKEN)
QUILL_SANDBOX_TESTS=1 uv run pytest ...  # also run Docker-sandbox tests (needs a free port 8888)
```

`live` tests (real LLM) and `sandbox` tests (Docker/E2B) are skipped by default so the offline
suite is the green guarantee.

## License / use

Free to read and run, like the course. Built and verified for teaching.
