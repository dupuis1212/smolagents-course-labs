# Module 3 — Tools: Giving Quill New Powers

Quill stops re-writing throwaway pandas in every run and gains its first **reusable, frozen
toolbox**. `quill/tools/data.py` ships three data tools — `load_dataset` and
`profile_dataframe` (both via the `@tool` decorator) and `save_chart` (a `Tool` subclass that
boots matplotlib lazily in `setup()`) — and `build_quill()` wires them in alongside
`WebSearchTool()` and `VisitWebpageTool()` for web access.

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-03/.env.example module-03/.env   # then put your HF token in it

uv run python -m quill.agent data/sales.csv "Which product category grew fastest, and back it up with a saved chart?"
# -> the trajectory (Quill calls profile_dataframe, writes pandas, draws a chart, calls
#    save_chart), the final answer (with the saved chart path), then a RunResult recap.
#    A PNG appears under outputs/.
```

Run it from inside `module-03/` so `data/sales.csv`, `outputs/`, and the `quill` package
resolve.

## Test it

```bash
uv run pytest module-03/tests/                       # offline (FakeModel) — no token needed
QUILL_LIVE_TESTS=1 uv run pytest module-03/tests/    # also runs the 1 real LLM run (needs HF_TOKEN)
```

The offline tests run with no network: they instantiate all three tools (validation passes at
construction), call `profile_dataframe("data/sales.csv")` directly and assert a non-empty
summary, call `save_chart` on a real matplotlib figure and assert a PNG path is returned and
the file exists, verify lazy `setup()` (matplotlib's `Agg` backend boots only on the first
call), and drive Quill end to end — profile → chart → `save_chart` → `final_answer` — through
the shared `FakeModel`. The Module 2 loop tests (pandas, self-correction, `RunResult`,
`replay()`) all still pass here too.

## The toolbox

| Tool | Built with | What it does |
|---|---|---|
| `load_dataset(path) -> str` | `@tool` | Load a CSV/Parquet, print + return a shape/columns summary |
| `profile_dataframe(path) -> str` | `@tool` | Schema, dtypes, `describe()`, missing-value counts as text |
| `save_chart` | `Tool` subclass | `setup()` forces the non-interactive `Agg` backend; `forward` saves the current figure to `outputs/` and returns its path |
| `web_search` | `WebSearchTool()` | DuckDuckGo by default (needs `ddgs`); 4 default tools share `name="web_search"` — use exactly one |
| `visit_webpage` | `VisitWebpageTool()` | Fetch a URL, convert HTML → markdown (needs `markdownify`) |

`build_quill` does **not** pass `add_base_tools=True`: `FinalAnswerTool` is always added, and
a `CodeAgent` excludes the `python_interpreter` tool anyway (it already runs Python).

## What this module deliberately does NOT do

No `make_model()` / provider swap (Module 4) — the `model_id` stays passed directly. Local
executor only, no Docker/E2B (Module 5). No MCP, no `Tool.from_hub`/`from_langchain`, no
`push_to_hub`/`save`, no `ToolCollection` (Module 9) — the tools are *written* in pushable
style but are not pushed. No multi-agent (Module 10), no real image/vision input (Module 11) —
Agent Types are explained, not exercised in multimodal. No `QuillReport` /
`final_answer_checks` (Module 8).

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
