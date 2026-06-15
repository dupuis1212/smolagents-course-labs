# Lab 3 — Give Quill a reusable toolbox: `@tool`, the `Tool` class, and web access

**Goal:** turn Quill from "re-write pandas in every run" into an agent with a real,
**frozen toolbox** — `load_dataset` and `profile_dataframe` (via the `@tool` decorator),
`save_chart` (a `Tool` subclass with a lazy `setup()`), plus `WebSearchTool()` and
`VisitWebpageTool()` — all wired into `build_quill()`.

**You'll see:** Quill profile the dataset with a tool, write its own pandas/matplotlib, call
`save_chart` to write a PNG to `outputs/`, and return a `final_answer` that includes the saved
chart path.

**Observable result:**

```bash
uv run python -m quill.agent data/sales.csv "Which product category grew fastest, and back it up with a saved chart?"
```

prints the trajectory (Quill calls `profile_dataframe`, then pandas, then `save_chart`), a
`final_answer` containing the saved chart path, and a `RunResult` recap — and a PNG appears
under `outputs/`.

## Steps

1. **Setup** — copy the Module 2 state (the cumulative rule: M3 code must still pass the M1–M3
   smoke tests), then sync the pins. `matplotlib` is *not* a smolagents dependency, so add it
   explicitly:
   ```bash
   uv venv --python 3.11
   uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
   cp module-03/.env.example module-03/.env   # then add your HF token; never commit .env
   ```
   `[toolkit]` provides `ddgs>=9.0.0` (for `WebSearchTool`, *not* the old `duckduckgo-search`
   package — it was renamed upstream) and `markdownify>=0.14.1` (for `VisitWebpageTool`).

2. **`load_dataset` via `@tool`** (`quill/tools/data.py`) — frozen signature
   `load_dataset(path: str) -> str`. The `@tool` decorator derives the tool's schema from the
   function: the **name** comes from the function name, the **description** from the first
   docstring paragraph, each **input's** description from the `Args:` section, the input
   **types** from the parameter type hints, and the **output_type** from the return hint. So
   type hints on every parameter *and* the return, plus an `Args:` section, are non-negotiable.
   ```python
   from smolagents import tool

   @tool
   def load_dataset(path: str) -> str:
       """Load a tabular dataset (CSV or Parquet) and return a short text summary.

       Args:
           path: Filesystem path to the dataset (a .csv or .parquet file).
       """
       import pandas as pd                      # import INSIDE the function (pushable rule)
       df = pd.read_csv(path)
       summary = f"Loaded {path}: {df.shape[0]} rows x {df.shape[1]} columns. Columns: {list(df.columns)}."
       print(summary)                           # print() -> the next step's Observation
       return summary                           # the return value also comes back as Observation
   ```
   Two habits matter (the full "good tools" theory is Module 7): `print()` a useful one-liner
   so the model has something to reason over, and raise an **informative `ValueError`** when
   the path is missing or the format is unsupported, so the agent can self-correct instead of
   crashing the run. (In the repo the read + error handling live in a small `_read_table`
   helper shared by both data tools.)

3. **`profile_dataframe` via `@tool`** — frozen signature
   `profile_dataframe(path: str) -> str`. Same rules (internal imports, type hints, `Args:`,
   informative `ValueError`). It returns the schema, dtypes, `describe()` statistics, and a
   missing-value count per column as text.

4. **`save_chart` as a `Tool` subclass with `setup()`** — this is the one tool that is *not*
   a `@tool` function, because it needs a one-time, expensive init: selecting matplotlib's
   non-interactive `Agg` backend. That belongs in `setup()`, which smolagents runs **lazily on
   the first call only** (when `not self.is_initialized`), not at construction. You declare
   `name`/`description`/`inputs`/`output_type` as **class attributes** and write `forward`.
   ```python
   from smolagents import Tool

   class save_chart(Tool):                      # canonical name is exactly "save_chart"
       name = "save_chart"
       description = (
           "Save the CURRENT matplotlib figure to outputs/ as a PNG and return its path. "
           "Draw your chart first; do NOT use plt.show() (it saves nothing)."
       )
       inputs = {"filename": {"type": "string",
                              "description": "Optional base filename.", "nullable": True}}
       output_type = "string"

       def setup(self):
           import matplotlib
           matplotlib.use("Agg")                # non-interactive: write files, never display
           super().setup()                      # flips is_initialized so this runs ONCE

       def forward(self, filename=None):
           import os, matplotlib.pyplot as plt
           os.makedirs("outputs", exist_ok=True)
           name = (filename or "chart") + ("" if (filename or "").endswith(".png") else ".png")
           out = os.path.join("outputs", name)
           fig = plt.gcf()
           if not fig.get_axes():
               raise ValueError("No figure to save — draw a chart before calling save_chart.")
           fig.savefig(out, bbox_inches="tight")
           plt.close(fig)
           return out                           # feeds chart_paths of QuillReport (Module 8)
   ```
   Note `__init__` takes no argument other than `self` (we use the base one) — a pushable rule
   so Module 9 can push this tool to the Hub with no rewrite. We do **not** push it here.

5. **Wire it into `build_quill`** (`quill/agent.py`) — instantiate the `Tool` subclass with
   `save_chart()`; the `@tool` functions are already tool objects:
   ```python
   from smolagents import CodeAgent, VisitWebpageTool, WebSearchTool
   from .tools import load_dataset, profile_dataframe, save_chart

   QUILL_IMPORTS = ["pandas", "numpy", "matplotlib.*"]   # superset toward M5; never "*"

   agent = CodeAgent(
       tools=[load_dataset, profile_dataframe, save_chart(),
              WebSearchTool(), VisitWebpageTool()],
       model=...,
       additional_authorized_imports=QUILL_IMPORTS,       # so the agent can plot before saving
       max_steps=8,
   )
   ```
   We add `"matplotlib.*"` to the authorized imports so the agent can draw a figure before
   handing it to `save_chart` — a clean *addition* toward Module 5's frozen list, not a rename,
   and never the `"*"` wildcard. We do **not** pass `add_base_tools=True`: `FinalAnswerTool` is
   already added for every agent, and a `CodeAgent` excludes the `python_interpreter` tool
   anyway (it *is* the interpreter). Use exactly one web-search tool — four default tools share
   `name="web_search"`, and `agent.tools` is a dict keyed by `name`, so two would collide.

6. **Runtime toolbox (T3.11)** — `agent.tools` is a plain `dict` keyed by tool `name`. You can
   add or replace a tool at runtime without rebuilding the agent:
   ```python
   from smolagents import VisitWebpageTool
   agent = build_quill()
   print(sorted(agent.tools))          # ['final_answer','load_dataset','profile_dataframe',
                                       #  'save_chart','visit_webpage','web_search']
   extra = VisitWebpageTool()
   agent.tools[extra.name] = extra     # add/replace by name (same name => replaces)
   ```

7. **Run it and read the trajectory** — same as Module 2 (`agent.replay()` /
   `agent.memory.steps`, never `agent.logs`):
   ```bash
   uv run python -m quill.agent data/sales.csv "Which product category grew fastest, and back it up with a saved chart?"
   ```
   ```text
   ─ Step 1 ─
   Thought: Profile the dataset first, then compute growth and plot it.
   <code>
   summary = profile_dataframe("data/sales.csv")
   print(summary)
   </code>
   Observation: Profile of data/sales.csv  Shape: 108 rows x 6 columns ...

   ─ Step 2 ─
   <code>
   import pandas as pd
   df = pd.read_csv("data/sales.csv")
   q = df.assign(quarter=pd.to_datetime(df["month"]).dt.quarter)
   g = q.groupby(["category","quarter"])["net_rev"].sum().unstack()
   ((g[4]-g[1])/g[1]).plot(kind="bar")
   path = save_chart("category_growth")
   final_answer(f"Team grew fastest. Chart saved at {path}")
   </code>
   Observation: Final answer: Team grew fastest. Chart saved at outputs/category_growth.png
   ```

8. **Smoke tests** (`tests/smoke_test.py`) — offline (the shared `FakeModel`, no network):
   instantiate all three tools and assert `validate_arguments()` passes; assert
   `load_dataset.name`/`inputs`/`output_type` are the frozen schema; call
   `profile_dataframe("data/sales.csv")` directly and assert a non-empty summary; call
   `save_chart` on a real figure and assert a PNG path is returned and the file exists; assert
   `setup()` is lazy (`is_initialized` is `False` until the first call). The **live** test (one
   real run that produces a PNG) is marked `@pytest.mark.live`, skipped unless
   `QUILL_LIVE_TESTS=1`, and skips cleanly without `HF_TOKEN`. Live budget: ~1 LLM run.
   ```bash
   uv run pytest module-03/tests/                       # offline
   QUILL_LIVE_TESTS=1 uv run pytest module-03/tests/    # + 1 real run
   ```

## Try it yourself

1. Add a fourth `@tool`, `top_n_by(path: str, column: str, n: int) -> str`, wire it into
   `build_quill`, and watch Quill use it. (Remember: hints on every parameter *and* the return,
   plus an `Args:` section, or the schema won't derive.)
2. Swap `WebSearchTool()` for `WebSearchTool(engine="bing")` and note that the `name` stays
   `web_search` — which is exactly why you keep only one web-search tool in the agent at a time.

## What this lab does NOT do (yet)

No `make_model()` / LiteLLM swap (Module 4) — the `model_id` is still passed directly, and
`quill/config.py` does not exist yet. Local executor only, no Docker/E2B, no `"*"` wildcard
(Module 5). No MCP, no `Tool.from_hub`/`from_langchain`, no `push_to_hub`/`save`, no
`load_tool`, no `ToolCollection` (Module 9) — the tools are *written* in pushable style but are
not pushed. No multi-agent (Module 10). No real image/vision input (Module 11) — Agent Types
are explained in the article, not exercised in multimodal. No `QuillReport` /
`final_answer_checks` (Module 8) — `final_answer` returns a plain answer plus the chart path.
Verified against **smolagents 1.26.0**.
