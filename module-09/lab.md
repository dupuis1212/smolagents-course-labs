# Lab 9 — Plug Quill into the ecosystem: MCP, the Hub, and tool interop

**Goal:** stop Quill's tools being a silo. Connect Quill to a **stdio MCP data server** for SQL/file
access (a tool you did not write), load **one tool from the Hub**, and **publish** Quill's own
`save_chart` tool back to the Hub — with no rewrite, because `save_chart` obeyed the "pushable" rules
since Module 3.

**You'll see:** the MCP entry points imported the right way (`ToolCollection.from_mcp` / `MCPClient`,
and that **`Tool.from_mcp` does not exist**); a stdio `server_parameters` built without starting a
server; Quill's toolbox gain an MCP/Hub tool via `extra_tools=`; `save_chart.save(...)` write a
Space-ready `save_chart.py` + `app.py` + `requirements.txt` offline; and (live) Quill answer a data
question by calling a real MCP `read_query` tool and returning a validated `QuillReport`.

**Observable result:**

```bash
uv run python -m quill.scripts.build_sales_db
uv run python -m quill.demos.mcp_demo "Which product category grew fastest last quarter?"
```

```text
[Quill] Connecting to MCP server -> stdio: uvx mcp-server-sqlite --db-path data/sales.db
[Quill] Question: Which product category grew fastest last quarter?
...
 ─ Executing: rows = read_query("SELECT category, SUM(net_rev) ... GROUP BY category")
Observation: [("Team", 184213.0), ("Pro", 151002.0), ("Free", 0.0)]
...                       # ← Quill draws a chart with matplotlib, calls save_chart, builds a report
===== REPORT (rendered Markdown) =====
# Which product category grew fastest last quarter?

## Findings
- Team grew fastest in the final quarter, leading net revenue.

## Charts
- `outputs/category_growth.png`

[Quill] Done. (The MCP server subprocess was torn down on the with-block exit.)
```

> The exact trajectory varies — LLMs are non-deterministic. What is guaranteed: Quill *calls an MCP
> tool* and returns a *validated `QuillReport`*.

And publishing the tool:

```bash
uv run python -m quill.scripts.push_save_chart --push --repo <you>/quill-save-chart   # needs HF_TOKEN (write)
# -> Published save_chart to https://huggingface.co/spaces/<you>/quill-save-chart
```

Verified against **smolagents 1.26.0**.

---

## Step 1 — Setup

Start from the cumulative Module 8 state (this `module-09/` already carries it forward). Install the
**MCP extra** on top of what the earlier modules need:

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit,litellm,openai,docker,mcp]==1.26.0" \
  "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
```

`[mcp]` brings `mcpadapt>=0.1.13` + `mcp` — the bridge that turns MCP tools into smolagents `Tool`
objects. Check `uvx` (it ships with `uv`; we use it to launch the stdio MCP SQLite server):

```bash
uvx --version
```

`HF_TOKEN` is never committed — copy `.env.example` to `.env` and put your token there (it powers the
model calls and, for `--push`, the Hub upload — that one needs **write** access).

---

## Step 2 — Plug in a stdio MCP data server

A stdio MCP server runs as a **subprocess on your machine** and talks over stdin/stdout. We use
`mcp-server-sqlite` (the reference SQLite MCP server), launched by `uvx`, serving a SQLite file. In
`quill/tools/mcp.py`, `data_mcp_server_params()` *describes* it (it does not start it):

```python
def data_mcp_server_params(db_path="data/sales.db", package="mcp-server-sqlite"):
    return StdioServerParameters(
        command="uvx",
        args=[package, "--db-path", db_path],
        env={"UV_PYTHON": "3.12", **os.environ},   # pin the interpreter, forward PATH; no secret
    )
```

Build the DB the server serves (idempotent; pure local pandas + sqlite3):

```bash
uv run python -m quill.scripts.build_sales_db          # data/sales.csv -> data/sales.db (table "sales")
```

> **Freshness (06 §9):** the MCP-server ecosystem moves — re-verify the exact package name and the
> `--db-path` flag the day you run this. In production you would **pin a version** and **audit the
> source**, not track "latest".
>
> **Transports.** stdio is one of three. The others use a dict: streamable-http (the current
> default) `{"url": ".../mcp", "transport": "streamable-http"}` and the **deprecated** SSE
> `{"url": ".../sse"}`. `http_server_params()` builds the streamable-http shape — shown for
> completeness; this lab stays stdio to be laptop-only.

---

## Step 3 — Build Quill with the MCP tools

`run_with_mcp(task)` (in `quill/agent.py`) connects for the duration of one run:

```python
with ToolCollection.from_mcp(
    server_parameters,
    trust_remote_code=True,        # the security gate — a stdio server runs local code (see Step 5 / M5)
    structured_output=False,       # pin it: the default will flip to True in a future release
) as tool_collection:
    with build_quill(model=model, extra_tools=[*tool_collection.tools]) as agent:
        return agent.run(task)
```

`ToolCollection.from_mcp` **must** be used as a context manager: it starts a background asyncio
thread for the server and closes it on `with` exit (otherwise you leak a subprocess per request).
`extra_tools=` is the new, addition-only knob on `build_quill` — MCP tools are **appended** to
Quill's frozen local toolbox (the local tools and their order are untouched).

For a long-running service, prefer `MCPClient` (connect once, reuse across many runs, `disconnect()`
at shutdown), and pass a **list** of `server_parameters` to attach several servers at once:

```python
with MCPClient([sqlite_params, other_params]) as tools:   # `tools` is a list[Tool]
    agent = build_quill(extra_tools=tools)
```

---

## Step 4 — Load ONE tool from the Hub

A tool published to the Hub is a **Space repo** carrying its source. `load_tool` (or
`Tool.from_hub`) downloads and reconstructs it as a smolagents `Tool`. Because the Hub code runs
**locally**, `trust_remote_code=True` is required — the same interop gate as MCP. See
`quill/scripts/load_hub_tool.py`:

```python
from smolagents import load_tool
tool = load_tool("m-ric/text-to-image", trust_remote_code=True)   # runs the Hub code locally
agent.tools[tool.name] = tool                                     # attach at runtime (toolbox is a dict)
```

> **`Tool.from_mcp` does not exist.** MCP lives on `ToolCollection.from_mcp` / `MCPClient`.
> `Tool.from_*` is for *other* ecosystems: `from_hub` / `from_space` (a deployed Gradio Space) /
> `from_gradio` (a `gradio_tools` object) / `from_langchain` (delegates to LangChain `run()`).
> Pick by **source**: MCP for a deployed, multi-language, shared standard; `from_langchain` for code
> you already have in LangChain; `from_hub` for a published, reusable Hub tool.
>
> **Freshness (06 §9):** re-verify the Hub repo exists the day you run this — Hub repos move.

---

## Step 5 — Publish `save_chart` to the Hub

`save_chart` (frozen in Module 3) was written **pushable** from day one. The three rules
(`quill/tools/data.py` already obeys them):

1. Methods are self-contained (use only args / class attributes).
2. **Every import is INSIDE a method** — `save_chart.setup`/`forward` import `matplotlib` inside
   themselves, never at the top of `data.py`.
3. `__init__` takes **no argument other than `self`** (init args are not serializable to the Hub) —
   `save_chart` does not override `__init__` at all.

These rules exist because the Hub **re-executes the tool's code in a fresh Space**, with none of your
original environment. First inspect locally (no network), then publish (`quill/scripts/push_save_chart.py`):

```python
save_chart().save("build/save_chart_tool", tool_file_name="save_chart")   # save_chart.py + app.py + requirements.txt
save_chart().push_to_hub("<you>/quill-save-chart", token=os.environ["HF_TOKEN"])   # a Space repo with a Gradio UI
```

```bash
uv run python -m quill.scripts.push_save_chart                 # local save only (the pushable proof)
uv run python -m quill.scripts.push_save_chart --push --repo <you>/quill-save-chart   # publish (HF_TOKEN write)
```

A clean `save()` IS the proof the rules held — a top-level import or an `__init__(self, model_name)`
would make it raise. The token is read with `os.environ["HF_TOKEN"]`, never hard-coded.

---

## Step 6 — End-to-end demo

```bash
uv run python -m quill.demos.mcp_demo "Which product category grew fastest last quarter?"
```

Quill connects to the stdio MCP server, gets its SQL tools, runs `read_query(...)` against
`sales.db`, draws a chart with its **local** matplotlib (the MCP tool and the local tools coexist),
saves it with `save_chart`, and returns a validated `QuillReport`. Read the trajectory from the run
output (or `agent.replay()` / `agent.memory.steps` — never the removed `agent.logs`).

> **What runs where:** the MCP `read_query` tool runs **server-side** (in the `uvx` subprocess) —
> *outside* Quill's sandbox. The Python Quill *writes* to call it (and its pandas/matplotlib) runs
> *inside* the sandbox, under the frozen least-privilege import lock. Adding MCP/Hub tools never
> widens that lock.

---

## Step 7 — Tests

```bash
uv run pytest module-09/tests/                    # offline: no token, no MCP connection, no Docker
QUILL_LIVE_TESTS=1 uv run pytest module-09/tests/ # also the real stdio-MCP run (uvx + HF_TOKEN + sales.db)
```

**Live budget:** ~5 real LLM runs total (carried-forward M6/M7/M8 + the new M9 MCP run = 1). The M9
live test also needs `uvx` and a built `data/sales.db`; it **skips cleanly** if any is missing.

The offline tests open **no MCP connection** — they assert the helper shapes and import the entry
points only (connecting is a `live` test). They prove `Tool.from_mcp` does **not** exist,
`data_mcp_server_params()` builds a stdio params object without a subprocess, `http_server_params()`
defaults to streamable-http (not SSE), `build_quill(extra_tools=...)` appends ecosystem tools without
widening the import lock, and `save_chart.save(...)` writes a Space-ready tool offline.

---

## Try it yourself (not graded)

1. **Multi-server.** Pass a **list** to `MCPClient([sqlite_params, other_params])` (a second stdio
   server, e.g. a filesystem MCP server) and watch Quill pick the right tool per question.
2. **A whole collection.** Load every tool in a Hub **collection** with
   `ToolCollection.from_hub(collection_slug=..., trust_remote_code=True)` (Spaces only; tools load
   lazily) and see which Quill reaches for unprompted.

---

## What this lab does NOT do

No multi-agents (M10), no pushing an *agent* / `GradioUI` / `app.py` (M13), no telemetry on MCP calls
(M14), no writing an MCP server from scratch, no hosted streamable-http (stdio only here), no RAG
(M12), no paid service, no hard-coded key.
