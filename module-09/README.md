# Module 9 — Tool Interop: MCP, the Hub, and Other Ecosystems

Quill from Module 8 is reliable, but every tool it owns is a function **you** wrote in this repo.
That is a silo. The real leverage of smolagents is not only writing tools — it is **consuming and
sharing** them through standards. This module opens Quill's toolbox to the ecosystem:

- it **connects to a stdio MCP data server** (`uvx mcp-server-sqlite`) that hands it SQL tools
  (e.g. `read_query`) without you writing one;
- it can **load a tool from the Hub** (`load_tool` / `Tool.from_hub`) and attach it at runtime;
- it **publishes its own `save_chart`** back to the Hub (`tool.save` / `tool.push_to_hub`) — with
  **zero rewrites**, because `save_chart` obeyed the "pushable" rules since Module 3.

**MCP (Model Context Protocol)** is the open standard Anthropic introduced in November 2024 — "a
**USB-C port for AI applications**". One protocol replaces N×M ad-hoc tool adapters with N+M. Its
three primitives are **tools** (model-controlled), **resources** (app-controlled), and **prompts**
(user-controlled). smolagents is an **MCP client**; the bridge is **mcpadapt** (the `[mcp]` extra =
`mcpadapt>=0.1.13` + `mcp`), which turns each remote MCP tool into a normal smolagents `Tool`.

## The two paths into MCP (both verified against smolagents 1.26.0)

```python
# Path 1 — ToolCollection.from_mcp: a CONTEXT MANAGER for a one-shot run.
from smolagents import ToolCollection, CodeAgent
from mcp import StdioServerParameters

params = StdioServerParameters(command="uvx", args=["mcp-server-sqlite", "--db-path", "data/sales.db"])
with ToolCollection.from_mcp(params, trust_remote_code=True, structured_output=False) as tc:
    agent = CodeAgent(tools=[*tc.tools], model=model)   # connection closes on `with` exit

# Path 2 — MCPClient: explicit lifecycle, multi-server (pass a LIST), survives many runs.
from smolagents import MCPClient
with MCPClient(params) as tools:           # `tools` is a list[Tool]
    agent = CodeAgent(tools=tools, model=model)
# or manually: client = MCPClient(params); tools = client.get_tools(); ... client.disconnect()
```

> ⚠️ **`Tool.from_mcp` does not exist.** MCP lives ONLY on `ToolCollection.from_mcp` / `MCPClient`.
> Inventing `Tool.from_mcp` is the #1 stale-tutorial error. `Tool.from_*` is for *other*
> ecosystems (`from_hub`, `from_space`, `from_gradio`, `from_langchain`).

## The three transports (`server_parameters` shape selects it)

| Transport | `server_parameters` | Code runs | Status (as of 1.26.0) |
|---|---|---|---|
| **stdio** | `StdioServerParameters(command=..., args=[...], env={...})` | **a subprocess on YOUR machine** | current (local) |
| **streamable-http** | `{"url": ".../mcp", "transport": "streamable-http"}` | remote server | **default** |
| **HTTP+SSE** | `{"url": ".../sse"}` / `"transport": "sse"` | remote server | **deprecated** |

The default flipped from SSE to **streamable-http in smolagents 1.21.0**. Many pre-2026 tutorials
still show SSE by default — they are wrong.

> ⚠️ **Common misconception: "MCP makes my agent safer because it's a standard."** False. MCP
> standardizes the *connection*, not the *trust*. A stdio server executes local code exactly like
> `subprocess`; `trust_remote_code=True` disables a deliberate guard. The standard lowers
> integration cost, **not** attack surface — which *grows* with every third-party server you attach.

## The interop security gate (ties to Module 5)

`trust_remote_code=True` is **required** to run MCP tools (and for `load_tool` / `Tool.from_hub`).
It is not a formality: a **stdio MCP server always executes code on your machine**. Setting it
`True` is you signing off that you trust this server as much as your own code — the M5 threat model
(prompt injection → arbitrary local code → exfiltration). The smolagents docs: "Only use MCP
servers from trusted sources. Malicious servers can execute harmful code on your machine."

`structured_output` (MCP) is pinned `False` in this lab **on purpose**: the default will flip to
`True` in a future release, and it is purely informational — it enriches the prompt with the tool's
output schema, it does **not validate** the output. Validation stays `final_answer_checks` (M8).

## What Module 9 adds to Quill

`quill/tools/mcp.py` (NEW) describes a server (it does NOT connect at import):

```python
data_mcp_server_params(db_path="data/sales.db") -> StdioServerParameters   # the local stdio server
http_server_params(url=..., transport="streamable-http") -> dict           # remote (shown for completeness)
describe_server_params(params) -> str                                       # a pure one-liner for logs
```

`build_quill` is **extended by addition** with `extra_tools: list | None = None` — MCP/Hub/LangChain
tools are **appended** to Quill's frozen local toolbox. `run_with_mcp(task)` opens
`ToolCollection.from_mcp(...)` for one run and wires `extra_tools=[*tc.tools]`.

`quill/scripts/` (NEW): `build_sales_db` (CSV → SQLite), `load_hub_tool` (`load_tool` + attach at
runtime), `push_save_chart` (`save_chart.save` then `push_to_hub`).

## Run it

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit,litellm,openai,docker,mcp]==1.26.0" \
  "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
uvx --version          # `uvx` ships with `uv` — needed to launch the stdio MCP server
cp module-09/.env.example module-09/.env   # then put your HF token in it
```

Run from inside `module-09/` so `data/`, `outputs/` and the `quill` package resolve.

```bash
uv run python -m quill.scripts.build_sales_db                 # build data/sales.db (no token)
uv run python -m quill.demos.mcp_demo "Which product category grew fastest last quarter?"  # MCP run (needs HF_TOKEN)
uv run python -m quill.scripts.push_save_chart               # local save (the pushable proof; no token)
uv run python -m quill.scripts.push_save_chart --push --repo <you>/quill-save-chart  # publish (needs HF_TOKEN with write)
```

The MCP demo prints Quill's trajectory — you see the `CodeAgent` call an MCP tool (e.g.
`read_query(...)`) whose result comes back as an `Observation:`, then produce a `QuillReport`.

## Test it

```bash
uv run pytest module-09/tests/                    # offline (no token, no MCP connection, no Docker)
QUILL_LIVE_TESTS=1 uv run pytest module-09/tests/ # also the real stdio-MCP run (needs uvx + HF_TOKEN + data/sales.db)
```

The offline tests run with **no network, zero tokens, and NO MCP connection** — they assert the
helper *shapes* and import the MCP entry points, but never open a server (connecting is a `live`
test only). They prove:

- `ToolCollection` and `MCPClient` import from smolagents; **`Tool.from_mcp` does NOT exist** (and
  `from_hub`/`from_space`/`from_gradio`/`from_langchain` do); `from_mcp` is a context manager with
  `trust_remote_code` + `structured_output` params;
- `data_mcp_server_params()` builds a stdio `StdioServerParameters` (`uvx mcp-server-sqlite
  --db-path data/sales.db`) **without starting a subprocess**; `http_server_params()` defaults to
  **streamable-http** (not SSE); `describe_server_params` is pure;
- `build_quill(extra_tools=...)` appends ecosystem tools after the frozen toolbox, reachable by
  name, **without widening the frozen import lock**; a runtime `agent.tools[name] = t` works;
  `run_with_mcp` / `build_sql_task` have the right shape;
- the pushable proof: `save_chart.save(dir)` writes `save_chart.py` + `app.py` + `requirements.txt`
  **offline** (a clean save IS proof the M3 contract held); `__init__` takes no args; imports live
  inside `setup`/`forward`;
- `build_sales_db` materializes a queryable SQLite `sales` table from `data/sales.csv`.

Every Module 2–8 test still passes here (the cumulative suite).

## What this module deliberately does NOT do

- **No multi-agents** (`managed_agents`, sub-agent `name`+`description`) — that is Module 10.
- **No pushing an *agent*** (`agent.push_to_hub`, `GradioUI`, `app.py`, `agent.json`) — only a
  *tool* is published here (Module 13).
- **No telemetry** on the MCP calls (Module 14).
- **No writing an MCP server from scratch**, and **no hosted streamable-http** server — shown in
  code, but the lab runs **stdio** locally to stay laptop-only.
- **No RAG / `RetrieverTool`** (Module 12), even though an MCP server could serve a corpus.
- **No paid service; no hard-coded key** (`os.environ["HF_TOKEN"]` + `.env`).

See `lab.md` for the step-by-step. Verified against **smolagents 1.26.0**.
