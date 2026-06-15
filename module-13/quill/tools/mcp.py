"""Quill's MCP integration — open the toolbox to the ecosystem (Module 9).

Until now every tool Quill owns is a Python function YOU wrote in this repo (``data.py``)
plus the smolagents web tools. **MCP (Model Context Protocol)** — the open standard Anthropic
introduced in November 2024, "a USB-C port for AI applications" — lets Quill *consume*
standardized tools served by a separate process or host, without writing the tool. smolagents
is the **MCP client**; the bridge is **mcpadapt** (the ``[mcp]`` extra = ``mcpadapt>=0.1.13`` +
``mcp``), which turns each remote MCP tool into a normal smolagents ``Tool`` object.

This module is the ONE place Quill decides which MCP server to connect to and how. It does NOT
build the agent (that is ``quill/agent.py``) and it does NOT connect at import time — connecting
runs another process (stdio) or hits a network (http), which the tests must not do offline. The
two helpers here only *describe* a server; the connection is opened by ``ToolCollection.from_mcp``
/ ``MCPClient`` at the call site, always inside a context manager.

Two API paths (both verified against smolagents 1.26.0):

- ``ToolCollection.from_mcp(server_parameters, trust_remote_code=False, structured_output=None)``
  — a **context manager**. ``with ToolCollection.from_mcp(...) as tc:`` spins up a background
  thread with an asyncio loop talking to the server, exposes the tools as ``tc.tools``, and tears
  the connection down on ``with`` exit. Use it for a one-shot run. Quill's ``run_with_mcp`` (in
  ``agent.py``) uses this path.
- ``MCPClient(server_parameters, adapter_kwargs=None, structured_output=None)`` — explicit
  lifecycle. ``connect()`` runs in ``__init__``; ``get_tools() -> list[Tool]``; ``disconnect()``
  closes it. As a context manager, ``with MCPClient(params) as tools:`` yields the ``list[Tool]``
  directly. ``server_parameters`` may be a single ``StdioServerParameters``, a ``dict`` (http),
  or a **list** of either to attach several servers at once. Prefer this for a long-running
  service (connect once, reuse across many runs, ``disconnect()`` at shutdown).

Three transports (the ``server_parameters`` shape selects it):

- **stdio** — ``StdioServerParameters(command=..., args=[...], env={...})``. The server runs as a
  **subprocess on YOUR machine**. This is what ``data_mcp_server_params()`` builds (laptop-only,
  no hosted service). It also means the server **executes code locally** — hence the security gate
  below.
- **streamable-http** (the current default, as of smolagents 1.26.0) —
  ``{"url": "http://host:port/mcp", "transport": "streamable-http"}``. The code runs on the remote
  server. ``http_server_params()`` builds this shape (shown for completeness; the lab runs stdio).
- **HTTP+SSE** — ``{"url": ".../sse"}`` or ``"transport": "sse"``. **Deprecated** (the default
  flipped from SSE to streamable-http in smolagents 1.21.0); use it only for legacy servers.

Security (the interop gate — ties back to Module 5's threat model):
``trust_remote_code=True`` is REQUIRED to actually run MCP tools (and for ``load_tool`` /
``Tool.from_hub``). It is not a formality: a **stdio MCP server always executes code on your
machine** (it is a local subprocess, exactly like ``subprocess.run``), so setting it ``True`` is
you signing off that you trust this server as much as your own code. MCP standardizes the
*connection*, never the *trust* — every third-party server you attach widens your attack surface
(prompt-injection → arbitrary local code → exfiltration, the M5 vectors). The smolagents docs put
it plainly: "Only use MCP servers from trusted sources. Malicious servers can execute harmful code
on your machine." A remote *streamable-http* server does not run code on your machine, but still
"proceed with caution".

``Tool.from_mcp`` does **not** exist (a freshness trap from pre-2026 tutorials): MCP lives ONLY on
``ToolCollection.from_mcp`` / ``MCPClient``. ``Tool.from_*`` covers other ecosystems (hub, space,
gradio, langchain) — see ``quill/scripts/load_hub_tool.py``.
"""
from __future__ import annotations

import os
from typing import Any

# These are the public MCP entry points (re-exported so callers import them from one place and the
# offline tests can assert they are importable). ``Tool`` is exported too so a test can prove the
# negative: there is NO ``Tool.from_mcp``.
from mcp import StdioServerParameters
from smolagents import MCPClient, Tool, ToolCollection

__all__ = [
    "StdioServerParameters",
    "MCPClient",
    "ToolCollection",
    "Tool",
    "DEFAULT_SQLITE_DB_PATH",
    "DEFAULT_MCP_SQLITE_PACKAGE",
    "data_mcp_server_params",
    "http_server_params",
    "describe_server_params",
]

# The SQLite database the MCP server serves (built from data/sales.csv by
# quill/scripts/build_sales_db.py). Relative to the working directory (run Quill from module-09/).
DEFAULT_SQLITE_DB_PATH = "data/sales.db"

# The stdio MCP server we launch with `uvx`. `mcp-server-sqlite` is the reference SQLite MCP
# server (it exposes a `read_query`/`list_tables`/`describe_table` capability over a SQLite file).
# `uvx` (ships with `uv`) fetches and runs it in an ephemeral environment — no global install.
# NOTE (freshness, 06 §9): re-verify the exact package name and the DB-path arg the day you run
# this — the MCP server ecosystem moves. In production you would PIN a version (e.g.
# "mcp-server-sqlite==<x.y.z>") and audit the source rather than track "latest".
DEFAULT_MCP_SQLITE_PACKAGE = "mcp-server-sqlite"


def data_mcp_server_params(
    db_path: str = DEFAULT_SQLITE_DB_PATH,
    package: str = DEFAULT_MCP_SQLITE_PACKAGE,
) -> StdioServerParameters:
    """Describe the **stdio** MCP data server Quill connects to (does NOT start it).

    Returns a ``mcp.StdioServerParameters`` that, when handed to
    ``ToolCollection.from_mcp(...)`` / ``MCPClient(...)``, launches ``uvx <package>
    --db-path <db_path>`` as a **local subprocess** and exposes its SQL tools (e.g. a
    ``read_query`` tool Quill calls to run SELECTs against ``sales.db``) as smolagents
    ``Tool`` objects. Building the params is pure and side-effect-free — the subprocess
    only starts at the ``with`` site — so this is safe to call in offline tests.

    ``env`` pins ``UV_PYTHON=3.12`` (so ``uvx`` resolves a known interpreter) and forwards the
    rest of the current environment (``**os.environ``) so the subprocess inherits PATH etc. We do
    NOT inject any secret here; the server only needs the local DB file.

    Args:
        db_path: path to the SQLite file the server serves (relative to the working directory).
            Defaults to ``data/sales.db`` (build it with ``quill/scripts/build_sales_db.py``).
        package: the uvx-runnable MCP server package. Defaults to ``mcp-server-sqlite``. Pin a
            version in production; re-verify the name at build time (06 §9 freshness note).

    Returns:
        A ``StdioServerParameters`` describing the local stdio MCP server. WHERE the code runs:
        a subprocess on YOUR machine — which is why ``trust_remote_code=True`` is required to use
        the tools it serves (the interop security gate, M5/T9.5).
    """
    return StdioServerParameters(
        command="uvx",
        # `--db-path` is mcp-server-sqlite's flag for the SQLite file to serve. (Re-verify the
        # flag name for the package/version you pin — 06 §9.)
        args=[package, "--db-path", db_path],
        # UV_PYTHON pins the interpreter uvx builds the server env with; forward the rest of the
        # environment so the subprocess has a normal PATH. No secret is added here.
        env={"UV_PYTHON": "3.12", **os.environ},
    )


def http_server_params(
    url: str = "http://127.0.0.1:8000/mcp",
    transport: str = "streamable-http",
) -> dict[str, Any]:
    """Describe a **remote** MCP server over HTTP (shown for completeness; the lab runs stdio).

    Returns the ``dict`` form of ``server_parameters`` for an already-hosted MCP server. Unlike
    stdio, a remote server does NOT execute code on your machine (the code runs server-side) — but
    you should still ``proceed with caution`` and only connect to servers you trust.

    Args:
        url: the server's MCP endpoint. For ``streamable-http`` this typically ends in ``/mcp``.
        transport: ``"streamable-http"`` (the current default, as of smolagents 1.26.0) or the
            **deprecated** ``"sse"`` (use only for legacy servers — the default flipped away from
            SSE in smolagents 1.21.0; many older tutorials still show ``sse`` and are wrong).

    Returns:
        A ``dict`` like ``{"url": url, "transport": transport}`` to pass straight to
        ``ToolCollection.from_mcp(...)`` / ``MCPClient(...)``.
    """
    return {"url": url, "transport": transport}


def describe_server_params(server_parameters: Any) -> str:
    """One-line, human-readable description of an MCP ``server_parameters`` (for logs/demos).

    Pure helper — no connection is opened. Reports the transport (stdio for a
    ``StdioServerParameters``, or the ``transport`` key of a dict) and the command/url, so a demo
    can print WHAT it is about to connect to before the ``with`` block runs the subprocess/request.

    Args:
        server_parameters: a ``StdioServerParameters`` (stdio) or a ``dict`` (http).

    Returns:
        e.g. ``"stdio: uvx mcp-server-sqlite --db-path data/sales.db"`` or
        ``"streamable-http: http://127.0.0.1:8000/mcp"``.
    """
    if isinstance(server_parameters, StdioServerParameters):
        argv = " ".join([server_parameters.command, *(server_parameters.args or [])])
        return f"stdio: {argv}"
    if isinstance(server_parameters, dict):
        transport = server_parameters.get("transport", "streamable-http")
        return f"{transport}: {server_parameters.get('url', '?')}"
    return f"unknown server_parameters: {server_parameters!r}"
