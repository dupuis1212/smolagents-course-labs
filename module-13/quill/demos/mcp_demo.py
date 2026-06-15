"""End-to-end MCP demo: Quill answers a data question via a stdio MCP SQL tool (Module 9).

The observable result of the lab (06 §6):

    uv run python -m quill.demos.mcp_demo "Which product category grew fastest last quarter?"

Quill connects to a stdio MCP data server (``uvx mcp-server-sqlite --db-path data/sales.db``),
gets its SQL tools as smolagents ``Tool`` objects, and answers — you see the ``CodeAgent`` call an
MCP tool (e.g. ``read_query(...)``) whose result comes back as an ``Observation:``, then produce a
validated ``QuillReport``. The connection is opened/closed by ``run_with_mcp`` (a
``ToolCollection.from_mcp`` context manager): the server subprocess starts for the run and is torn
down on exit.

This makes REAL model calls (needs ``HF_TOKEN``) and launches a REAL subprocess (needs ``uvx`` +
``data/sales.db`` — build it with ``python -m quill.scripts.build_sales_db``). It is an explicit
entry point; importing this module does nothing. Run it from the ``module-09/`` directory so
``data/``, ``outputs/`` and the ``quill`` package resolve.
"""
from __future__ import annotations

DEFAULT_QUESTION = "Which product category grew fastest from the first to the last quarter of 2025?"


def main() -> int:
    """CLI: ``python -m quill.demos.mcp_demo ["<question>"]`` — Quill answers via MCP SQL + pandas."""
    import os
    import sys

    from smolagents import ActionStep

    from quill.agent import build_sql_task, run_with_mcp
    from quill.report import QuillReport
    from quill.tools.mcp import data_mcp_server_params, describe_server_params

    if not os.environ.get("HF_TOKEN"):
        print("This demo makes real model calls — set HF_TOKEN (.env) first.")
        return 1
    if not os.path.exists("data/sales.db"):
        print("data/sales.db is missing. Build it: uv run python -m quill.scripts.build_sales_db")
        return 1

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
    params = data_mcp_server_params()
    print(f"[Quill] Connecting to MCP server -> {describe_server_params(params)}")
    print(f"[Quill] Question: {question}\n")

    # run_with_mcp opens ToolCollection.from_mcp(..., trust_remote_code=True,
    # structured_output=False), wires the MCP tools into Quill via extra_tools, runs, then closes.
    output = run_with_mcp(build_sql_task(question), server_parameters=params)

    print("\n===== REPORT (rendered Markdown) =====")
    print(output.to_markdown() if isinstance(output, QuillReport) else output)
    print("\n[Quill] Done. (The MCP server subprocess was torn down on the with-block exit.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
