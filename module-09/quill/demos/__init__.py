"""Quill's Module 9 demos (NEW in M9).

- ``mcp_demo`` — the end-to-end run: Quill answers a data question by combining a stdio MCP SQL
  tool (``read_query`` against ``data/sales.db``) with its own local pandas/matplotlib, and
  returns a validated ``QuillReport``. Makes real model + MCP calls (needs ``HF_TOKEN`` and a
  working ``uvx``), so it is an explicit entry point, never run at import time.
"""

__all__: list[str] = []
