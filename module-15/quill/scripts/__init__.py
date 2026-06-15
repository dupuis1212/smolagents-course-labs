"""Quill's Module 9 ecosystem scripts (NEW in M9).

Small, runnable entry points that demonstrate Quill opening its toolbox to the Hub:

- ``build_sales_db``    — build ``data/sales.db`` (SQLite) from ``data/sales.csv`` so the stdio
                          MCP SQLite server has something to serve. Pure local, no network.
- ``load_hub_tool``     — load ONE tool from the Hub (``load_tool`` / ``Tool.from_hub``,
                          ``trust_remote_code=True``) and attach it to Quill's runtime toolbox.
- ``push_save_chart``   — publish Quill's frozen ``save_chart`` tool to the Hub
                          (``tool.save`` for local inspection, then ``tool.push_to_hub``),
                          proving the M3 "pushable" contract held with no rewrite.

These scripts touch the network/Hub only when run with the right args + ``HF_TOKEN``; importing
them is side-effect-free so the offline tests can assert their shape without any network.
"""

__all__: list[str] = []
