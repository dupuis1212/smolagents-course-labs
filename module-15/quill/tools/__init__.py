"""Quill's reusable, frozen data toolbox (introduced Module 3).

This is where Quill stops re-writing throwaway pandas in every run and gains a real
toolbox. The three data tools are the FROZEN contract (06-FIL-ROUGE-SPEC §2): their
signatures are stable and only ever extended by ADDITION, never renamed or re-signed.

- ``load_dataset(path: str) -> str``      (``@tool``)
- ``profile_dataframe(path: str) -> str`` (``@tool``)
- ``save_chart``                          (``Tool`` subclass with a lazy ``setup()``)

Later modules ADD to this package (e.g. a ``run_sql`` from an MCP server in M9) without
touching these signatures.

Module 15 ADDS the idempotence helpers (``set_run_signature`` / ``clear_run_signature``) — a pure
addition that makes an un-named ``save_chart`` deterministic across re-runs; the three tool
signatures above are untouched.
"""
from .data import (
    RUN_SIGNATURE_ENV,
    clear_run_signature,
    load_dataset,
    profile_dataframe,
    save_chart,
    set_run_signature,
)

__all__ = [
    "load_dataset",
    "profile_dataframe",
    "save_chart",
    "set_run_signature",
    "clear_run_signature",
    "RUN_SIGNATURE_ENV",
]
