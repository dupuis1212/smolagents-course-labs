"""Quill — a code-first data-analyst agent, built one capability per module.

This is where the canonical ``quill/`` package is born (Module 1 used a throwaway
``quill_intro``). Module 2 ships Quill v0: ``build_quill()`` returns a ``CodeAgent`` that
answers a question about a CSV by writing pandas.

Later modules ADD to this package (never delete): ``config.py``/``make_model`` (M4),
``tools/`` (M3), ``sandbox.py`` (M5), ``report.py``/``QuillReport`` (M8), and so on.
"""
from .agent import DEFAULT_MODEL_ID, QUILL_IMPORTS, build_quill, build_task

__all__ = ["DEFAULT_MODEL_ID", "QUILL_IMPORTS", "build_quill", "build_task"]
