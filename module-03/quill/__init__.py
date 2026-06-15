"""Quill — a code-first data-analyst agent, built one capability per module.

This is where the canonical ``quill/`` package is born (Module 1 used a throwaway
``quill_intro``). Module 2 shipped Quill v0; Module 3 gives it a reusable, frozen toolbox:
``load_dataset``, ``profile_dataframe``, ``save_chart`` (in ``quill/tools/data.py``) plus
web access, all wired into ``build_quill()``.

Later modules ADD to this package (never delete): ``config.py``/``make_model`` (M4),
``sandbox.py`` (M5), ``report.py``/``QuillReport`` (M8), and so on.
"""
from .agent import DEFAULT_MODEL_ID, QUILL_IMPORTS, build_quill, build_task
from .tools import load_dataset, profile_dataframe, save_chart

__all__ = [
    "DEFAULT_MODEL_ID",
    "QUILL_IMPORTS",
    "build_quill",
    "build_task",
    "load_dataset",
    "profile_dataframe",
    "save_chart",
]
