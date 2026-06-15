"""Quill — a code-first data-analyst agent, built one capability per module.

This is where the canonical ``quill/`` package is born (Module 1 used a throwaway
``quill_intro``). Module 2 shipped Quill v0; Module 3 gave it a reusable, frozen toolbox
(``load_dataset``, ``profile_dataframe``, ``save_chart``) plus web access. Module 4 adds the
FROZEN model layer: ``quill/config.py`` with ``Settings`` + ``make_model()`` — the single,
env-driven place that decides what powers Quill — and ``build_quill()`` now calls it.

Later modules ADD to this package (never delete): ``sandbox.py`` (M5), ``report.py`` /
``QuillReport`` (M8), and so on.
"""
from .agent import QUILL_IMPORTS, build_quill, build_task
from .config import DEFAULT_MODEL_ID, Settings, make_model
from .tools import load_dataset, profile_dataframe, save_chart

__all__ = [
    "DEFAULT_MODEL_ID",
    "QUILL_IMPORTS",
    "Settings",
    "build_quill",
    "build_task",
    "make_model",
    "load_dataset",
    "profile_dataframe",
    "save_chart",
]
