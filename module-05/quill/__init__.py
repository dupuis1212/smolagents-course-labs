"""Quill — a code-first data-analyst agent, built one capability per module.

This is where the canonical ``quill/`` package is born (Module 1 used a throwaway
``quill_intro``). Module 2 shipped Quill v0; Module 3 gave it a reusable, frozen toolbox
(``load_dataset``, ``profile_dataframe``, ``save_chart``) plus web access. Module 4 added the
FROZEN model layer: ``quill/config.py`` with ``Settings`` + ``make_model()`` — the single,
env-driven place that decides what powers Quill. Module 5 adds the FROZEN sandbox policy:
``quill/sandbox.py`` with ``resolve_executor()`` — the single place that decides WHERE Quill's
generated Python runs (``QUILL_EXECUTOR`` in {local, docker, e2b}) and WHAT it may import
(locked to a least-privilege list, never ``"*"``) — and ``build_quill()`` now calls it.

Later modules ADD to this package (never delete): ``callbacks.py`` (M6), ``report.py`` /
``QuillReport`` (M8), ``team.py`` (M10), and so on.
"""
from .agent import QUILL_IMPORTS, build_quill, build_task
from .config import DEFAULT_MODEL_ID, Settings, make_model
from .sandbox import (
    QUILL_AUTHORIZED_IMPORTS,
    SUPPORTED_EXECUTORS,
    resolve_executor,
)
from .tools import load_dataset, profile_dataframe, save_chart

__all__ = [
    "DEFAULT_MODEL_ID",
    "QUILL_AUTHORIZED_IMPORTS",
    "QUILL_IMPORTS",
    "SUPPORTED_EXECUTORS",
    "Settings",
    "build_quill",
    "build_task",
    "make_model",
    "resolve_executor",
    "load_dataset",
    "profile_dataframe",
    "save_chart",
]
