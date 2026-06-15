"""Quill — a code-first data-analyst agent, built one capability per module.

This is where the canonical ``quill/`` package is born (Module 1 used a throwaway
``quill_intro``). Module 2 shipped Quill v0; Module 3 gave it a reusable, frozen toolbox
(``load_dataset``, ``profile_dataframe``, ``save_chart``) plus web access. Module 4 added the
FROZEN model layer: ``quill/config.py`` with ``Settings`` + ``make_model()`` — the single,
env-driven place that decides what powers Quill. Module 5 adds the FROZEN sandbox policy:
``quill/sandbox.py`` with ``resolve_executor()`` — the single place that decides WHERE Quill's
generated Python runs (``QUILL_EXECUTOR`` in {local, docker, e2b}) and WHAT it may import
(locked to a least-privilege list, never ``"*"``) — and ``build_quill()`` now calls it. Module
6 adds ``quill/callbacks.py``: step callbacks ``(memory_step, agent)`` that prune stale
DataFrame dumps from memory and log per-step token cost, wired via ``step_callbacks=`` — plus
multi-turn (``reset=False``) via ``run_multi_turn``. Module 7 makes Quill reliable and frugal:
``build_quill`` gains ``planning_interval=`` (periodic ``PlanningStep``) and ``instructions=``
(APPENDED to the system prompt — never replacing it), the data tools' docstrings are sharpened,
and ``quill/bench.py`` measures the baseline-vs-improved step drop. Module 8 turns Quill's
answer into a CONTRACT: ``quill/report.py`` adds the FROZEN ``QuillReport`` / ``Source`` schema
(``to_markdown()`` renders ``[n]`` citations) and two 3-arg ``final_answer_checks`` (chart +
web-source), wired into ``build_quill(final_answer_checks=...)`` so a half-finished report is
rejected and Quill self-corrects.

Later modules ADD to this package (never delete): ``team.py`` (M10), ``retriever.py`` (M12),
and so on.
"""
from .agent import (
    DEFAULT_PLANNING_INTERVAL,
    QUILL_IMPORTS,
    QUILL_INSTRUCTIONS,
    build_quill,
    build_report_task,
    build_task,
    run_multi_turn,
)
from .callbacks import (
    KEEP_LAST,
    MAX_OBS_CHARS,
    PRUNE_MARKER,
    log_step_cost,
    prune_old_observations,
    quill_callbacks,
)
from .config import DEFAULT_MODEL_ID, Settings, make_model
from .report import (
    QUILL_FINAL_ANSWER_CHECKS,
    QuillReport,
    Source,
    check_has_chart,
    check_has_source_for_web_claims,
    quill_final_answer_checks,
)
from .sandbox import (
    QUILL_AUTHORIZED_IMPORTS,
    SUPPORTED_EXECUTORS,
    resolve_executor,
)
from .tools import load_dataset, profile_dataframe, save_chart

__all__ = [
    "DEFAULT_MODEL_ID",
    "DEFAULT_PLANNING_INTERVAL",
    "KEEP_LAST",
    "MAX_OBS_CHARS",
    "PRUNE_MARKER",
    "QUILL_AUTHORIZED_IMPORTS",
    "QUILL_FINAL_ANSWER_CHECKS",
    "QUILL_IMPORTS",
    "QUILL_INSTRUCTIONS",
    "QuillReport",
    "SUPPORTED_EXECUTORS",
    "Settings",
    "Source",
    "build_quill",
    "build_task",
    "build_report_task",
    "run_multi_turn",
    "check_has_chart",
    "check_has_source_for_web_claims",
    "log_step_cost",
    "prune_old_observations",
    "quill_callbacks",
    "quill_final_answer_checks",
    "make_model",
    "resolve_executor",
    "load_dataset",
    "profile_dataframe",
    "save_chart",
]
