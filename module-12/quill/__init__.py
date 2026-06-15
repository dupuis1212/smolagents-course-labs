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
rejected and Quill self-corrects. Module 9 opens Quill's toolbox to the ecosystem:
``quill/tools/mcp.py`` describes a stdio MCP data server (``data_mcp_server_params``),
``build_quill`` gains ``extra_tools=`` (append MCP/Hub/LangChain tools), and ``run_with_mcp``
connects via ``ToolCollection.from_mcp`` (``trust_remote_code=True``, ``structured_output=False``
pinned) for one run. ``quill/scripts/`` loads a tool from the Hub and PUBLISHES the unchanged
``save_chart`` to the Hub (proof its M3 "pushable" contract held).

Module 11 gives Quill EYES. ``build_quill`` gains ``browse=`` (add the OPTIONAL
``vision_browser`` ``CodeAgent`` that drives helium/Chrome and reads screenshots via the
``save_screenshot`` step_callback in ``quill/callbacks.py``); ``review_charts`` re-reads the saved
charts with a VLM via ``run(images=[...])`` and appends the verdict to ``QuillReport.caveats`` (no
new field — the M8 schema is frozen). ``quill/team.py`` adds ``build_vision_browser`` (canonical
name ``vision_browser``). Image INPUT needs only a VLM — NOT the ``[vision]`` extra (that is a
browser).

Later modules ADD to this package (never delete): ``team.py`` (M10), ``retriever.py`` (M12),
and so on.
"""
from .agent import (
    CHART_REVIEW_PROMPT,
    DEFAULT_PLANNING_INTERVAL,
    QUILL_IMPORTS,
    QUILL_INSTRUCTIONS,
    build_quill,
    build_report_task,
    build_sql_task,
    build_task,
    review_charts,
    run_multi_turn,
    run_with_mcp,
)
from .callbacks import (
    KEEP_LAST,
    KEEP_LAST_SCREENSHOTS,
    MAX_OBS_CHARS,
    PRUNE_MARKER,
    SCREENSHOT_SETTLE_SECONDS,
    log_step_cost,
    prune_old_observations,
    prune_old_screenshots,
    quill_callbacks,
    save_screenshot,
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
from .team import (
    VISION_BROWSER_DESCRIPTION,
    VISION_BROWSER_MAX_STEPS,
    VISION_BROWSER_NAME,
    WEB_RESEARCHER_DESCRIPTION,
    WEB_RESEARCHER_MAX_STEPS,
    WEB_RESEARCHER_NAME,
    build_vision_browser,
    build_web_researcher,
)
from .tools import load_dataset, profile_dataframe, save_chart

__all__ = [
    "CHART_REVIEW_PROMPT",
    "DEFAULT_MODEL_ID",
    "DEFAULT_PLANNING_INTERVAL",
    "KEEP_LAST",
    "KEEP_LAST_SCREENSHOTS",
    "MAX_OBS_CHARS",
    "PRUNE_MARKER",
    "SCREENSHOT_SETTLE_SECONDS",
    "QUILL_AUTHORIZED_IMPORTS",
    "QUILL_FINAL_ANSWER_CHECKS",
    "QUILL_IMPORTS",
    "QUILL_INSTRUCTIONS",
    "QuillReport",
    "SUPPORTED_EXECUTORS",
    "Settings",
    "Source",
    "VISION_BROWSER_DESCRIPTION",
    "VISION_BROWSER_MAX_STEPS",
    "VISION_BROWSER_NAME",
    "WEB_RESEARCHER_DESCRIPTION",
    "WEB_RESEARCHER_MAX_STEPS",
    "WEB_RESEARCHER_NAME",
    "build_quill",
    "build_task",
    "build_report_task",
    "build_sql_task",
    "build_vision_browser",
    "build_web_researcher",
    "review_charts",
    "run_multi_turn",
    "run_with_mcp",
    "check_has_chart",
    "check_has_source_for_web_claims",
    "log_step_cost",
    "prune_old_observations",
    "prune_old_screenshots",
    "quill_callbacks",
    "quill_final_answer_checks",
    "save_screenshot",
    "make_model",
    "resolve_executor",
    "load_dataset",
    "profile_dataframe",
    "save_chart",
]
