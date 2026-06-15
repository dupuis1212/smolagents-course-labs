"""Quill's sandbox policy — the FROZEN contract (06-FIL-ROUGE-SPEC §2, Module 5).

A ``CodeAgent`` writes and runs LLM-generated Python every step. Until now that code ran
in the reader's own process via the ``LocalPythonExecutor`` — an AST allow-list interpreter
that *reduces* the attack surface but, in the library's own words, "is not a security
sandbox." This module gives Quill a single, frozen place to decide TWO things:

1. **Where** the agent's Python runs — the ``executor_type`` (``local`` / ``docker`` / ``e2b``).
2. **What** it is allowed to import — ``additional_authorized_imports``, locked to Quill's
   minimal least-privilege list and NEVER the ``"*"`` wildcard.

The frozen contract every later module reuses:

    resolve_executor() -> tuple[str, list[str]]

It reads ``QUILL_EXECUTOR`` from the environment, validates it against the supported set, and
returns ``(executor_type, additional_authorized_imports)``. That is ALL it does:
``resolve_executor`` does NOT build the agent — ``build_quill`` (in ``quill/agent.py``) calls
this and hands the pair to ``CodeAgent(executor_type=..., additional_authorized_imports=...)``.
Keeping the decision in one function means "change Quill's isolation/import policy" is a
one-file edit, and the import lock can never silently drift to ``"*"`` somewhere downstream.

Approach taken (and NOT taken):
- This is **Approach 1** (snippet-in-sandbox): with ``QUILL_EXECUTOR=docker`` only the Python
  snippets the agent generates go to the container; the model calls and tools stay local, so
  secrets never enter the box. The trade-off: a remote executor + ``managed_agents`` raises
  ``Exception("Managed agents are not yet supported with remote code execution.")`` — so
  **multi-agents (Module 10) needs Approach 2** (the whole agent inside the sandbox), which is
  the capstone (Module 15). We do NOT do Approach 2 here.
- ``modal`` and ``blaxel`` are real ``executor_type`` values in smolagents 1.26.0 but are NOT
  on Quill's required path, so they are not in ``QUILL_EXECUTOR``'s accepted set. Add them by
  AMENDING ``SUPPORTED_EXECUTORS`` (an explicit addition), never by accepting anything.

Verified against smolagents 1.26.0. Note: ``executor_type="wasm"`` was REMOVED in 1.26.0
(PR #2321); it must never appear here.
"""
from __future__ import annotations

import os

# QUILL_EXECUTOR default. "local" is the dev/default executor (instant, free, no isolation —
# "low risk, not zero"); "docker" is the recommended choice the moment inputs/model are not
# fully trusted. We default to "local" so the offline tests and a quick laptop run need no
# Docker daemon; the article and lab push you to "docker" for any untrusted path.
DEFAULT_EXECUTOR = "local"

# The supported set for Quill (a SUBSET of smolagents' Literal["local","blaxel","e2b","modal",
# "docker"]). Quill's required path is local + docker + e2b; modal/blaxel exist but are not
# wired here. NOTE: "wasm" is intentionally absent — it was removed from smolagents in 1.26.0.
SUPPORTED_EXECUTORS = ("local", "docker", "e2b")

# Quill's FROZEN least-privilege import allow-list. This is additive to smolagents'
# BASE_BUILTIN_MODULES (11 always-on stdlib modules: collections, datetime, itertools, math,
# queue, random, re, stat, statistics, time, unicodedata) — i.e. the EFFECTIVE
# agent.authorized_imports is BASE_BUILTIN_MODULES ∪ this list. It is locked to exactly what a
# data-analysis agent needs:
#   - "pandas"        : load / clean / aggregate tabular data
#   - "numpy"         : numerics behind pandas
#   - "matplotlib.*"  : draw figures for save_chart ("numpy.*"-style wildcard authorizes the
#                       package AND its submodules, e.g. matplotlib.pyplot)
#   - "json"          : (de)serialize small structured payloads
#   - "statistics"    : also in BASE_BUILTIN_MODULES; listed for intent/clarity
# It is extended ONLY by explicit addition to this constant — and is NEVER the "*" wildcard
# (which authorizes EVERY import; the library warns "Use this at your own risk!"). Least
# privilege beats convenience: a data snippet has no business importing os, socket, or
# subprocess.
QUILL_AUTHORIZED_IMPORTS: list[str] = [
    "pandas",
    "numpy",
    "matplotlib.*",
    "json",
    "statistics",
]


def resolve_executor() -> tuple[str, list[str]]:
    """Decide Quill's executor and its locked import list — the single source of truth (M5).

    Reads ``QUILL_EXECUTOR`` from the environment (default ``"local"``), validates it against
    ``SUPPORTED_EXECUTORS`` (``{"local", "docker", "e2b"}``), and returns the pair
    ``(executor_type, additional_authorized_imports)`` for ``build_quill`` to pass straight to
    ``CodeAgent``.

    - ``"local"`` -> the in-process ``LocalPythonExecutor`` (AST allow-list; NOT a security
      sandbox — fine for trusted inputs/model, instant and free).
    - ``"docker"`` -> a local Docker container via the Jupyter Kernel Gateway (real isolation;
      needs the ``[docker]`` extra + a running daemon). The recommended choice for untrusted
      inputs/model or a publicly exposed agent.
    - ``"e2b"`` -> an E2B cloud microVM (needs the ``[e2b]`` extra + ``E2B_API_KEY``).

    The import list returned is ALWAYS ``QUILL_AUTHORIZED_IMPORTS`` (the frozen least-privilege
    list), regardless of executor — a fresh ``list`` copy so a caller can extend it locally
    without mutating the frozen constant.

    Returns:
        ``(executor_type, additional_authorized_imports)`` — e.g. ``("local", ["pandas",
        "numpy", "matplotlib.*", "json", "statistics"])``.

    Raises:
        ValueError: if ``QUILL_EXECUTOR`` is set to anything outside ``SUPPORTED_EXECUTORS``.
            We fail loud (never silently fall back to ``local``) so a typo — or a stale
            ``QUILL_EXECUTOR=wasm`` from an old tutorial — never quietly drops the sandbox.
    """
    executor_type = os.environ.get("QUILL_EXECUTOR", DEFAULT_EXECUTOR).strip().lower()
    if executor_type not in SUPPORTED_EXECUTORS:
        raise ValueError(
            f"Unknown QUILL_EXECUTOR {executor_type!r}. "
            f"Supported executors: {', '.join(SUPPORTED_EXECUTORS)}. "
            "(Note: 'wasm' was removed from smolagents in 1.26.0 — it is not a valid value.)"
        )
    # Return a copy so callers can never mutate the frozen constant by accident.
    return executor_type, list(QUILL_AUTHORIZED_IMPORTS)
