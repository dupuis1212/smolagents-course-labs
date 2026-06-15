"""Quill's observability layer — OpenTelemetry tracing in one line (NEW, Module 14).

Until now Quill *ran* but you could not *watch* it: when a run went sideways there was no
readable trace to open, only ``agent.replay()`` after the fact (M6, in-process). This module
gives Quill **eyes**. smolagents emits **OpenTelemetry** spans; **OpenInference** (maintained by
Arize) ships the instrumentor that turns those spans on for any backend:

    from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    SmolagentsInstrumentor().instrument()   # BEFORE you build/run the agent

The ONE rule that the whole module hangs on (06-FIL-ROUGE-SPEC §2): call ``instrument()``
**BEFORE** ``build_quill(...)``. Instrumentation patches smolagents' classes; if you patch them
*after* the agent is constructed and a run is under way, the first steps' spans are already gone —
adding telemetry "later" never captures the run that is breaking right now.

The import is from **``openinference.instrumentation.smolagents``** — NOT ``from smolagents import
...``. The instrumentor itself is backend-agnostic: the SAME call feeds Langfuse, Arize Phoenix or
any OTLP collector. We expose two backends, driven by env so a run with no backend is never broken:

    QUILL_TELEMETRY ∈ {none, langfuse, phoenix}   (default: none)

- ``none``    : a clean no-op. ``instrument()`` does nothing and returns ``False`` — the default,
                so importing/using Quill never requires a telemetry backend or extra.
- ``langfuse``: set ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``
                (``https://cloud.langfuse.com`` = EU, ``https://us.cloud.langfuse.com`` = US),
                ``get_client().auth_check()``, then ``SmolagentsInstrumentor().instrument()``.
- ``phoenix`` : start a local collector (``python -m phoenix.server.main serve``), then
                ``from phoenix.otel import register; register()`` and ``instrument()``. Traces land
                at ``http://0.0.0.0:6006/projects/``.

A third one-liner, **MLflow autolog** (``mlflow.smolagents.autolog()``), is mentioned in the
article but not wired here. A trace is a tree of **spans**: a root span (the run), an LLM span per
model call, a tool span per tool call — and for Quill's multi-agent team the ``web_researcher``
span is a CHILD of the manager span (nested spans, the only sane way to read a multi-agent run).

The ``[telemetry]`` extra pulls arize-phoenix + opentelemetry-sdk + opentelemetry-exporter-otlp +
``openinference-instrumentation-smolagents>=0.1.15`` (as of smolagents 1.26.0; telemetry semantic
conventions are still evolving — the span attribute names may move, so re-verify the OpenInference
instrumentation version at build time). For a Langfuse-only setup you can install
``openinference-instrumentation-smolagents`` alone and skip arize-phoenix.

What this module does NOT do (06 §2): no telemetry logic leaks into ``agent.py``. ``build_quill``
is untouched; the entry point (``quill/__main__.py``) calls ``instrument()`` BEFORE it builds Quill.
"""
from __future__ import annotations

import os

# QUILL_TELEMETRY selects the backend. Default "none" = the instrumentation stays OFF, so a run
# with no Langfuse/Phoenix configured is never broken (and the offline tests need no backend).
DEFAULT_BACKEND = "none"
SUPPORTED_BACKENDS = ("none", "langfuse", "phoenix")

# Langfuse host presets (06 §9): the EU and US clouds. Self-hosting points LANGFUSE_HOST at your
# own URL. We do NOT hard-code keys — they come from the environment (.env), never the source.
LANGFUSE_EU_HOST = "https://cloud.langfuse.com"
LANGFUSE_US_HOST = "https://us.cloud.langfuse.com"

__all__ = [
    "DEFAULT_BACKEND",
    "SUPPORTED_BACKENDS",
    "LANGFUSE_EU_HOST",
    "LANGFUSE_US_HOST",
    "resolve_backend",
    "instrument",
]


def resolve_backend(backend: str | None = None) -> str:
    """Resolve the telemetry backend: the explicit arg, else ``QUILL_TELEMETRY``, else ``none``.

    Args:
        backend: an explicit backend name; when ``None`` we read ``QUILL_TELEMETRY`` from the
            environment (default ``"none"``). The value is lower-cased and stripped.

    Returns:
        One of ``{"none", "langfuse", "phoenix"}``.

    Raises:
        ValueError: if the resolved value is not a supported backend — we fail loud so a typo
            (``QUILL_TELEMETRY=langufse``) never silently disables tracing.
    """
    value = (backend if backend is not None else os.environ.get("QUILL_TELEMETRY", DEFAULT_BACKEND))
    value = value.strip().lower()
    if value not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown QUILL_TELEMETRY {value!r}. "
            f"Supported backends: {', '.join(SUPPORTED_BACKENDS)}."
        )
    return value


def instrument(backend: str | None = None) -> bool:
    """Turn on OpenTelemetry tracing for smolagents — the ONE call, BEFORE you build the agent.

    Wraps ``SmolagentsInstrumentor().instrument()`` (OpenInference). Quill instruments only when
    asked: with ``QUILL_TELEMETRY=none`` (the default) this is a clean no-op that returns ``False``,
    so a run without a backend is never broken and the offline tests need no Langfuse/Phoenix.

    The CRITICAL ordering (06 §2): call this **before** ``build_quill(...)`` / ``agent.run(...)``.
    Instrumentation patches smolagents' classes; instrument after the run is under way and the first
    steps' spans are already lost.

    Backend wiring:
    - ``langfuse``: read ``LANGFUSE_*`` env keys, ``get_client().auth_check()`` (a fast credential
      probe), then ``SmolagentsInstrumentor().instrument()``.
    - ``phoenix`` : ``from phoenix.otel import register; register()`` (points the OTLP exporter at
      the local Phoenix collector) then ``SmolagentsInstrumentor().instrument()``.

    Both backends use the SAME instrumentor — only the exporter setup differs. We import the
    instrumentor and the backend SDK lazily (inside this function), so importing ``quill.telemetry``
    never requires the ``[telemetry]`` extra; you pay for the import only when you instrument.

    Args:
        backend: ``"none"`` / ``"langfuse"`` / ``"phoenix"``; when ``None``, read ``QUILL_TELEMETRY``.

    Returns:
        ``True`` if instrumentation was turned on, ``False`` for the ``none`` no-op.

    Raises:
        ValueError: on an unknown backend (via :func:`resolve_backend`).
        RuntimeError: if a chosen backend's SDK is not installed — with a fix-it pip line, instead
            of a bare ``ImportError`` — so the run fails with an actionable message, not a stack of
            import noise.
    """
    resolved = resolve_backend(backend)
    if resolved == "none":
        # The default: do NOTHING. A run with no backend configured is never broken by telemetry,
        # and the offline tests assert exactly this no-op.
        return False

    if resolved == "phoenix":
        _register_phoenix()
    elif resolved == "langfuse":
        _register_langfuse()

    # The ONE backend-agnostic call (OpenInference). It must run BEFORE the agent is built — the
    # caller (quill/__main__.py) does so. Importing here keeps quill.telemetry import-light.
    try:
        from openinference.instrumentation.smolagents import SmolagentsInstrumentor
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise RuntimeError(
            "Telemetry needs the OpenInference instrumentor. Install the extra: "
            "pip install 'smolagents[telemetry]==1.26.0' (or, Langfuse-only, "
            "pip install openinference-instrumentation-smolagents)."
        ) from exc

    SmolagentsInstrumentor().instrument()
    return True


def _register_phoenix() -> None:
    """Point the OTLP exporter at a local Arize Phoenix collector (``phoenix.otel.register``).

    Run the collector first: ``python -m phoenix.server.main serve`` (UI at
    ``http://0.0.0.0:6006/projects/``). ``register()`` wires the OpenTelemetry SDK to it; the
    ``SmolagentsInstrumentor`` then emits spans there. Phoenix runs locally (good for one machine,
    no account), which is also why ``[telemetry]`` pulls arize-phoenix.
    """
    try:
        from phoenix.otel import register
    except ImportError as exc:  # pragma: no cover - exercised only without the extra installed
        raise RuntimeError(
            "Phoenix telemetry needs arize-phoenix. Install: "
            "pip install 'smolagents[telemetry]==1.26.0', then start the collector with "
            "`python -m phoenix.server.main serve`."
        ) from exc

    register()


def _register_langfuse() -> None:
    """Verify Langfuse credentials before instrumenting (``get_client().auth_check()``).

    Langfuse is configured purely by env keys (``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` /
    ``LANGFUSE_HOST``); the OpenTelemetry exporter the OpenInference instrumentor feeds picks them
    up. ``auth_check()`` is a fast probe that fails early with a clear message if the keys are wrong,
    rather than silently dropping every span. Langfuse can be cloud (EU/US) or self-hosted — for a
    Langfuse-only setup you may install ``openinference-instrumentation-smolagents`` alone (no
    arize-phoenix).
    """
    try:
        from langfuse import get_client
    except ImportError as exc:  # pragma: no cover - exercised only without langfuse installed
        raise RuntimeError(
            "Langfuse telemetry needs the langfuse package. Install: pip install langfuse, "
            "and set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST "
            "(https://cloud.langfuse.com = EU, https://us.cloud.langfuse.com = US)."
        ) from exc

    # auth_check() raises if the credentials are missing/invalid — fail early, not span-by-span.
    get_client().auth_check()
