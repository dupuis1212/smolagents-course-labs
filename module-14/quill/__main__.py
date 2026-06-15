"""CLI entry point so ``uv run python -m quill "<question>" --data <csv>`` works.

As of Module 8 the canonical command is
``python -m quill "Which category grew fastest, and is that consistent with the public trend?" --data data/sales.csv``
(the question is positional, the CSV comes from ``--data``; default ``data/sales.csv``). It asks
Quill for a validated ``QuillReport``, prints the backend/model, the rendered report Markdown (with
``[n]`` citations) and the run's token cost. ``python -m quill`` delegates to that same cost-aware
entry point in ``quill/run.py``. The full ReAct trajectory printer still lives at
``python -m quill.agent`` (which also returns a validated report) if you want the step-by-step
replay and the rejected-final-answer digest instead.

As of Module 10, the Quill ``build_quill()`` builds is a **manager** ``CodeAgent`` over a
``web_researcher`` sub-agent (``managed_agents=[...]``), so a question that needs external context
(e.g. ``python -m quill "Is our Q3 churn (in data/customers.csv) high vs the SaaS industry
average?" --data data/customers.csv``) shows the manager DELEGATING to ``web_researcher`` in the
trajectory before it returns the cited ``QuillReport``. The CLI itself is unchanged — the team is
wired inside ``build_quill``.

As of Module 11, two OPTIONAL flags give Quill eyes (both OFF by default, so this command is
unchanged for everyone who does not pass them):

- ``--review`` — after the run, a VLM RE-READS each saved chart via ``run(images=[...])`` and the
  verdict is appended to ``QuillReport.caveats`` (e.g.
  ``python -m quill "Analyze data/sales.csv and chart monthly revenue, then check the chart
  yourself." --review``). Image input needs only a VLM (point ``QUILL_MODEL_ID`` at one) — NOT the
  ``[vision]`` extra.
- ``--browse`` — add the OPTIONAL ``vision_browser`` sub-agent (helium/Chrome) for JS/chart-heavy
  pages a text scraper cannot read. Needs a local Chrome + ``pip install 'smolagents[vision]'`` + a
  VLM. It stays ``executor_type="local"`` (Approach 2 — the whole team in a sandbox — is Module 15).

As of Module 13, Quill becomes a web app. ``python -m quill --ui`` launches the ``GradioUI``
(``quill/ui.py``) — a chat where anyone uploads a CSV and asks a question — instead of running the
one-shot report path. ``--ui --share`` also opens a temporary public ``*.gradio.live`` tunnel. The
one-shot ``python -m quill "<question>"`` path is UNCHANGED for everyone who does not pass ``--ui``
(it still delegates to ``quill/run.py``). The dedicated launcher ``python -m quill.ui`` is
equivalent; ``python -m quill.publish`` ships Quill to the Hub as a Space.

As of Module 14, Quill can be OBSERVED. Before ANY agent is built, this entry point calls
``quill.telemetry.instrument()`` — which turns on OpenTelemetry tracing only when
``QUILL_TELEMETRY`` is ``langfuse`` or ``phoenix`` (default ``none`` = a clean no-op, so a run with
no backend is never broken). The ordering IS the point (06 §2): ``instrument()`` MUST run BEFORE
``build_quill(...)`` or the first steps' spans are lost. With a backend configured, a run emits a
trace where the manager span nests the ``web_researcher`` span. The eval harness
(``python -m quill.eval.run_evals --out eval/results/run-<name>.json``) scores Quill over the golden
set behind a regression gate — see ``quill/eval/``.
"""
import sys

from .run import main
from .telemetry import instrument


def _entrypoint() -> int:
    """Dispatch: ``--ui`` launches the Gradio web app (M13); otherwise the one-shot CLI (unchanged).

    M14: telemetry is turned on FIRST (06 §2 — instrument BEFORE building the agent). With
    ``QUILL_TELEMETRY=none`` (the default) this is a no-op, so the CLI behaves exactly as in M13
    unless you opt into a Langfuse/Phoenix backend.
    """
    # M14: instrument() reads QUILL_TELEMETRY and, for langfuse/phoenix, calls
    # SmolagentsInstrumentor().instrument() — which MUST precede build_quill so no span is missed.
    instrument()

    argv = sys.argv[1:]
    if "--ui" in argv:
        # M13: launch the web app. Imported lazily so the one-shot path never needs the [gradio]
        # extra. --share opens a temporary public tunnel (default: local only).
        from .ui import launch_ui

        launch_ui(share="--share" in argv)
        return 0
    return main()


if __name__ == "__main__":
    raise SystemExit(_entrypoint())
