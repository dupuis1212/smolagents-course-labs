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
"""
from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
