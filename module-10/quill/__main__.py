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
"""
from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
