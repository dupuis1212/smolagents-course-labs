"""Quill's output contract — ``QuillReport`` + ``Source`` + ``final_answer_checks`` (FROZEN M8).

This is the module where Quill stops returning free-form text and starts returning a *contract*.
Before Module 8, ``final_answer(...)`` could return a paragraph, a half-built dict, or the words
"growth is strong" with no number and no chart — three runs of the same question, three shapes,
nothing you could audit, compare, cite (Module 12) or score (Module 14). This file freezes that
shape and adds the runtime validation that refuses a half-finished report.

Two ideas live here, and the article hammers that they are NOT the same thing:

1. **Shape** — the ``QuillReport`` dataclass. It guarantees a *form*: a question, findings, chart
   paths, sources, caveats. A perfectly-typed ``QuillReport`` can still be empty
   (``chart_paths=[]``, ``sources=[]``). Shape is not validity.
2. **Business rules** — the ``final_answer_checks``. They guarantee *content*: an analysis without
   a saved chart is not a report; a web-backed claim without a source is not auditable. These are
   the rules ``response_format`` / ``use_structured_outputs_internally`` can NEVER enforce — those
   only constrain the JSON shape of a model call, never the meaning of the answer.

**The frozen schema (06-FIL-ROUGE-SPEC §2 — do NOT add a field without updating that spec first):**

    QuillReport{ question: str, findings: list[str], chart_paths: list[str],
                 sources: list[Source], caveats: list[str] }
    Source{ url: str, title: str }

``to_markdown()`` renders the findings with numbered ``[n]`` citations into ``sources`` — the
contract M12 (RAG citations) and M14 (eval/judge) both build on, which is why it is frozen now.

**The checks are 3-arg ``(final_answer, memory, agent) -> bool``** — the call site in smolagents
1.26.0 is ``agents.py``'s ``_validate_final_answer``:

    assert check_function(final_answer, self.memory, agent=self)

So a check that returns ``False`` (or raises) makes that ``assert`` fail; smolagents wraps it in an
``AgentError`` which is **caught and stored in ``ActionStep.error``** — the run does NOT crash, it
loops and the model self-corrects on the next step (the *exact same* mechanism that recovers from a
``KeyError`` in the agent's code). The guided-tour 2-arg example only "works" because the 3rd arg is
passed as a keyword; the robust form is the 3-arg one written here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# The web tools Quill can call (Module 3). When the agent's generated code calls one of these, the
# resulting report makes web-backed claims, so it must cite at least one source. These names are the
# canonical smolagents tool names (`WebSearchTool().name`, `VisitWebpageTool().name`) — kept here as
# the single source of the "did Quill go to the web?" heuristic used by the source check.
WEB_TOOL_NAMES = ("web_search", "visit_webpage")


@dataclass
class Source:
    """A single cited source: a URL and a human-readable title.

    FROZEN (06-FIL-ROUGE-SPEC §2). M12 populates these from the RAG retriever / web tools; M14
    counts them when scoring a report. Both fields are required — a citation with no URL is not
    auditable, and a URL with no title is not readable.
    """

    url: str
    title: str


@dataclass
class QuillReport:
    """Quill's validated output contract (FROZEN M8 — 06-FIL-ROUGE-SPEC §2).

    Returned by ``final_answer(report)`` and validated by the ``final_answer_checks`` below. The
    five fields are the whole contract; do NOT add a sixth without first updating 06 §2 (M12 and
    M14 depend on this exact shape).

    Fields:
        question: the analysis question, restated so the report is self-contained.
        findings: the answer as a list of short, standalone claims (each may carry a ``[n]``
            citation that ``to_markdown`` resolves into ``sources``).
        chart_paths: paths of charts ``save_chart`` produced — the evidence behind the findings.
            ``check_has_chart`` refuses a report with an empty list.
        sources: the cited ``Source`` objects. ``check_has_source_for_web_claims`` refuses an
            empty list WHEN the run actually went to the web.
        caveats: honest limitations (small sample, stale data, an assumption made). Preferred over
            a crash when a constraint cannot be met — degrade with a caveat, do not fail silently.
    """

    question: str
    findings: list[str] = field(default_factory=list)
    chart_paths: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render the report as Markdown with numbered ``[n]`` citations into ``sources``.

        Each finding keeps any ``[n]`` markers it already carries; a "Sources" section lists each
        source as ``[n] [title](url)`` so the numbers line up. Charts are listed under their own
        heading, caveats under theirs. Empty sections are omitted so a minimal report still renders
        cleanly. This Markdown is the artefact M12 cites against and M14 scores.

        Returns:
            A Markdown string. For a report with one source, the body contains the marker ``[1]``
            and the Sources section contains ``[1] [<title>](<url>)``.
        """
        lines: list[str] = [f"# {self.question}", ""]

        if self.findings:
            lines.append("## Findings")
            for finding in self.findings:
                lines.append(f"- {finding}")
            lines.append("")

        if self.chart_paths:
            lines.append("## Charts")
            for path in self.chart_paths:
                lines.append(f"- `{path}`")
            lines.append("")

        if self.caveats:
            lines.append("## Caveats")
            for caveat in self.caveats:
                lines.append(f"- {caveat}")
            lines.append("")

        if self.sources:
            lines.append("## Sources")
            # 1-based numbering so the [n] markers in `findings` resolve here.
            for n, source in enumerate(self.sources, start=1):
                lines.append(f"[{n}] [{source.title}]({source.url})")
            lines.append("")

        # Strip the single trailing blank line for a tidy artefact.
        return "\n".join(lines).rstrip() + "\n"


def _used_a_web_tool(memory) -> bool:
    """Did this run actually go to the web? (the heuristic behind the source check)

    Quill is a ``CodeAgent``: it calls ``web_search`` / ``visit_webpage`` from inside the Python it
    writes, so the evidence is in each ``ActionStep``'s code — both ``code_action`` (the parsed
    snippet) and the ``python_interpreter`` ``ToolCall.arguments`` (which holds that same code).
    We scan those strings for a web-tool name. Reading ``memory.steps`` is the supported way to
    inspect a run (NEVER the removed ``agent.logs``).

    Timing note: the validator runs BEFORE smolagents appends the CURRENT step to memory, so a web
    call made in the *same* step as ``final_answer`` is not yet visible here. In practice Quill
    searches the web in earlier exploration steps and answers later, so the evidence is already in
    memory; a web call in the final step would only be caught on a subsequent attempt.

    Args:
        memory: the agent's ``AgentMemory`` (``memory.steps`` is the trajectory).

    Returns:
        ``True`` if any step's code references a web tool, else ``False``.
    """
    steps = getattr(memory, "steps", None) or []
    for step in steps:
        code = getattr(step, "code_action", None) or ""
        for call in getattr(step, "tool_calls", None) or []:
            args = getattr(call, "arguments", None)
            if isinstance(args, str):
                code += "\n" + args
        if any(name in code for name in WEB_TOOL_NAMES):
            return True
    return False


def check_has_chart(final_answer, memory, agent) -> bool:
    """3-arg ``final_answer_check`` (06 §2): a report must include at least one saved chart.

    Quill is a *data* analyst — an analysis without a chart is not a report. Returns ``False`` when
    the final answer is not a ``QuillReport`` (wrong shape) or its ``chart_paths`` is empty. A
    ``False`` does NOT crash the run: smolagents stores the resulting ``AgentError`` in
    ``ActionStep.error`` and loops, so Quill sees *why* it was rejected and draws + saves a chart on
    the next step. The message below is what the model reads to self-correct, so it is actionable.

    Args:
        final_answer: the value Quill passed to ``final_answer`` (expected: a ``QuillReport``).
        memory: the agent's ``AgentMemory`` (unused here; part of the frozen 3-arg signature).
        agent: the running agent (unused here; part of the frozen 3-arg signature).

    Returns:
        ``True`` if ``final_answer`` is a ``QuillReport`` with a non-empty ``chart_paths``.

    Raises:
        ValueError: with an actionable message when the rule is not met. (Either raising or
            returning ``False`` rejects the answer — we raise so the model sees the reason.)
    """
    if not isinstance(final_answer, QuillReport):
        raise ValueError(
            "Final answer rejected: the answer must be a QuillReport "
            "(import it from quill.report and pass it to final_answer), not "
            f"a {type(final_answer).__name__}."
        )
    if not final_answer.chart_paths:
        raise ValueError(
            "Final answer rejected: a report must include at least one saved chart. "
            "Draw a matplotlib chart, call save_chart to save it, add the returned path to "
            "QuillReport.chart_paths, then call final_answer again."
        )
    return True


def check_has_source_for_web_claims(final_answer, memory, agent) -> bool:
    """3-arg ``final_answer_check`` (06 §2): a web-backed report must cite at least one source.

    The rule is conditional: if Quill went to the web (called ``web_search`` / ``visit_webpage`` in
    its code — see ``_used_a_web_tool``) then the report MUST carry at least one ``Source``; a
    web-backed claim with no URL is not auditable. A purely local analysis (no web tool used) needs
    no sources and passes. Like every check, a ``False``/raise loops the agent instead of crashing —
    Quill re-reads the page it fetched and records the URL + title before answering again.

    Args:
        final_answer: the value Quill passed to ``final_answer`` (expected: a ``QuillReport``).
        memory: the agent's ``AgentMemory``; ``memory.steps`` reveals whether a web tool was used.
        agent: the running agent (its ``agent.memory`` mirrors ``memory``; kept for the signature).

    Returns:
        ``True`` if no web tool was used, OR a ``QuillReport`` with a non-empty ``sources`` list.

    Raises:
        ValueError: with an actionable message when web claims lack a source.
    """
    if not isinstance(final_answer, QuillReport):
        raise ValueError(
            "Final answer rejected: the answer must be a QuillReport, not "
            f"a {type(final_answer).__name__}."
        )
    if _used_a_web_tool(memory) and not final_answer.sources:
        raise ValueError(
            "Final answer rejected: this analysis used the web (web_search/visit_webpage) but "
            "QuillReport.sources is empty. Add a Source(url=..., title=...) for every web-backed "
            "claim, cite it as [n] in the matching finding, then call final_answer again."
        )
    return True


# The list build_quill wires into final_answer_checks= (M8). Ordered cheapest-first: the chart
# check is unconditional, the source check is conditional on web usage.
QUILL_FINAL_ANSWER_CHECKS = [check_has_chart, check_has_source_for_web_claims]


def quill_final_answer_checks() -> list:
    """Return Quill's ``final_answer_checks`` (a fresh list copy, so a caller cannot mutate ours).

    ``build_quill`` calls this and passes the result to ``CodeAgent(final_answer_checks=...)``.
    Each entry is a 3-arg ``(final_answer, memory, agent) -> bool`` validator.
    """
    return list(QUILL_FINAL_ANSWER_CHECKS)


__all__ = [
    "Source",
    "QuillReport",
    "WEB_TOOL_NAMES",
    "check_has_chart",
    "check_has_source_for_web_claims",
    "QUILL_FINAL_ANSWER_CHECKS",
    "quill_final_answer_checks",
]
