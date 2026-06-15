"""The LLM-as-judge — scoring a ``QuillReport`` without it grading itself (NEW, Module 14).

Some outcomes have no exact answer ("is this report any good?"). For those we use an **LLM-as-judge**:
a SEPARATE model call that scores Quill's output against an explicit **rubric**. The four rules the
article teaches as non-negotiable, all enforced here:

1. **Explicit rubric** — numeric criteria, not "rate this 1-10". The judge scores three axes, 0-2
   each (max 6): does the report cover the golden item's ``expected_points``? are the findings
   backed (chart / numbers / sources)? are sources cited ``[n]`` when ``min_sources`` asks for them?
2. **Evidence-before-score** — the judge writes its ``rationale`` FIRST, then the ``scores``. A
   score posed before its justification is less reliable, so the JSON is ordered rationale → scores.
3. **Structured output** — the judge returns parsable JSON ``{rationale, scores{...}, verdict}``,
   not free prose (the same spirit as M8's ``response_format`` — we reuse the IDEA, not the schema).
4. **Calibration** — compare the judge's scores to YOUR human labels on ~10-20 examples and aim for
   a decent correlation (~0.80 as of the research, an order of magnitude, not an absolute gate). A
   judge you never calibrated measures its own bias. See :func:`calibration_correlation` below — it
   is documented + optional, not run on every eval.

The two hard pitfalls the article hammers:
- **Never let an agent grade itself.** The judge is a SEPARATE ``make_model`` call (06 §2: one model
  factory for Quill AND the judge), ideally a different/stronger ``model_id`` — set
  ``QUILL_JUDGE_MODEL_ID`` to swap it. Quill grading Quill validates its own mistakes.
- The judge **also costs tokens** every eval. A golden set of N items = N Quill runs + N judge
  calls — count it (the eval harness does).

The judge takes the FROZEN ``QuillReport`` (M8) as input and never modifies it. The
``QuillReport`` schema is the same one M8 validates at run time (``final_answer_checks``) — here it
is scored a posteriori. One schema, used twice.
"""
from __future__ import annotations

import json
import re

from ..report import QuillReport

# The rubric's per-axis max (0-2) and the total (3 axes -> 6). report_quality is reported out of 6.
JUDGE_RUBRIC_AXIS_MAX = 2
JUDGE_RUBRIC_AXES = ("coverage", "grounding", "citations")
JUDGE_RUBRIC_MAX = JUDGE_RUBRIC_AXIS_MAX * len(JUDGE_RUBRIC_AXES)  # 6

# task_success is an OUTCOME: did the report cover the expected_points? The coverage axis drives it;
# we call the run a success when coverage clears this fraction of its max (2/2 or close).
TASK_SUCCESS_COVERAGE_THRESHOLD = 0.5  # coverage >= 1/2 of axis max counts the outcome as a success


def build_judge_prompt(report: QuillReport, item: dict) -> str:
    """Build the judge's rubric prompt — evidence-before-score, structured JSON out.

    Names the golden item's ``expected_points`` and ``min_sources``, hands the judge the report's
    rendered Markdown (the artefact, via ``to_markdown``), and DEMANDS the JSON be ordered rationale
    FIRST then scores (evidence-before-score). The judge must NOT re-run or fix the report — only
    score what it is given.

    Args:
        report: the ``QuillReport`` to score (FROZEN M8 schema — read-only here).
        item: a golden-set entry ``{id, question, dataset, expected_points[], min_sources}``.

    Returns:
        The prompt string for a single judge ``model.generate`` call.
    """
    expected = "\n".join(f"  - {p}" for p in item.get("expected_points", [])) or "  (none)"
    min_sources = item.get("min_sources", 0)
    return (
        "You are a STRICT, impartial evaluator of a data-analysis report. You did NOT write this "
        "report and you must NOT fix or re-run it — only SCORE what you are given.\n\n"
        f"QUESTION:\n  {item.get('question', '')}\n\n"
        f"EXPECTED POINTS the report must cover:\n{expected}\n\n"
        f"MINIMUM SOURCES expected: {min_sources}\n\n"
        "THE REPORT (rendered Markdown):\n"
        "-----\n"
        f"{report.to_markdown()}"
        "-----\n\n"
        "RUBRIC — score each axis 0, 1 or 2:\n"
        "  coverage : do the findings cover ALL the expected points? "
        "(0 none, 1 some, 2 all)\n"
        "  grounding: are the findings backed by a saved chart and concrete numbers, not vague "
        "claims? (0 unbacked, 1 partial, 2 well-backed)\n"
        "  citations: when sources are expected, are claims cited [n] into real sources? "
        "(0 missing, 1 partial, 2 fully cited; score 2 if no sources were expected and none needed)\n\n"
        "RULES:\n"
        "  - Write your RATIONALE FIRST (the evidence), THEN the scores. The order matters.\n"
        "  - Reply with ONE JSON object and nothing else, in this exact shape:\n"
        '    {"rationale": "<2-4 sentences citing evidence>", '
        '"scores": {"coverage": <0-2>, "grounding": <0-2>, "citations": <0-2>}, '
        '"verdict": "pass" | "fail"}\n'
        "  - verdict is \"pass\" only if the report covers the expected points.\n"
    )


def parse_judge_response(text: str) -> dict:
    """Parse the judge's JSON ``{rationale, scores{...}, verdict}`` — robust to surrounding prose.

    A judge model may wrap its JSON in a code fence or a sentence; we extract the first ``{...}``
    block and parse it. Missing axes default to 0 (a judge that omits an axis scores it 0, not a
    crash). The result is clamped to the rubric range so a misbehaving judge can never invent a 9/2.

    Args:
        text: the raw judge message content.

    Returns:
        ``{"rationale": str, "scores": {axis: int}, "verdict": "pass"|"fail",
        "report_quality": int}`` where ``report_quality`` is the summed, clamped rubric total.

    Raises:
        ValueError: if no JSON object can be found at all (the judge returned unparsable prose) —
            an actionable failure, not a silent zero.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(
            "Judge response was not parsable JSON. Expected an object "
            '{"rationale": ..., "scores": {...}, "verdict": ...}; got: '
            f"{text[:200]!r}"
        )
    data = json.loads(match.group(0))

    raw_scores = data.get("scores", {}) or {}
    scores: dict[str, int] = {}
    for axis in JUDGE_RUBRIC_AXES:
        value = raw_scores.get(axis, 0)
        try:
            value = int(round(float(value)))
        except (TypeError, ValueError):
            value = 0
        scores[axis] = max(0, min(JUDGE_RUBRIC_AXIS_MAX, value))

    verdict = str(data.get("verdict", "fail")).strip().lower()
    if verdict not in ("pass", "fail"):
        verdict = "pass" if sum(scores.values()) >= JUDGE_RUBRIC_MAX - 1 else "fail"

    return {
        "rationale": str(data.get("rationale", "")).strip(),
        "scores": scores,
        "verdict": verdict,
        "report_quality": sum(scores.values()),  # 0..JUDGE_RUBRIC_MAX (6)
    }


def judge_report(report: QuillReport, item: dict, model) -> dict:
    """Score ONE ``QuillReport`` against a golden item with an LLM-as-judge (a SEPARATE call).

    Builds the rubric prompt (:func:`build_judge_prompt`), makes ONE ``model.generate`` call, and
    parses the structured JSON (:func:`parse_judge_response`). The ``model`` is a separate judge
    model (06 §2: from ``make_model`` — ideally a DIFFERENT ``model_id`` than Quill's, so the agent
    never grades itself). This is the evaluator-optimizer pattern composed by hand: Quill generates,
    the judge evaluates.

    Args:
        report: the ``QuillReport`` to score (FROZEN M8 schema; not modified).
        item: a golden-set entry (its ``expected_points`` / ``min_sources`` drive the rubric).
        model: a ``smolagents.Model`` for the judge (its ``generate`` returns a ``ChatMessage``).

    Returns:
        The parsed dict from :func:`parse_judge_response`: ``{rationale, scores, verdict,
        report_quality}``.
    """
    from smolagents import ChatMessage, MessageRole

    prompt = build_judge_prompt(report, item)
    messages = [ChatMessage(role=MessageRole.USER, content=prompt)]
    chat_message = model.generate(messages)
    content = chat_message.content
    if not isinstance(content, str):
        # Some models return a list of content blocks; join the text parts.
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return parse_judge_response(content)


def calibration_correlation(judge_scores: list[float], human_labels: list[float]) -> float:
    """Pearson correlation between the judge's scores and YOUR human labels (calibration helper).

    Documented + OPTIONAL (not run on every eval, 06 §6): run this on ~10-20 hand-labelled reports
    to check the judge tracks human judgement. Aim for ~0.80 as an order of magnitude (research, as
    of smolagents 1.26.0), not an absolute pass/fail. A judge you never calibrated measures its own
    bias (length/format/position), so re-run it periodically as the judge model drifts.

    Args:
        judge_scores: the judge's ``report_quality`` (or any axis) per example.
        human_labels: your hand-assigned scores for the SAME examples, same order.

    Returns:
        The Pearson correlation coefficient in ``[-1, 1]``; ``0.0`` if either series is constant
        (no variance to correlate) or the inputs are too short.
    """
    n = len(judge_scores)
    if n < 2 or n != len(human_labels):
        return 0.0
    mean_j = sum(judge_scores) / n
    mean_h = sum(human_labels) / n
    cov = sum((j - mean_j) * (h - mean_h) for j, h in zip(judge_scores, human_labels))
    var_j = sum((j - mean_j) ** 2 for j in judge_scores)
    var_h = sum((h - mean_h) ** 2 for h in human_labels)
    if var_j == 0 or var_h == 0:
        return 0.0
    return cov / (var_j ** 0.5 * var_h ** 0.5)
