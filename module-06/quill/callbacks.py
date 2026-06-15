"""Quill's step callbacks — context engineering on the agent's memory (Module 6).

A smolagents agent's memory is just a list of typed steps (``agent.memory.steps``). Every
step the agent reconstructs that list into chat messages with
``write_memory_to_messages()`` and sends them to the model — so anything still sitting in an
old step's ``observations`` is RE-SENT to the LLM on every later step. A ``CodeAgent`` that
``print(df.head(50))`` once therefore pays for that dump on every subsequent step (latency
and tokens, both growing without bound on a long ``reset=False`` session).

This module is the lever. A **step callback** has the frozen smolagents signature
``(memory_step, agent)`` and is run inside the agent's ``_finalize_step`` (via a
``CallbackRegistry``) AFTER each step. Because it receives the ``agent``, it can read AND
MUTATE ``agent.memory.steps`` — i.e. rewrite history before the next
``write_memory_to_messages()`` call. We use that to:

- ``prune_old_observations`` — null out the big DataFrame dumps in stale ``ActionStep``s, so
  the model stops re-reading data it has already summarised.
- ``log_step_cost`` — print one ``step N: <in>+<out> tokens`` line per step from
  ``ActionStep.token_usage`` (the minimal cost observability brick before any telemetry).

``quill_callbacks()`` returns the list to hand to ``CodeAgent(step_callbacks=...)``;
``build_quill`` wires it in. This file is itself a reuse point: Module 11 adds a screenshot
callback that prunes ``observations_images`` with the exact same pattern.

What this does NOT do (downstream, on purpose):
- No ``planning_interval`` / ``PlanningStep`` configuration (Module 7). A ``PlanningStep`` is
  simply ignored here — the callbacks only act on ``ActionStep``.
- No disk/Hub persistence of memory (Module 13). ``reset=False`` keeps memory in RAM only.
- No ``AgentError`` handling (Module 8). ``error`` is only ever a field we leave untouched.

Signature note (smolagents 1.26.0): ``CallbackRegistry.callback`` calls a 1-arg callback as
``cb(memory_step)`` and a multi-arg one as ``cb(memory_step, agent=agent)``. We always write
the two-arg form ``(memory_step, agent)`` — never the removed ``agent.logs`` and never an
inverted ``(agent, memory_step)``.
"""
from __future__ import annotations

from typing import Callable

from smolagents import ActionStep

# How many of the most recent steps to keep verbatim. Anything OLDER than this many steps
# behind the current one gets its bulky observation pruned. Two is the production default the
# article argues for: the model has almost always summarised a dump into its own reasoning
# within a step or two, after which re-sending the raw bytes only costs tokens.
KEEP_LAST = 2

# Only prune observations bigger than this — small ones (a one-line summary, a number) are
# cheap and sometimes the only record of what happened, so we leave them alone.
MAX_OBS_CHARS = 1000

# The marker we leave in place of a pruned dump, so a later replay()/inspection still shows
# that a step ran and produced output — just not the multi-kB body.
PRUNE_MARKER = "[pruned: large DataFrame dump removed to save tokens]"


def prune_old_observations(memory_step: ActionStep, agent) -> None:
    """Null out big observations from all but the last ``KEEP_LAST`` steps — a ``step_callback``.

    Signature is the frozen smolagents callback shape ``(memory_step, agent)``. It is called
    after each step (from ``_finalize_step``); at that point ``memory_step`` is the step that
    JUST finished and ``agent.memory.steps`` holds the steps BEFORE it, in order. We keep the
    most recent ``KEEP_LAST`` ``ActionStep``s verbatim (``memory_step`` plus the last
    ``KEEP_LAST - 1`` prior ones) and, for any older ``ActionStep`` whose ``observations``
    exceed ``MAX_OBS_CHARS``, replace the body with ``PRUNE_MARKER``. The next
    ``write_memory_to_messages()`` then sends the marker, not the multi-kB dump — that is where
    the token saving actually happens.

    Age is measured by **position in ``agent.memory.steps``**, NOT by ``step_number``. This is
    the subtle, load-bearing choice: ``run(reset=False)`` restarts ``step_number`` at 1 every
    turn (smolagents resets it at the top of each run — see its own "there can be steps from
    previous runs" comment), so a previous turn's steps keep small numbers. Keying on the
    number would compute a negative/tiny gap for those steps and skip them — leaving a prior
    turn's fat dumps in memory forever, which is the exact long multi-turn session this
    callback exists to keep cheap. The order of the list, by contrast, is always globally
    monotonic across turns, so list position is the true age.

    We mutate ``observations`` in place (the canonical smolagents pattern, generalised from
    the vision browser's ``save_screenshot`` callback which prunes ``observations_images``).
    A ``PlanningStep``/``TaskStep`` is ignored — only ``ActionStep`` carries observations.

    Args:
        memory_step: the step that just finished (only acted on if it is an ``ActionStep``).
        agent: the running agent, so we can read/mutate ``agent.memory.steps``.
    """
    if not isinstance(memory_step, ActionStep):
        return

    # Prior ActionSteps, in memory order (oldest first). memory_step is normally not in this
    # list yet, but we exclude it defensively in case the finalize/append order ever changes.
    prior_action_steps = [
        step for step in agent.memory.steps
        if isinstance(step, ActionStep) and step is not memory_step
    ]
    # Keep the tail (the last KEEP_LAST-1 prior steps, which together with memory_step are the
    # last KEEP_LAST action steps); everything before that index is "old" and gets pruned.
    keep_from = max(0, len(prior_action_steps) - (KEEP_LAST - 1))
    for step in prior_action_steps[:keep_from]:
        observations = step.observations
        if observations is not None and len(observations) > MAX_OBS_CHARS and observations != PRUNE_MARKER:
            step.observations = PRUNE_MARKER


def log_step_cost(memory_step: ActionStep, agent) -> None:
    """Print one ``step N: <in>+<out> tokens`` line per step — a ``step_callback``.

    Reads ``memory_step.token_usage`` (a ``TokenUsage`` with ``input_tokens`` /
    ``output_tokens`` / ``total_tokens``). It can legitimately be ``None`` — an offline fake
    model, or a step that errored before the model replied — so we test ``is not None``
    explicitly rather than swallow an ``AttributeError`` in a silent ``try/except``.

    This is the minimal cost-observability brick: cost in plain sight, per step, before you
    reach for real telemetry (OpenTelemetry, Module 14).

    Args:
        memory_step: the step that just finished (only logged if it is an ``ActionStep``).
        agent: the running agent (unused here; part of the frozen callback signature).
    """
    if not isinstance(memory_step, ActionStep):
        return

    usage = memory_step.token_usage
    if usage is None:
        # No usage to report (offline model / pre-model error) — stay quiet but explicit.
        return

    print(
        f"[Quill] step {memory_step.step_number}: "
        f"{usage.input_tokens}+{usage.output_tokens} tokens "
        f"(total {usage.total_tokens})"
    )


def quill_callbacks() -> list[Callable]:
    """The step_callbacks list Quill wires into its agent — prune first, then log.

    Returned as a LIST (run on every step, regardless of step type — the functions themselves
    short-circuit on non-``ActionStep`` steps). The dict-by-type form
    ``{ActionStep: [prune_old_observations, log_step_cost]}`` is equally valid (smolagents
    runs those only for that step type) and is left as a "Try it yourself" in the lab.

    Order matters: we prune stale observations BEFORE logging, so the logged step reflects the
    memory state the next ``write_memory_to_messages()`` will actually serialise.

    Returns:
        ``[prune_old_observations, log_step_cost]`` — hand straight to
        ``CodeAgent(step_callbacks=...)``.
    """
    return [prune_old_observations, log_step_cost]
