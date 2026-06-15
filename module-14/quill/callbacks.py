"""Quill's step callbacks — context engineering on the agent's memory (Module 6, Module 11).

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
``build_quill`` wires it in.

**Module 11 change — the screenshot callback (``save_screenshot``).** This file was always a
reuse point, and Module 11 cashes that in. ``save_screenshot(memory_step, agent)`` is the EXACT
same callback hook in its full *vision* form: it takes a browser screenshot every step and
injects it into ``memory_step.observations_images`` — the ``list[PIL.Image.Image] | None`` field
on ``ActionStep`` that a VLM reads on the NEXT step — then PRUNES the screenshots from steps
older than ``current - KEEP_LAST_SCREENSHOTS`` so context (and cost) stays bounded. A screenshot
is far more expensive in tokens than a one-line observation (a single image can cost as much as
hundreds of text tokens), so unbounded screenshots blow up the bill: pruning is not optional. It
is the SAME context-engineering idea as ``prune_old_observations`` above, applied to the
``observations_images`` channel instead of the text ``observations``. ``save_screenshot`` is
wired onto the OPTIONAL ``vision_browser`` sub-agent (``quill/team.py``), not the manager.

What this does NOT do (downstream, on purpose):
- No ``planning_interval`` / ``PlanningStep`` configuration (Module 7). A ``PlanningStep`` is
  simply ignored here — the callbacks only act on ``ActionStep``.
- No disk/Hub persistence of memory (Module 13). ``reset=False`` keeps memory in RAM only.
- No ``AgentError`` handling (Module 8). ``error`` is only ever a field we leave untouched.
- No image INPUT via ``run(images=...)`` here — that is the VLM chart self-review in
  ``quill/agent.py`` (Module 11). This callback is the browser-screenshot half of vision: it
  feeds the VLM PIXELS of the page the agent is looking at, captured from helium's Chrome. The
  ``smolagents[vision]`` extra (helium + selenium) is what makes ``save_screenshot`` runnable —
  NOT what enables image input (that needs only a VLM). See ``quill/agent.py`` for that trap.

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

# Module 11 (the screenshot callback). Keep only the LAST screenshot(s); null out the
# observations_images of any step that is now >= this many steps behind the current one. The
# canonical smolagents web-browser example keeps step N and N-1 and clears N-2 and older — i.e.
# it prunes a step once it is two steps behind. We express that as "prune steps whose number is
# <= current - 2", so this value is 2. A screenshot is the single most expensive thing in a VLM
# context (one image ≈ hundreds of text tokens), so this bound is what keeps a 20-step browse
# from costing 20 images' worth of tokens on every step — it caps it at ~2.
KEEP_LAST_SCREENSHOTS = 2

# How long to let the page settle (JS animations, lazy images) before grabbing the screenshot,
# in seconds. The browser is asynchronous: shoot too early and you capture a half-rendered page.
# One second is the canonical example's value — bump it for heavier dashboards.
SCREENSHOT_SETTLE_SECONDS = 1.0


def prune_old_observations(memory_step: ActionStep, agent) -> None:
    """Null out big observations from steps older than ``KEEP_LAST`` — a ``step_callback``.

    Signature is the frozen smolagents callback shape ``(memory_step, agent)``. It is called
    after each step (from ``_finalize_step``); at that point ``memory_step`` is the step that
    JUST finished and ``agent.memory.steps`` holds the steps BEFORE it. We walk those prior
    steps and, for any ``ActionStep`` that is now more than ``KEEP_LAST`` steps behind the
    current one and whose ``observations`` exceed ``MAX_OBS_CHARS``, replace the body with
    ``PRUNE_MARKER``. The next ``write_memory_to_messages()`` then sends the marker, not the
    multi-kB dump — that is where the token saving actually happens.

    We mutate ``observations`` in place (the canonical smolagents pattern, generalised from
    the vision browser's ``save_screenshot`` callback which prunes ``observations_images``).
    A ``PlanningStep``/``TaskStep`` is ignored — only ``ActionStep`` carries observations.

    Args:
        memory_step: the step that just finished (only acted on if it is an ``ActionStep``).
        agent: the running agent, so we can read/mutate ``agent.memory.steps``.
    """
    if not isinstance(memory_step, ActionStep):
        return

    current = memory_step.step_number
    for step in agent.memory.steps:
        if not isinstance(step, ActionStep) or step is memory_step:
            continue
        # "older than KEEP_LAST steps behind": gap from the current step number. A reset=False
        # turn restarts step_number, so prior-turn steps have small numbers and are pruned too
        # (their big dumps are exactly what we want gone once a new turn is underway).
        if current - step.step_number < KEEP_LAST:
            continue
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


def prune_old_screenshots(memory_step: ActionStep, agent) -> None:
    """Null out ``observations_images`` on steps older than ``KEEP_LAST_SCREENSHOTS`` (Module 11).

    The image half of ``prune_old_observations``. Walks ``agent.memory.steps`` and, for every
    ``ActionStep`` whose ``step_number`` is ``<= current - KEEP_LAST_SCREENSHOTS``, sets
    ``observations_images = None`` — dropping the bulky PNG so the next
    ``write_memory_to_messages()`` does NOT re-send it to the VLM. This is the canonical
    smolagents web-browser pruning loop, isolated so it can be unit-tested OFFLINE with hand-built
    ``ActionStep``s carrying a fake PIL image (no real browser, no model).

    Why a separate field from text observations: a screenshot lives in
    ``ActionStep.observations_images`` (``list[PIL.Image.Image] | None``), not in the text
    ``observations`` string. A VLM reads the images; pruning them is what bounds the per-step
    image cost. We keep the LAST couple of screenshots (the agent usually needs to see the page it
    just acted on and the one before) and clear the rest.

    Args:
        memory_step: the step that just finished (only acted on if it is an ``ActionStep``).
        agent: the running agent, so we can read/mutate ``agent.memory.steps``.
    """
    if not isinstance(memory_step, ActionStep):
        return

    current = memory_step.step_number
    for step in agent.memory.steps:
        if not isinstance(step, ActionStep):
            continue
        # Prune any step that is now KEEP_LAST_SCREENSHOTS or more behind the current one. The
        # current step (and the one just before it) keep their screenshot; older ones are cleared.
        if step.step_number <= current - KEEP_LAST_SCREENSHOTS:
            step.observations_images = None


def save_screenshot(memory_step: ActionStep, agent) -> None:
    """Screenshot the browser into ``observations_images`` each step, pruning old ones (Module 11).

    The canonical smolagents vision-browser ``step_callback`` (helium + selenium), and the full
    *vision* form of this module's ``step_callback`` pattern. After each step it:

    1. lets the page settle (``sleep(SCREENSHOT_SETTLE_SECONDS)`` — JS/animations finish);
    2. grabs the current Chrome window as a PNG via ``helium.get_driver().get_screenshot_as_png()``;
    3. stores it as a PIL image in ``memory_step.observations_images = [Image.open(...).copy()]`` —
       the field a VLM reads on the NEXT step (so the agent literally *sees* the page);
    4. PRUNES the screenshots from steps ``<= current - KEEP_LAST_SCREENSHOTS`` (the cost guard —
       see ``prune_old_screenshots``);
    5. appends the current URL to the text ``observations`` so the model also has it in words.

    ``helium`` is imported lazily INSIDE the function so importing ``quill.callbacks`` never
    requires the ``smolagents[vision]`` extra — only RUNNING the vision browser does. When no
    browser is open (``helium.get_driver()`` returns ``None``), we skip the screenshot but STILL
    prune (and still record whatever URL we can): the callback never crashes a run for lack of a
    live driver. That guard is also what lets the OFFLINE test inject a fake PIL image: it calls
    ``prune_old_screenshots`` directly, and asserts ``save_screenshot`` injects + prunes when a
    fake driver is supplied via ``_screenshot_png``.

    NOTE (the Module 11 trap): the ``[vision]`` extra is helium+selenium (a BROWSER). It is what
    makes THIS callback runnable. It is NOT what lets a model *see* an image — that is a property
    of the model (a VLM) plus ``run(images=...)``. Two different things; see ``quill/agent.py``.

    Args:
        memory_step: the step that just finished (only acted on if it is an ``ActionStep``).
        agent: the running ``CodeAgent`` (the vision_browser), so we can prune ``agent.memory``.
    """
    if not isinstance(memory_step, ActionStep):
        return

    from time import sleep

    sleep(SCREENSHOT_SETTLE_SECONDS)  # let JS animations / lazy images settle before the shot

    png, current_url = _screenshot_png()
    if png is not None:
        image = _png_to_pil(png)
        memory_step.observations_images = [image]
        # Prune old screenshots ONLY once we actually have a new one to keep — otherwise a run
        # with no browser would silently clear history it never replaced.
        prune_old_screenshots(memory_step, agent)

    if current_url is not None:
        existing = memory_step.observations or ""
        memory_step.observations = existing + f"\nCurrent url: {current_url}"


def _screenshot_png():
    """Return ``(png_bytes, current_url)`` from helium's live Chrome, or ``(None, None)``.

    Isolated so ``save_screenshot`` stays a thin, readable orchestration and so the live browser
    dependency lives in ONE place. ``helium`` is imported here (lazily): no browser, no import
    cost on ``quill.callbacks``. If no driver is open we return ``(None, None)`` and the caller
    degrades gracefully (prune-only). This is the seam an OFFLINE test patches to inject a fake
    PNG without a real Chrome.
    """
    try:
        import helium
    except ImportError:
        # The [vision] extra is not installed — running the vision browser is impossible, but
        # importing callbacks (and the rest of Quill) must still work. Degrade to prune-only.
        return None, None

    driver = helium.get_driver()
    if driver is None:
        return None, None
    return driver.get_screenshot_as_png(), getattr(driver, "current_url", None)


def _png_to_pil(png: bytes):
    """Decode PNG bytes into a standalone PIL image (``.copy()`` detaches it from the buffer)."""
    from io import BytesIO

    from PIL import Image

    return Image.open(BytesIO(png)).copy()


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
