"""Quill's research team — managed sub-agents (Module 10; Module 11 adds ``vision_browser``).

Until Module 9, Quill was ONE ``CodeAgent`` ("one brain") that planned, wrote the pandas,
saved charts, AND went to the web itself (it owned ``WebSearchTool`` + ``VisitWebpageTool``).
Every page it visited landed *whole* in its own memory — the doc's own complaint:
"why fill the memory of the code-generating agent with all the content of the webpages
visited?". As of Module 10, Quill becomes a **manager** ``CodeAgent`` over a specialised
sub-agent, the **``web_researcher``**. The manager plans and analyses; the sub-agent fetches
and summarises web context and returns only a short summary with source URLs. The manager's
memory stays clean — that is **context isolation**, the real reason to reach for multi-agent
(NOT "more agents = better results").

**The current way to wire a team (smolagents 1.26.0).** There is no ``ManagedAgent`` — it was
deprecated in 1.8.0, dropped from the docs in 1.21.0, and is ABSENT from ``agents.py`` in
1.26.0. Any tutorial that does ``from smolagents import ManagedAgent`` is dead. The current
mechanism: ANY agent (``CodeAgent`` or ``ToolCallingAgent``) becomes callable by a manager the
moment you give it a **``name``** and a **``description``** ("to make this agent callable by
its manager"), then pass it to the manager via ``managed_agents=[...]`` (a ``MultiStepAgent``
argument, inherited by both agent types). ``build_quill`` (``quill/agent.py``) does exactly
that with the ``web_researcher`` this module builds.

**Why the worker is a ``ToolCallingAgent`` and the manager a ``CodeAgent``.** The decision
matrix from Module 3: a ``CodeAgent`` is a "problem solver" — it composes Python, ideal for
planning + arithmetic = the analyst manager (Quill). A ``ToolCallingAgent`` is a "dispatcher" —
it emits JSON tool calls validated against a schema, a single timeline = stepping through the
web one page at a time. Web navigation does not need to compose Python; the JSON tool-call is
safer and sufficient. (A ``ToolCallingAgent`` can fan out parallel calls via ``max_tool_threads``
/ a ``ThreadPoolExecutor`` — a real capability, but we do NOT use it here: the researcher is a
single timeline.)

**How the manager calls the sub-agent (the internals).** For a ``CodeAgent`` manager, smolagents
exposes each managed agent INSIDE the Python sandbox as a callable function whose signature is
auto-generated from the system-prompt template::

    def web_researcher(task: str, additional_args: dict[str, Any]) -> str:
        '''<the sub-agent's description>
        Args:
            task: Long detailed description of the task.
            additional_args: Dictionary of extra inputs (e.g. images, dataframes).
        '''

So Quill literally writes ``summary = web_researcher("Find the SaaS industry average churn")``
in its generated code, like calling any tool, and the run prints
"Here is the final answer from your managed agent 'web_researcher': ...". The
``additional_args`` channel is how rich objects (images, DataFrames) flow DOWN to a sub-agent —
the hook Module 11 reuses for vision. (A ``ToolCallingAgent`` manager would instead dispatch via
``execute_tool_call(tool_name, arguments)`` and list the team as
``- {{ agent.name }}: {{ agent.description }}`` in its prompt.)

**The constraint (Module 5 / T10.7).** A remote ``executor_type`` (docker/e2b) PLUS
``managed_agents`` raises ``Exception("Managed agents are not yet supported with remote code
execution.")`` — ``create_python_executor`` refuses the combination. The reason: in Approach 1
(snippet-in-sandbox) the model stays local and secrets (the HF token) are NOT shipped into the
sandbox, so a sub-agent could not authenticate its own LLM calls from inside. So Quill's team
runs in ``executor_type="local"`` here; running the WHOLE team *inside* a sandbox (Approach 2)
is the capstone, Module 15.

**Module 11 change — the OPTIONAL ``vision_browser`` sub-agent.** Some pages give a text
scraper NOTHING: a JS dashboard, an interactive chart, a canvas. For those, Module 11 adds a
SECOND managed sub-agent, the **``vision_browser``** — a ``CodeAgent`` (it composes Python to
drive a browser) that pilots **helium** (over **selenium**, Chrome) and *looks at screenshots*
of the page via a ``save_screenshot`` ``step_callback`` (``quill/callbacks.py``). Unlike the
``web_researcher`` (text only, always on), the ``vision_browser`` is OFF by default and only
wired in via ``build_quill(browse=True)``: it needs a real local Chrome (the ``smolagents[vision]``
extra = helium + selenium) and a VLM, and a vision browse is expensive (a screenshot per step).
It carries the canonical ``name="vision_browser"`` (06 §2 — never "browser"/"web_vision").

**The vision_browser stays ``executor_type="local"`` (Module 15 territory otherwise).** It is a
``CodeAgent`` that drives a real browser, so its Python must run where Chrome is — locally. And
like the ``web_researcher``, a remote executor + ``managed_agents`` raises the same exception, so
the whole team stays local in this module (Approach 2 — the entire system in a sandbox — is M15).

What this module does NOT add: no Approach 2 (Module 15), no nested sub-agents-of-sub-agents
(mentioned only), no RAG/retriever (Module 12). Frozen contracts are untouched: the model ALWAYS
comes from ``make_model`` (M4), and the run still ends in a cited ``QuillReport`` (M8).
"""
from __future__ import annotations

from smolagents import (
    CodeAgent,
    Model,
    ToolCallingAgent,
    VisitWebpageTool,
    WebSearchTool,
    tool,
)

from .callbacks import save_screenshot
from .config import make_model

# The canonical sub-agent name (06-FIL-ROUGE-SPEC §2). It MUST be exactly "web_researcher"
# (not "web_agent"/"researcher"/"WebAgent"): the manager calls it by this name, the M11
# diagram extends this team, and the tests pin it. `name` + `description` together are what
# make an agent callable by a manager (passed via managed_agents=[...] in build_quill).
WEB_RESEARCHER_NAME = "web_researcher"

# The description the manager reads to decide WHEN and HOW to delegate. For a CodeAgent manager
# this text becomes the docstring of the auto-generated `web_researcher(task, additional_args)`
# function in the sandbox, so it must read like a tool's docstring: what it does, what to give
# it, what it returns. Keep it focused — a vague description makes the manager mis-delegate.
WEB_RESEARCHER_DESCRIPTION = (
    "Searches the web and visits pages to fetch missing external context (industry "
    "benchmarks, market averages, public trends, definitions). Give it ONE focused "
    "question as the task; it returns a short text summary with the source URLs it used. "
    "It does NOT see your dataset — pass any numbers it needs inside the question. Use it "
    "only when the answer is not in the local data."
)

# The researcher's step budget (06 §6, T7 production note). A sub-agent runs its OWN ReAct loop
# up to max_steps, so each delegation can cost several LLM calls — a multi-agent run can triple
# the calls of a single agent. 10 is enough to search, visit a couple of pages and summarise,
# while the HF free tier ($0.10/month as of smolagents 1.26.0) stays a real wall. Lower it (e.g.
# to 3) and a deep search returns "empty-handed", which the manager then has to handle.
WEB_RESEARCHER_MAX_STEPS = 10


def build_web_researcher(
    model: Model | None = None,
    *,
    max_steps: int = WEB_RESEARCHER_MAX_STEPS,
    provide_run_summary: bool = False,
) -> ToolCallingAgent:
    """Build the ``web_researcher`` — Quill's managed web-research sub-agent (Module 10).

    A ``ToolCallingAgent`` (a "dispatcher": JSON tool calls, single timeline) tooled with
    ``WebSearchTool`` + ``VisitWebpageTool`` and capped at ``max_steps``. It carries the
    canonical ``name="web_researcher"`` and a focused ``description`` — the two attributes that
    make it **callable by a manager** when passed to ``managed_agents=[...]`` (the current
    mechanism; ``ManagedAgent`` is gone). ``build_quill`` (``quill/agent.py``) constructs this and
    registers it on the manager.

    The model ALWAYS comes from ``make_model`` (the M4 frozen contract — ONE place to swap the
    backend); pass your own ``model`` to run offline (a fake model in tests). ``role="researcher"``
    is forwarded so a later module could give the researcher a cheaper/faster model than the
    analyst manager without touching any call site — in M10 every role uses the same default.

    Args:
        model: a ``smolagents.Model``; if ``None``, ``make_model(role="researcher")`` is used.
        max_steps: the sub-agent's own ReAct step budget (default ``WEB_RESEARCHER_MAX_STEPS`` =
            10). Each step is an LLM call, so this is the per-delegation cost ceiling. Lower it to
            see the "empty-handed" failure mode surface back to the manager.
        provide_run_summary: when ``True``, the manager sees the sub-agent's full reasoning trace,
            not only its final answer (smolagents' ``provide_run_summary`` knob). Default ``False``
            (the manager gets only the summary string) — the "Try it yourself" toggle in the lab.

    Returns:
        A ``ToolCallingAgent`` named ``web_researcher``, ready to drop into ``managed_agents=[...]``.
    """
    return ToolCallingAgent(
        # The dispatcher's toolset: search the web, then read a page. Exactly ONE web-search tool
        # (web_search), per 06 §6. The FinalAnswerTool is added automatically (it is how the
        # sub-agent returns its summary), and for a ToolCallingAgent the python_interpreter tool
        # is included by the library — we do NOT pass add_base_tools (we want a focused toolset).
        tools=[WebSearchTool(), VisitWebpageTool()],
        # M4 frozen contract: the model comes from make_model — never an InferenceClientModel built
        # here, never HfApiModel. role="researcher" is forwarded for a future per-role swap.
        model=model or make_model(role="researcher"),
        # These TWO attributes are what make this agent callable by a manager (06 §2). The manager
        # calls it by this exact name; the description becomes the delegated function's docstring.
        name=WEB_RESEARCHER_NAME,
        description=WEB_RESEARCHER_DESCRIPTION,
        # The sub-agent's own ReAct budget — the per-delegation cost ceiling (T7 production note).
        max_steps=max_steps,
        # Optional knob (off by default): surface the sub-agent's reasoning to the manager, not
        # just its final answer. The lab's "Try it yourself" flips this to True to compare.
        provide_run_summary=provide_run_summary,
    )


# ======================================================================================
# Module 11 (OPTIONAL): the vision_browser sub-agent — a CodeAgent driving helium + Chrome.
# ======================================================================================

# The canonical sub-agent name (06-FIL-ROUGE-SPEC §2). It MUST be exactly "vision_browser"
# (not "browser"/"web_vision"/"VisionAgent"): the manager calls it by this name and the tests
# pin it. name + description together are what make an agent callable by a manager.
VISION_BROWSER_NAME = "vision_browser"

# The description the manager reads to decide WHEN to delegate to the vision browser. For a
# CodeAgent manager this becomes the docstring of the auto-generated `vision_browser(task,
# additional_args)` function in the sandbox. It must steer the manager to use vision ONLY when a
# text scraper would fail (JS/chart-heavy pages) — vision is expensive, so the description gates it.
VISION_BROWSER_DESCRIPTION = (
    "Visits JS- and chart-heavy web pages that a text scraper cannot read (interactive "
    "dashboards, canvas charts, image-only content) and LOOKS at screenshots of them with a "
    "vision model. Give it a URL and exactly what to find on the page; it navigates, reads what "
    "it sees, and returns a short text summary of what it found. Slower and more expensive than "
    "the web_researcher (it screenshots every step), so use it ONLY when a normal page visit "
    "returns nothing useful."
)

# The vision browser's step budget (06 §6). Each step takes a screenshot (≈ hundreds of VLM
# tokens) AND is an LLM call, so this is BOTH a cost and a latency ceiling. The canonical
# smolagents example uses 20; we cap a touch lower because Quill's vision browse is a fallback,
# not the main path. Bound it: a runaway browser screenshotting 50 steps would torch the free tier.
VISION_BROWSER_MAX_STEPS = 15


@tool
def go_back() -> None:
    """Go back one page in the browser history (like clicking the browser's Back button)."""
    import helium

    helium.get_driver().back()


@tool
def close_popups() -> None:
    """Close any modal or popup on the page by sending the Escape key (NOT for cookie banners)."""
    from selenium.webdriver.common.keys import Keys

    import helium

    helium.get_driver().find_element("tag name", "body").send_keys(Keys.ESCAPE)


@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """Search for ``text`` on the current page (like Ctrl-F) and scroll to the nth match.

    Args:
        text: the visible text to find on the page.
        nth_result: which occurrence to jump to (1-based; default the first match).
    """
    from selenium.webdriver.common.by import By

    import helium

    driver = helium.get_driver()
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if not elements:
        raise ValueError(f"No match found for {text!r} on the page.")
    if nth_result > len(elements):
        raise ValueError(f"Only {len(elements)} matches for {text!r} (asked for #{nth_result}).")
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    return f"Found {len(elements)} match(es) for {text!r}; scrolled to #{nth_result}."


def vision_browser_tools() -> list:
    """The vision browser's navigation tools — simple ``@tool`` functions over helium.

    Returned as a fresh list. These are the canonical smolagents web-browser helpers
    (``go_back`` / ``close_popups`` / ``search_item_ctrl_f``); the agent writes raw helium calls
    (``go_to(...)``, ``click(...)``, ``scroll_down(...)``) directly in its sandboxed code, so the
    toolbox stays tiny. ``helium`` / ``selenium`` are imported INSIDE each tool (so importing
    ``quill.team`` never needs the ``[vision]`` extra — only RUNNING the browser does).
    """
    return [go_back, close_popups, search_item_ctrl_f]


def build_vision_browser(
    model: Model | None = None,
    *,
    max_steps: int = VISION_BROWSER_MAX_STEPS,
) -> CodeAgent:
    """Build the OPTIONAL ``vision_browser`` — Quill's screenshot-reading sub-agent (Module 11).

    A ``CodeAgent`` (it composes Python to drive a browser) tooled with the helium navigation
    helpers, with ``"helium"`` on ``additional_authorized_imports`` (so its sandboxed code may
    ``from helium import *``) and the ``save_screenshot`` ``step_callback`` (``quill/callbacks.py``)
    that injects a PNG of the page into ``observations_images`` each step and prunes old ones. It
    carries the canonical ``name="vision_browser"`` + a focused ``description`` — the two
    attributes that make it callable by a manager when passed to ``managed_agents=[...]``.

    The model ALWAYS comes from ``make_model`` (the M4 frozen contract). For this sub-agent it
    MUST resolve to a **VLM** (vision-language model) — image input is a property of the MODEL, not
    of any extra. Point ``QUILL_MODEL_ID`` at a VLM (e.g. ``Qwen/Qwen2-VL-72B-Instruct`` via the
    ``hf`` backend, or ``gpt-4o`` via ``litellm``). ``role="vision"`` is forwarded so a later
    module could give the browser its own model without touching call sites.

    It STAYS ``executor_type="local"`` (the smolagents default here): helium needs a real local
    Chrome, and a remote executor + ``managed_agents`` would raise the M10 exception. Running the
    whole team inside a sandbox is Approach 2 — the capstone, Module 15.

    The reader must ALSO preload helium into the executor before the first run::

        agent.python_executor("from helium import *", agent.state)

    (this lab does it in ``quill/agent.py`` when ``browse=True``). We do NOT do it at construction
    because it would open a Chrome window the moment you build the agent.

    Args:
        model: a ``smolagents.Model`` (must be a VLM for a real browse); if ``None``,
            ``make_model(role="vision")`` is used.
        max_steps: the sub-agent's own ReAct budget (default ``VISION_BROWSER_MAX_STEPS`` = 15).
            BOTH a cost and a latency ceiling — every step is an LLM call AND a screenshot.

    Returns:
        A ``CodeAgent`` named ``vision_browser``, ready to drop into ``managed_agents=[...]``.
    """
    return CodeAgent(
        # Tiny navigation toolset; the agent writes raw helium calls (go_to/click/scroll_down) in
        # its own sandboxed code. The FinalAnswerTool is auto-added (it is how it returns).
        tools=vision_browser_tools(),
        # M4 frozen contract: the model comes from make_model — and for a REAL browse it must be a
        # VLM (Qwen2-VL / gpt-4o). role="vision" is forwarded for a future per-role model swap.
        model=model or make_model(role="vision"),
        # name + description make it callable by the manager (06 §2). NEVER ManagedAgent.
        name=VISION_BROWSER_NAME,
        description=VISION_BROWSER_DESCRIPTION,
        # The agent's code does `from helium import *`, so helium MUST be authorized in its
        # sandbox. This is an explicit, scoped addition for THIS sub-agent only — Quill's frozen
        # least-privilege import lock (quill/sandbox.py) is untouched (no "helium", and never "*").
        additional_authorized_imports=["helium"],
        # The screenshot step_callback: shoot the page into observations_images each step + prune
        # the old shots (cost). The SAME callback hook as Module 6, in its full vision form.
        step_callbacks=[save_screenshot],
        # Bound the browse — every step is an LLM call AND a screenshot (cost + latency ceiling).
        max_steps=max_steps,
    )


__all__ = [
    "WEB_RESEARCHER_NAME",
    "WEB_RESEARCHER_DESCRIPTION",
    "WEB_RESEARCHER_MAX_STEPS",
    "build_web_researcher",
    "VISION_BROWSER_NAME",
    "VISION_BROWSER_DESCRIPTION",
    "VISION_BROWSER_MAX_STEPS",
    "build_vision_browser",
    "vision_browser_tools",
    "go_back",
    "close_popups",
    "search_item_ctrl_f",
]
