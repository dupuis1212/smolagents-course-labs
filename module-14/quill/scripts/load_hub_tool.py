"""Load ONE tool from the Hub and attach it to Quill's runtime toolbox (Module 9 — T3.13/T3.15).

A tool published to the Hub is a **Space repo** carrying the tool's source. ``load_tool`` (a
top-level smolagents function) and ``Tool.from_hub`` (the classmethod) both download that source
and reconstruct it as a smolagents ``Tool`` you can hand to any agent — so you reuse a teammate's
tool instead of rewriting it. Because the Hub code runs **locally**, ``trust_remote_code=True`` is
REQUIRED (the same interop security gate as MCP: you are running someone else's code on your
machine — only do it for a source you trust).

CONTRAST with MCP (06 §9): there is **no** ``Tool.from_mcp`` — MCP lives on
``ToolCollection.from_mcp`` / ``MCPClient``. ``Tool.from_*`` is for OTHER ecosystems:

    Tool.from_hub(repo_id, token=None, trust_remote_code=False)   # a Hub Space tool (local exec)
    Tool.from_space(space_id, name, description, api_name=None)   # a deployed Gradio Space
    Tool.from_gradio(gradio_tool)                                 # a gradio_tools object
    Tool.from_langchain(langchain_tool)                           # delegates to LangChain run()
    load_tool(repo_id, ..., trust_remote_code=False)              # top-level, same as from_hub

This script only DEMONSTRATES the load + attach; it makes a network call (and needs trust), so it
is guarded behind ``main()`` and never runs at import time. The offline test asserts the helper's
shape, not a real download.

    uv run python -m quill.scripts.load_hub_tool                     # default example repo
    uv run python -m quill.scripts.load_hub_tool m-ric/text-to-image # any Hub tool repo
"""
from __future__ import annotations

# A small, well-known example tool repo on the Hub. Re-verify it exists the day you run this
# (06 §9 — Hub repos move). Override it with the CLI arg / the function param for any other tool.
DEFAULT_HUB_TOOL_REPO = "m-ric/text-to-image"


def load_hub_tool(repo_id: str = DEFAULT_HUB_TOOL_REPO, *, trust_remote_code: bool = True):
    """Load a tool from the Hub with ``load_tool`` (Hub code runs locally — trust required).

    Args:
        repo_id: the Hub repo id of the tool (a Space repo carrying the tool source).
        trust_remote_code: forwarded to ``load_tool`` — REQUIRED ``True`` to run the Hub code
            locally (the interop security gate; only load tools you trust).

    Returns:
        a smolagents ``Tool`` ready to add to an agent's toolbox.
    """
    # Imported here so importing this module needs no network and no token.
    from smolagents import load_tool

    print(f"Loading Hub tool {repo_id!r} (trust_remote_code={trust_remote_code}) — runs its code locally.")
    tool = load_tool(repo_id, trust_remote_code=trust_remote_code)
    print(f"Loaded tool: name={tool.name!r}, output_type={tool.output_type!r}.")
    return tool


def attach_to_quill(agent, tool) -> None:
    """Attach a loaded tool to a built Quill at RUNTIME (the toolbox is a name-keyed dict).

    ``agent.tools`` is a ``dict[str, Tool]`` keyed by ``tool.name`` (T3.4/T3.11), so a tool can be
    added after construction — the same seam ``build_quill(extra_tools=...)`` uses at build time.

    Args:
        agent: a built Quill (``CodeAgent``).
        tool: the ``Tool`` to add (e.g. from ``load_hub_tool``).
    """
    agent.tools[tool.name] = tool
    print(f"Attached {tool.name!r} to Quill's toolbox (now {len(agent.tools)} tools).")


def main() -> int:
    """CLI: ``python -m quill.scripts.load_hub_tool [<repo_id>]`` (makes a real Hub call)."""
    import sys

    from quill.agent import build_quill

    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_HUB_TOOL_REPO
    tool = load_hub_tool(repo_id)
    with build_quill() as agent:
        attach_to_quill(agent, tool)
        print("Quill toolbox names:", sorted(agent.tools))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
