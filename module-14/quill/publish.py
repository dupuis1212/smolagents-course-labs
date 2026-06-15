"""Ship Quill to the Hub — ``save`` / ``push_to_hub`` and the ``app.py`` Space (Module 13, T12.3).

This wraps the agent-level ``MultiStepAgent.push_to_hub`` (distinct from the *tool*-level
``tool.push_to_hub`` you used in Module 9). Pushing an agent SERIALISES the whole Quill so it runs
as a one-click Hugging Face **Space**.

**What ``save()`` / ``push_to_hub`` write (the core of the module).**
``MultiStepAgent.save(output_dir)`` writes — and ``push_to_hub(repo_id)`` uploads — exactly:

- ``agent.json``       — the agent's dict representation (``to_dict()``).
- ``prompts.yaml``     — the agent's prompt templates.
- ``tools/<name>.py``  — ONE file per tool (``load_dataset.py``, ``profile_dataframe.py``,
                         ``save_chart.py``, ``retriever.py``, ``final_answer.py``).
- ``managed_agents/``  — each managed sub-agent's logic (so ``web_researcher/``), present ONLY
                         because Quill is multi-agent (M10). A solo agent would have no such dir.
- ``app.py``           — a ready-to-run ``GradioUI`` — THIS is what makes the repo run as a Space
                         with nothing else added.
- ``requirements.txt`` — the detected dependencies.

(The brief lists ``prompt.yaml``; smolagents 1.26.0 writes ``prompts.yaml`` — we assert the real
filename. ``app.py`` is the load-bearing one: it is the Space entry point.)

**The "pushable" friction (T3.15, applied at the AGENT level).** ``push_to_hub``/``save`` only
succeed if EVERY tool is "pushable": all imports live inside the tool's functions, ``__init__``
takes no argument beyond ``self``, and each tool body references no module-level helper (the saved
tool carries only its own source). Quill's tools were written to those rules from M3 — and Module
13 HARDENED them for the agent-level push: ``load_dataset``/``profile_dataframe`` inline their
read-table body (no shared ``_read_table``), and the ``RetrieverTool`` uses literal class
attributes and inlines its corpus walk (no ``load_corpus`` call). The friction the brief warns
about is real: a single outside reference makes ``agent.save`` raise. We surface the saved tree so
you can see it worked.

**Security (C7).** Pushing an agent PUBLISHES its tool code. Never embed a secret in a tool; the
token comes from ``.env`` (``HF_TOKEN``), never hard-coded. Use ``--private`` for a private repo.

**Loading it back (T12.4).** ``CodeAgent.from_hub(repo_id, trust_remote_code=True)`` reloads the
agent — ``trust_remote_code=True`` is MANDATORY because it downloads and EXECUTES remote tool code;
inspect it first (the same warning as the MCP ``trust_remote_code`` of M9). ``from_folder`` /
``from_dict`` are the local equivalents; deserialisation is gated by ``AGENT_REGISTRY`` (it maps
``"CodeAgent"``/``"ToolCallingAgent"`` strings to the classes). See ``reload_from_hub`` below.

CLI::

    uv run python -m quill.publish --repo "<your-user>/quill"             # push to the Hub
    uv run python -m quill.publish --repo "<your-user>/quill" --private   # push privately
    uv run python -m quill.publish --save-dir build/quill                 # save locally (offline)

The push itself needs ``HF_TOKEN`` and network; ``--save-dir`` runs FULLY OFFLINE (it is what the
smoke test exercises — ``agent.save(tmpdir)`` then assert the 6 artefacts).
"""
from __future__ import annotations

import pathlib

from .agent import build_quill

# The artefacts a successful agent save()/push_to_hub writes at the top level (smolagents 1.26.0).
# managed_agents/ is present because Quill has a sub-agent (web_researcher); app.py is the Space
# entry point. NOTE: 1.26.0 writes prompts.yaml (the brief's "prompt.yaml" is the same file).
EXPECTED_ARTIFACTS = [
    "agent.json",
    "prompts.yaml",
    "tools",
    "managed_agents",
    "app.py",
    "requirements.txt",
]

__all__ = [
    "EXPECTED_ARTIFACTS",
    "save_quill_locally",
    "publish_quill",
    "reload_from_hub",
    "main",
]


def save_quill_locally(output_dir: str, agent=None) -> list[str]:
    """Serialise the WHOLE Quill to ``output_dir`` (OFFLINE — no network) and list the artefacts.

    This is the offline proof that Quill is pushable: ``agent.save`` runs the SAME serialisation as
    ``push_to_hub`` (tool code, prompts, managed agents, the Space ``app.py``, requirements) but
    writes to a local folder instead of uploading. The smoke test calls this into a ``tmp_path`` and
    asserts the 6 artefacts exist.

    Args:
        output_dir: where to write the serialised agent.
        agent: the agent to save; ``None`` builds the real Quill via ``build_quill()`` (06 §2 — we
            never reconstruct the agent here). Tests pass a fake-model ``build_quill(model=...)``.

    Returns:
        The sorted list of top-level entry names written under ``output_dir``.
    """
    if agent is None:
        agent = build_quill()
    agent.save(output_dir)
    return sorted(p.name for p in pathlib.Path(output_dir).iterdir())


def publish_quill(
    repo_id: str,
    *,
    private: bool = False,
    commit_message: str = "Upload Quill",
    agent=None,
) -> str:
    """Push the WHOLE Quill to the Hub so it runs as a Space (T12.3) — NEEDS ``HF_TOKEN`` + network.

    Builds the real Quill (``build_quill()`` — never rebuilt here) and calls the agent-level
    ``push_to_hub(repo_id, commit_message=..., private=...)`` (signature: ``push_to_hub(repo_id,
    commit_message="Upload agent", private=None, token=None, create_pr=False)``). The token is read
    from the environment by ``huggingface_hub`` — we pass none here, and we NEVER hard-code one.

    This is a LIVE operation (it uploads to the Hub). The offline smoke test exercises
    ``save_quill_locally`` instead; the real push is covered by a ``live``-marked test.

    Args:
        repo_id: the target Hub repo, e.g. ``"your-user/quill"``.
        private: push to a private repo (``True``) — recommended if the tools touch anything
            sensitive; a public agent publishes its tool code (C7).
        commit_message: the commit message for this release.
        agent: the agent to push; ``None`` builds the real Quill.

    Returns:
        The repo URL returned by ``push_to_hub``.
    """
    if agent is None:
        agent = build_quill()
    return agent.push_to_hub(repo_id, commit_message=commit_message, private=private)


def reload_from_hub(repo_id: str, *, trust_remote_code: bool = True):
    """Reload a pushed Quill with ``CodeAgent.from_hub`` (T12.4) — NEEDS network.

    ``from_hub`` is a CLASSMETHOD inherited from ``MultiStepAgent``; prefer the class form
    ``CodeAgent.from_hub(...)`` over an instance ``agent.from_hub(...)`` (the latter works but
    misleads). ``trust_remote_code=True`` is MANDATORY — it downloads and EXECUTES remote tool code,
    so inspect the repo before trusting it (same warning as MCP's ``trust_remote_code``, M9).

    Args:
        repo_id: the Hub repo to load (e.g. ``"your-user/quill"``).
        trust_remote_code: required ``True`` to actually run the loaded tools.

    Returns:
        A reconstructed ``CodeAgent`` (the type comes from ``AGENT_REGISTRY`` via ``agent.json``).
    """
    from smolagents import CodeAgent

    return CodeAgent.from_hub(repo_id, trust_remote_code=trust_remote_code)


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m quill.publish --repo <id> [--private] | --save-dir <dir>``."""
    import sys

    args = sys.argv[1:] if argv is None else argv
    repo_id = None
    save_dir = None
    private = "--private" in args
    i = 0
    while i < len(args):
        if args[i] == "--repo" and i + 1 < len(args):
            repo_id = args[i + 1]
            i += 2
            continue
        if args[i] == "--save-dir" and i + 1 < len(args):
            save_dir = args[i + 1]
            i += 2
            continue
        i += 1

    if save_dir:
        artefacts = save_quill_locally(save_dir)
        print(f"[Quill] Saved the agent to {save_dir!r}. Top-level artefacts:")
        for name in artefacts:
            print(f"  - {name}")
        missing = [a for a in EXPECTED_ARTIFACTS if a not in artefacts]
        if missing:
            print(f"[Quill] WARNING: missing expected artefacts: {missing}")
        return 0

    if repo_id:
        url = publish_quill(repo_id, private=private)
        print(f"[Quill] Pushed to {url}. It now runs as a Space (app.py is the entry point).")
        print("[Quill] Reload it with: CodeAgent.from_hub(<repo>, trust_remote_code=True)")
        return 0

    print("Usage: python -m quill.publish --repo <user/quill> [--private] | --save-dir <dir>")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
