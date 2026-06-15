"""Publish Quill's frozen ``save_chart`` tool to the Hub (Module 9 — T3.15).

The point of this script: ``save_chart`` was written in Module 3 to obey the **"pushable" rules**,
so M9 publishes it with **zero rewrites** — the proof its M3 contract was right from day one. The
three pushable rules (06 §2 + smolagents docs):

  1. Methods are self-contained (use only their args / class attributes).
  2. **Every import lives INSIDE a method**, never at module top level. (``save_chart.setup`` and
     ``save_chart.forward`` import ``matplotlib`` inside themselves — see ``quill/tools/data.py``.)
  3. ``__init__`` takes **no argument other than ``self``** (init args are not serializable to the
     Hub). ``save_chart`` does not override ``__init__`` at all, so it is trivially compliant.

These rules exist because the Hub re-executes the tool's code in a FRESH Space, with none of your
original environment — a top-level ``import`` or an ``__init__(self, model_name)`` would have no
way to be reconstructed there, so ``save()`` / ``push_to_hub()`` raise.

Two sharing APIs (both verified against smolagents 1.26.0):

    tool.save(output_dir, tool_file_name="tool", make_gradio_app=True)
        # writes <output_dir>/<tool_file_name>.py + app.py + requirements.txt — INSPECT it
        # locally. We pass tool_file_name="save_chart" so the file is save_chart.py (the default
        # is "tool" -> tool.py).
    tool.push_to_hub(repo_id, commit_message="Upload tool", private=None, token=None,
                     create_pr=False)
        # uploads the same as a Space repo (with a Gradio UI); returns the Space URL

This script first ``save()``s locally (no network, safe in any test) and then — only with
``--push`` + ``HF_TOKEN`` — ``push_to_hub()``. We never hard-code the token (06 §3 /
``os.environ["HF_TOKEN"]``). The default repo uses your HF username; pass ``--repo`` to override.

    uv run python -m quill.scripts.push_save_chart                          # local save only
    uv run python -m quill.scripts.push_save_chart --push --repo me/quill-save-chart   # publish
"""
from __future__ import annotations

DEFAULT_LOCAL_DIR = "build/save_chart_tool"
DEFAULT_REPO = "quill-save-chart"  # combined with your HF username unless --repo is "user/name"


def save_save_chart_locally(output_dir: str = DEFAULT_LOCAL_DIR) -> str:
    """``save_chart.save(output_dir)`` — write the tool's Space files for local inspection.

    Proves the pushable rules pass WITHOUT any network: ``save()`` serializes the tool to
    ``<output_dir>/save_chart.py`` + ``app.py`` + ``requirements.txt``. If a rule were violated
    (a top-level import, an ``__init__`` arg) this would raise — so a clean save IS the proof.

    Args:
        output_dir: where to write the tool files (created if missing).

    Returns:
        The ``output_dir`` written.
    """
    from quill.tools import save_chart

    tool = save_chart()  # the frozen M3 Tool, instantiated (no __init__ args — pushable rule 3)
    # tool_file_name="save_chart" -> save_chart.py (the default "tool" would write tool.py).
    # save() raises if a pushable rule is broken; a clean write IS the proof the rules pass.
    tool.save(output_dir, tool_file_name="save_chart")
    print(f"Saved save_chart tool files to {output_dir}/ (save_chart.py, app.py, requirements.txt).")
    return output_dir


def push_save_chart(repo_id: str, *, private: bool | None = None) -> str:
    """``save_chart.push_to_hub(repo_id, token=os.environ['HF_TOKEN'])`` — publish to the Hub.

    Makes a real Hub upload, so it needs ``HF_TOKEN`` in the environment (never hard-coded). The
    token is read with ``os.environ["HF_TOKEN"]`` so a missing token fails loud and obvious.

    Args:
        repo_id: target Hub repo (e.g. ``"<user>/quill-save-chart"``). Created if missing.
        private: pass ``True`` for a private Space repo (default: the Hub's default).

    Returns:
        The published Space URL printed by smolagents.
    """
    import os

    from quill.tools import save_chart

    token = os.environ["HF_TOKEN"]  # KeyError (loud) if unset — never a hard-coded key (06 §3)
    tool = save_chart()
    url = tool.push_to_hub(repo_id, commit_message="Upload Quill save_chart tool", private=private,
                           token=token)
    print(f"Published save_chart to {url}")
    return url


def main() -> int:
    """CLI: local ``save()`` always; ``push_to_hub`` only with ``--push`` + ``HF_TOKEN``."""
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Save/publish Quill's save_chart tool to the Hub.")
    parser.add_argument("--push", action="store_true", help="also push_to_hub (needs HF_TOKEN)")
    parser.add_argument("--repo", default=None,
                        help="target repo id, e.g. 'me/quill-save-chart' (default: <user>/quill-save-chart)")
    parser.add_argument("--dir", default=DEFAULT_LOCAL_DIR, help="local save dir")
    parser.add_argument("--private", action="store_true", help="make the Space repo private")
    args = parser.parse_args()

    # Always: local save (the pushable-rules proof; no network).
    save_save_chart_locally(args.dir)

    if not args.push:
        print("Local save only. Re-run with --push and HF_TOKEN set to publish to the Hub.")
        return 0

    if not os.environ.get("HF_TOKEN"):
        print("--push requested but HF_TOKEN is not set. Set it (.env) and retry.")
        return 1

    repo_id = args.repo
    if repo_id is None:
        # Resolve the user's HF namespace from the token, so the default repo is <user>/<name>.
        from huggingface_hub import whoami

        user = whoami(token=os.environ["HF_TOKEN"]).get("name", "user")
        repo_id = f"{user}/{DEFAULT_REPO}"
    push_save_chart(repo_id, private=args.private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
