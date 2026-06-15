"""Space entry point — launch the Quill web app (Module 13).

This is the hand-written ``app.py`` for the labs repo (path (b) in the lab): a minimal, readable
Space entry point that wraps the REAL Quill in a Gradio UI. It exists so you can run the Space
locally and read exactly what serves it.

It delegates to ``quill.ui.launch_ui`` — which builds the agent via ``build_quill()`` (the frozen
construction entry point, 06 §2; ``app.py`` never reconstructs the agent) and wraps it in a
``QuillGradioUI`` with CSV upload and ``reset=False`` multi-turn memory.

Run it locally::

    uv run python app.py            # local only (share=False)
    uv run python app.py --share    # also open a temporary public *.gradio.live tunnel

Note the OTHER ``app.py``: ``agent.push_to_hub("user/quill")`` GENERATES its own ``app.py`` inside
the Hub repo (a ``GradioUI`` over the reloaded agent). That generated file — not this one — is what
makes the pushed repo run as a Space out of the box. This file is the readable local equivalent.
"""
from __future__ import annotations

import sys

from quill.ui import launch_ui

if __name__ == "__main__":
    launch_ui(share="--share" in sys.argv[1:])
