"""Module 1 — a bare CodeAgent, before Quill exists.

This package deliberately stays minimal (`quill_intro`, NOT `quill`): Module 1 freezes
nothing. The real `quill/` package — `make_model`, the data tools, the sandbox policy,
`QuillReport` — is born from Module 2 onward.
"""
from .first_agent import DEFAULT_MODEL_ID, build_first_agent, make_intro_model

__all__ = ["DEFAULT_MODEL_ID", "build_first_agent", "make_intro_model"]
