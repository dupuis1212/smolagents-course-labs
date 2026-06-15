"""Your very first CodeAgent.

No tools, no CSV, no sandbox yet — the point of Module 1 is to feel a bare ``CodeAgent``
run: the model writes Python, smolagents executes it, and the answer comes back.
"""
from __future__ import annotations

import os

from smolagents import CodeAgent, InferenceClientModel, Model

# Pin an explicit code-capable model. We do NOT rely on InferenceClientModel's default
# model_id: it is documented as "subject to change" (as of smolagents 1.26.0), and a
# data/code agent wants a coder/instruct model. Module 4 turns this into make_model().
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"


def make_intro_model() -> Model:
    """Build the hosted model. Reads HF_TOKEN from the environment (never hard-coded).

    The free Hugging Face Inference credit allowance is tiny and subject to change
    (as of smolagents 1.26.0); Module 4 documents swapping to a free provider key or a
    local model.
    """
    return InferenceClientModel(model_id=DEFAULT_MODEL_ID, token=os.environ.get("HF_TOKEN"))


def build_first_agent(model: Model | None = None) -> CodeAgent:
    """A CodeAgent with an empty toolset — `tools=[]` is valid and deliberate.

    Pass your own ``model`` to run offline (e.g. a fake model in tests); otherwise a
    hosted Hugging Face model is used.
    """
    return CodeAgent(tools=[], model=model or make_intro_model())
