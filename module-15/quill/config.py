"""Quill's model layer — the FROZEN model contract (06-FIL-ROUGE-SPEC §2, Module 4).

Until now the model was hard-wired in ``agent.py`` (an ``InferenceClientModel(...)`` built
right where the agent is created). That couples every future module to one backend and gives
you no single place to swap providers or watch cost. This module fixes that: ``config.py`` is
the ONE place where Quill decides what powers it.

The frozen contract every later module reuses:

    make_model(role: str = "analyst", **overrides) -> smolagents.Model

- Default: ``InferenceClientModel`` with an EXPLICIT coder ``model_id`` (we never rely on the
  library default model_id — it is documented as "subject to change" as of smolagents 1.26.0).
- Swap with two env vars, no agent-code edit:
    * ``QUILL_MODEL_BACKEND`` in {hf, litellm, local}   (default "hf")
    * ``QUILL_MODEL_ID``                                 (default the pinned coder model)
- Token via ``os.environ["HF_TOKEN"]`` (+ a .env file); never hard-coded.
- ``role`` is part of the frozen signature now (so the call sites are stable) but only selects
  the default in M4. A later module can branch on it (e.g. a cheaper model for a researcher
  sub-agent vs the analyst manager) without touching any caller.

What this module does NOT do (downstream, on purpose):
- No ``response_format`` / structured outputs (Module 8). ``generate`` accepts a
  ``response_format`` arg in 1.26.0; we just don't drive it here.
- No sandbox / ``executor_type`` (Module 5) — Quill still runs in the local executor.
- ``LiteLLMRouterModel``, ``bill_to=``, ``VLLMModel`` are real options described in the article
  but are NOT on Quill's required path.
"""
from __future__ import annotations

import os

from smolagents import InferenceClientModel, LiteLLMModel, Model

# --- Pinned defaults (06 §4). The model_id is EXPLICIT: a CodeAgent writes and runs Python
# every step, so it needs a code-capable model with a large context. We do NOT depend on
# InferenceClientModel's own default model_id (Qwen/Qwen3-Next-80B-A3B-Thinking as of 1.26.0,
# "subject to change"). Re-verify Inference-Providers availability at build time.
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_BACKEND = "hf"
SUPPORTED_BACKENDS = ("hf", "litellm", "local")

# Sensible local default if someone flips to the "local" backend without naming a model:
# Ollama (served at localhost:11434) reached via LiteLLM. The "ollama_chat/" prefix is what
# LiteLLM expects for an Ollama chat model.
DEFAULT_LOCAL_MODEL_ID = "ollama_chat/llama3.2"
OLLAMA_API_BASE = "http://localhost:11434"
# Ollama's default num_ctx is 2048, which "fails horribly" for a multi-step agent (the prompt
# + growing memory overflow it). We raise it well above that for any local Ollama run.
OLLAMA_NUM_CTX = 8192


class Settings:
    """Environment-driven configuration: the ONE place that knows which backend and model
    Quill uses, so "change the model in all of Quill" is a one-line, one-file change.

    The two knobs are exposed as **class properties** so they re-read the environment on every
    access. That keeps the swap honest: ``QUILL_MODEL_BACKEND=litellm uv run ...`` takes effect
    even if ``quill.config`` was imported before the env var was set, and tests can monkeypatch
    the environment without re-importing the module.
    """

    # The defaults live as plain attributes so callers/tests can read the pins directly.
    DEFAULT_BACKEND: str = DEFAULT_BACKEND
    DEFAULT_MODEL_ID: str = DEFAULT_MODEL_ID

    class _EnvProperty:
        """A read-only class-level property that resolves an env var at access time."""

        def __init__(self, env_var: str, default: str, lower: bool = False):
            self.env_var = env_var
            self.default = default
            self.lower = lower

        def __get__(self, instance, owner) -> str:
            value = os.environ.get(self.env_var, self.default)
            return value.strip().lower() if self.lower else value

    # QUILL_MODEL_BACKEND in {hf, litellm, local}. Default "hf" = the free Hugging Face
    # Inference-Providers path (one HF_TOKEN, provider="auto").
    MODEL_BACKEND = _EnvProperty("QUILL_MODEL_BACKEND", DEFAULT_BACKEND, lower=True)

    # Explicit coder model_id by default; override per backend with QUILL_MODEL_ID.
    MODEL_ID = _EnvProperty("QUILL_MODEL_ID", DEFAULT_MODEL_ID)


def make_model(role: str = "analyst", **overrides) -> Model:
    """Build the model that powers Quill — the single entry point for every LLM call.

    Dispatches on ``Settings.MODEL_BACKEND`` (read from ``QUILL_MODEL_BACKEND``):

    - ``"hf"`` (default) -> ``InferenceClientModel(model_id=Settings.MODEL_ID)``. The HF token
      is read from ``HF_TOKEN`` by the client; ``provider="auto"`` (the default since 1.16.0)
      routes the request through the HF router to the first available Inference Provider. To
      pin a partner, pass ``provider="together"`` (etc.) via ``overrides``.
    - ``"litellm"`` -> ``LiteLLMModel(model_id=Settings.MODEL_ID)`` — 100+ providers behind one
      class. The id format is ``"<provider>/<model>"`` (e.g. ``"anthropic/claude-3-5-sonnet-
      latest"``, ``"gpt-4o"``, ``"gemini/gemini-1.5-pro"``). Needs the ``[litellm]`` extra and
      that provider's key (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``, ...) in the environment.
    - ``"local"`` -> Ollama via ``LiteLLMModel`` (the easy laptop path): ``ollama_chat/<model>``
      against ``http://localhost:11434`` with ``num_ctx`` raised to a usable size. (For raw GPU
      serving you would reach for ``TransformersModel`` / ``VLLMModel`` instead — described in
      the article, not on Quill's required path.)

    Args:
        role: which Quill role the model is for (frozen for later modules; M4 only uses the
            default for every role).
        **overrides: forwarded to the model constructor. Completion kwargs like ``temperature``
            or ``max_tokens`` go HERE (passed at init and forwarded on every call) — that is the
            uniform smolagents pattern across all Model classes. You can also pass
            ``requests_per_minute=`` (an ApiModel client-side rate-limit guard) or ``provider=``.

    Returns:
        A ``smolagents.Model`` ready to hand to ``CodeAgent(model=...)``.

    Raises:
        ValueError: if ``QUILL_MODEL_BACKEND`` is not one of {hf, litellm, local}. We fail loud
            instead of silently falling back, so a typo never sends you to the wrong backend.
    """
    backend = Settings.MODEL_BACKEND

    # M14: a caller may pass an explicit `model_id` override (the eval judge points at a SEPARATE
    # model — ideally a different/stronger one than Quill, so the agent never grades itself). It
    # POPS here so it cleanly replaces the Settings.MODEL_ID default instead of duplicating the
    # keyword. No prior call site passes model_id, so M4 behaviour is unchanged when it is absent.
    model_id_override = overrides.pop("model_id", None)

    if backend == "hf":
        # token=None lets InferenceClientModel fall back to HF_TOKEN from the environment.
        return InferenceClientModel(model_id=model_id_override or Settings.MODEL_ID, **overrides)

    if backend == "litellm":
        # LiteLLM reads the relevant provider key from the environment itself.
        return LiteLLMModel(model_id=model_id_override or Settings.MODEL_ID, **overrides)

    if backend == "local":
        # If the user kept the HF default model_id, swap in a sensible local one.
        model_id = model_id_override or Settings.MODEL_ID
        if model_id == DEFAULT_MODEL_ID:
            model_id = DEFAULT_LOCAL_MODEL_ID
        kwargs = {"api_base": OLLAMA_API_BASE, "num_ctx": OLLAMA_NUM_CTX}
        kwargs.update(overrides)  # explicit caller overrides win
        return LiteLLMModel(model_id=model_id, **kwargs)

    raise ValueError(
        f"Unknown QUILL_MODEL_BACKEND {backend!r}. "
        f"Supported backends: {', '.join(SUPPORTED_BACKENDS)}."
    )
