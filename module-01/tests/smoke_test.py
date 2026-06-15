"""Module 1 smoke tests.

Live call budget: 1 real LLM call (only when QUILL_LIVE_TESTS=1 and HF_TOKEN is set).
Everything else runs OFFLINE via the shared FakeModel (see repo-root conftest.py), so the
agent loop is functionally verified without a network or token.

Run from the repo root: ``uv run pytest module-01/tests/``
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest
from smolagents import CodeAgent

# Make `quill_intro` importable when running from the repo root.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from quill_intro.first_agent import DEFAULT_MODEL_ID, build_first_agent  # noqa: E402


def test_build_first_agent_is_a_codeagent(fake_model):
    agent = build_first_agent(model=fake_model(["final_answer(42)"]))
    assert isinstance(agent, CodeAgent)


def test_first_agent_runs_offline(fake_model):
    """The bare CodeAgent writes & runs Python and returns the answer — no token needed."""
    agent = build_first_agent(model=fake_model(["result = sum(range(1, 101))\nfinal_answer(result)"]))
    out = agent.run("Calculate the sum of all integers from 1 to 100")
    assert out == 5050


def test_default_model_is_pinned_not_smolagents_default():
    # We pin an explicit coder model; we must NOT silently rely on the library default.
    assert DEFAULT_MODEL_ID == "Qwen/Qwen2.5-Coder-32B-Instruct"


@pytest.mark.live
def test_first_agent_live():
    assert os.environ.get("HF_TOKEN"), "live test needs HF_TOKEN"
    agent = build_first_agent()
    out = agent.run("Calculate the sum of all integers from 1 to 100")
    assert "5050" in str(out)
