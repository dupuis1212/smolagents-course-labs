"""Quill's sandbox demo (Module 5) — run in a sandbox, watch a dangerous import get blocked.

Two things this script makes concrete:

1. **A real run inside the chosen executor.** ``QUILL_EXECUTOR=docker uv run python -m
   quill.demo_sandbox`` builds Quill with ``executor_type="docker"`` (via the frozen
   ``resolve_executor()``), starts a Docker container, and cleans it up afterward (no dangling
   containers). The ``local`` path answers a small question about ``data/sales.csv`` and needs
   a model (``HF_TOKEN``). NOTE: under a REMOTE executor, smolagents must SEND the agent's
   custom tools into the sandbox; Quill's ``@tool`` data tools reference a module-level helper
   and are not remotely-serializable yet, so the docker normal-run reports that caveat instead
   of a full toolbox answer (the container and the import lock still work — see ``--attack``).

2. **The import lock and the loop cap, demonstrated WITHOUT a model.** ``--attack`` does NOT
   call an LLM. It builds Quill (offline-safe: a no-op fake model is injected) and feeds the
   agent's executor two hostile snippets DIRECTLY:
     - ``import os; os.system(...)`` -> blocked: ``os`` is not in Quill's authorized imports,
       so the executor raises ``InterpreterError`` and the host process is never touched.
     - a ``while True`` loop -> cut off by the executor's iteration cap
       (``MAX_WHILE_ITERATIONS = 1_000_000`` as of smolagents 1.26.0), again ``InterpreterError``.
   This proves the FIRST layer of defense (the AST allow-list) fires even before we reach the
   sandbox boundary — and it runs anywhere, no Docker or token needed.

Usage:
    uv run python -m quill.demo_sandbox                 # normal run (executor from QUILL_EXECUTOR)
    QUILL_EXECUTOR=docker uv run python -m quill.demo_sandbox
    uv run python -m quill.demo_sandbox --attack        # blocked import + loop cap (offline)

Verified against smolagents 1.26.0. ``executor_type="wasm"`` does not exist in 1.26.0.
"""
from __future__ import annotations

import sys

from smolagents import ChatMessage, MessageRole, Model
from smolagents.local_python_executor import InterpreterError

from .agent import build_quill, build_task
from .sandbox import resolve_executor


class _NoModel(Model):
    """An offline stand-in model: the --attack demo never asks the model for an action (it
    drives the executor directly), so this just satisfies CodeAgent's ``model=`` requirement
    without any network. If it were ever called it would say so loudly rather than hit an API.
    """

    def __init__(self) -> None:
        super().__init__(model_id="demo/no-model")

    def generate(self, messages, **kwargs) -> ChatMessage:  # pragma: no cover - never called
        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content="Thought: nothing to do.\n<code>\nfinal_answer('noop')\n</code>",
        )


def _run_snippet(agent, label: str, code: str) -> None:
    """Feed one snippet straight to the agent's executor and report blocked vs. allowed.

    We hit ``agent.python_executor(code)`` directly (the same executor the agent uses every
    step) so the demo shows the SANDBOX verdict, not the model's behaviour.
    """
    print(f"\n--- {label} ---")
    print(code.strip())
    try:
        agent.python_executor(code)
        print(">>> NOT blocked (unexpected).")
    except InterpreterError as exc:
        # blocked: the allow-list / cap fired before any harm was done.
        print(f">>> BLOCKED by the executor: InterpreterError: {exc}")


def attack_demo() -> int:
    """Show the import lock and the loop cap firing — offline, no Docker, no token."""
    executor_type, authorized = resolve_executor()
    print(f"[demo] executor={executor_type!r}  authorized_imports={authorized}")
    # Build Quill with a no-op model so we can reach its executor without any LLM call.
    agent = build_quill(model=_NoModel())

    _run_snippet(
        agent,
        "Attack 1: import os and run a shell command (data-exfil / arbitrary code)",
        # blocked: 'os' is not in authorized imports — the AST allow-list refuses the import.
        "import os\nos.system('echo pwned > /tmp/quill_pwned.txt')\n",
    )
    _run_snippet(
        agent,
        "Attack 2: a runaway loop (resource abuse / denial of service)",
        # blocked: the executor caps While-loop iterations (MAX_WHILE_ITERATIONS = 1_000_000).
        "x = 0\nwhile True:\n    x += 1\n",
    )

    print(
        "\n[demo] Both snippets were stopped by the LocalPythonExecutor's first layer of "
        "defense (the AST allow-list + caps). Remember: that layer is a surface-area "
        "REDUCTION, not a security sandbox — for untrusted inputs run QUILL_EXECUTOR=docker."
    )
    return 0


def normal_run(csv_path: str, question: str) -> int:
    """Run Quill once inside the chosen executor, print the answer, and clean up.

    Uses the context manager so a remote sandbox (Docker/E2B) is torn down deterministically —
    no dangling containers. This path makes a real model call (set HF_TOKEN), and a Docker run
    also needs a running daemon + the [docker] extra.
    """
    executor_type, authorized = resolve_executor()
    print(f"[demo] executor={executor_type!r}  authorized_imports={authorized}")
    if executor_type == "local":
        print(
            "[demo] NOTE: 'local' is the in-process executor (NOT a security sandbox). "
            "Set QUILL_EXECUTOR=docker for real isolation."
        )

    # `with build_quill(...) as agent:` guarantees agent.cleanup() runs — for Docker/E2B that
    # stops and removes the container/sandbox even if the run raises.
    with build_quill() as agent:
        try:
            result = agent.run(build_task(csv_path, question), return_full_result=True)
        except ValueError as exc:
            # REAL Approach-1 caveat: under a remote executor, smolagents sends the agent's
            # custom tools INTO the sandbox via send_tools(). Quill's @tool data tools reference
            # a module-level helper (_read_table), which the remote-serialization path rejects
            # ("SimpleTool validation failed ... Name _read_table is undefined"). The container
            # and the import lock work; sending these particular tools does not (yet).
            if "SimpleTool validation failed" in str(exc) and executor_type != "local":
                print(
                    "\n[demo] Quill's @tool data tools could not be SENT into the "
                    f"{executor_type!r} sandbox (a real Approach-1 caveat): {exc}\n"
                    "[demo] The container started and the import lock is active; making the "
                    "tools self-contained for remote sending is out of scope for this module."
                )
                return 0
            raise
        print("\n===== ANSWER =====")
        print(result.output)
        print(f"\n[demo] run state: {result.state}")
    print("[demo] sandbox cleaned up (context manager exited).")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI: ``--attack`` runs the offline block demo; otherwise a normal sandboxed run."""
    args = sys.argv[1:] if argv is None else argv
    if "--attack" in args:
        return attack_demo()

    positional = [a for a in args if not a.startswith("-")]
    csv_path = positional[0] if positional else "data/sales.csv"
    question = (
        positional[1]
        if len(positional) > 1
        else "How many rows are in the dataset, and which category has the highest net_rev?"
    )
    return normal_run(csv_path, question)


if __name__ == "__main__":
    raise SystemExit(main())
