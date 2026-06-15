# The smolagents production checklist (Quill v1.0)

A reusable, copy-pasteable checklist for shipping a smolagents agent you can defend. Every item
links to the module of the *smolagents Mastery* course that owns it, so a failing item tells you
exactly what to re-read. Verified against **smolagents 1.26.0** (latest at build time).

> The capstone's core claim: "shipping" is **not** "wrap it in a UI and push a Space". A UI on an
> un-isolated, un-hardened, un-measured system is a demo exposed to the internet. Shipping is
> isolation (Approach 2) + guard-rails (timeouts/caps/retries) + observability (telemetry) + a
> defensible quality signal (the eval gate). The UI is the LAST layer, not the first.

## Security / sandbox

- [ ] **Multi-agent isolation uses Approach 2.** Running a remote `executor_type` (`docker`/`e2b`)
      together with `managed_agents` raises `Exception("Managed agents are not yet supported with
      remote code execution.")`. To isolate a multi-agent system you run the WHOLE system *inside*
      the sandbox you create by hand (Approach 2), not `executor_type="docker"` (Approach 1). — M5, M15
- [ ] **`local` is not a sandbox.** The `LocalPythonExecutor` is an AST allow-list — "it is not a
      security sandbox". Use a remote executor (Approach 1) or Approach 2 for any untrusted
      input/model or a publicly exposed agent. — M5
- [ ] **`additional_authorized_imports` is minimal — never `"*"`.** Quill locks it to
      `["pandas", "numpy", "matplotlib.*", "json", "statistics"]` and extends it only by explicit
      addition. The `"*"` wildcard authorizes every import ("use at your own risk"). — M5
- [ ] **Docker hardening flags set** (Approach 2 / a remote Docker executor): `mem_limit="512m"`,
      `cpu_quota=50000`, `pids_limit=100`, `security_opt=["no-new-privileges"]`,
      `cap_drop=["ALL"]`, and run the code as `USER nobody`. — M5, M15
- [ ] **Secrets in env, never in source.** `HF_TOKEN` / provider keys / `E2B_API_KEY` come from the
      environment (`.env`, never committed). Approach 2 passes `HF_TOKEN` to the sandbox as a
      container env var. — M4, M5
- [ ] **`trust_remote_code` audited.** A stdio MCP server runs code on your machine; a Hub
      `from_hub` agent deserializes code. Set `trust_remote_code=True` only for sources you trust as
      much as your own code, and pin `structured_output=False` explicitly (the MCP default will
      flip). — M9, M13
- [ ] **No `executor_type="wasm"` / `WasmExecutor`** — removed in 1.26.0. Valid values:
      `{"local", "docker", "e2b", "modal", "blaxel"}`. — M5

## Reliability

- [ ] **`final_answer_checks` enforce content.** 3-arg `(final_answer, memory, agent)` validators
      reject a half-finished answer (no chart; a web claim with no source). A `False`/raise loops the
      agent (the `AgentError` lands in `ActionStep.error`) — it self-corrects, it does not crash. — M8
- [ ] **`max_steps` bounded** on the manager AND every sub-agent (Quill: manager 8, `web_researcher`
      10). An agent that does not converge raises `AgentMaxStepsError` instead of looping. — M8, M10, M15
- [ ] **Retries capped at 2.** Beyond two retries you are burning tokens on a broken loop. Use a
      bounded retry helper (`run_with_bounded_retries`) that re-raises the last error — never a silent
      `try/except`. — M15
- [ ] **Idempotent outputs.** A re-run of the same (question, dataset) overwrites the same chart path
      (`outputs/quill-<sig>.png`) instead of accumulating files, and the outputs dir is created with
      `exist_ok=True` — so retries and UI re-submits do not double-write or double-count. — M15
- [ ] **Resource cleanup is deterministic.** Use `with build_quill() as agent:` (or `agent.cleanup()`)
      for a remote executor; close the hand-made Approach-2 sandbox in a `try/finally`. No dangling
      container/microVM, even on an exception. — M5, M15
- [ ] **Errors handled, not swallowed.** Catch `AgentError` where you mean to; never a bare silent
      `except`. — M8

## Cost / performance

- [ ] **An explicit `model_id`.** Pin it via `make_model` (`Qwen/Qwen2.5-Coder-32B-Instruct` for
      Quill); do not rely on `InferenceClientModel`'s default (`Qwen/Qwen3-Next-80B-A3B-Thinking`,
      "subject to change" as of smolagents 1.26.0). — M4
- [ ] **`planning_interval` on longer jobs.** A periodic `PlanningStep` re-centres the agent and cuts
      redundant exploration — at the cost of one extra LLM call, so reserve it for multi-step work. — M7
- [ ] **Memory pruned.** `step_callbacks` prune stale big observations so they are not re-sent every
      step (context engineering). — M6
- [ ] **Cost/run visible.** Read tokens via `Monitor.get_total_token_counts()` /
      `chat_message.token_usage` — never the per-agent token attributes removed in 1.21.0. The HF free
      tier is ~$0.10/month (as of smolagents 1.26.0, subject to change); a multi-agent + vision + RAG
      run can be dozens of LLM calls. — M4, M14
- [ ] **Vision only when it pays.** A VLM call (chart self-review, the `vision_browser`) costs far
      more than a text call — use it as a fallback, not the main path. — M11

## Observability / quality

- [ ] **`SmolagentsInstrumentor().instrument()` runs BEFORE the agent is built.** Instrument after a
      run is under way and the first steps' spans are lost. Import it from
      `openinference.instrumentation.smolagents` (NOT `from smolagents import ...`). — M14
- [ ] **In-process inspection without a backend:** `agent.memory.steps`, `agent.replay()`,
      `agent.visualize()`, `agent.memory.get_full_steps()` — **never `agent.logs`** (removed in
      1.21.0). — M6, M15
- [ ] **An eval golden set + LLM-as-judge.** Score every `QuillReport` against frozen expected points
      with a SEPARATE judge model (never self-grading); compute `citations` deterministically. — M14
- [ ] **A regression gate ("green or no ship").** `run_evals.py` exits non-zero when TSR drops below
      the floor or cost/run exceeds the budget. Gate the release on it before tagging. — M14, M15

## Deploy

- [ ] **`GradioUI` wraps the hardened agent** (`file_upload_folder`, `reset_agent_memory=False` for
      multi-turn) — built from `build_quill()`, it does not reconstruct the agent. — M13
- [ ] **`push_to_hub` exports the right artefacts** (`agent.json`, `prompts.yaml`, `tools/`,
      `managed_agents/`, `app.py`, `requirements.txt`); every tool is self-contained (pushable). — M9, M13
- [ ] **`uv.lock` committed and pins exact.** `smolagents[...]==1.26.0`, `huggingface_hub>=1.0,<2`,
      `pandas>=2.2.3`, reproducible env. — M1, M15
- [ ] **Tag the release** (`git tag v1.0`) only when the eval gate is green and the sandboxed run
      works. — M15

## When NOT to ship an agent at all

- [ ] **Could a deterministic workflow do this?** smolagents' own guidance: "regularize towards not
      using agentic behaviour"; Anthropic: "the best agentic systems are the simplest". Use an agent
      only when the task is genuinely not determinisable in advance (Quill's open-ended data analysis
      qualifies — a fixed pipeline does not). — M2, M7

---

*Try it yourself: add a new line (e.g. "rate-limit per user") to a section above, then write the
`smoke_test.py` assertion that verifies it — closing the loop between the checklist and the tests.*
