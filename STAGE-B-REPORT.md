# Stage B — independent QA + polish pass (Quill labs + articles, modules 01–15)

Date: 2026-06-15
Scope: an independent reviewer per module audited each article
(`cours/smolagents/content/module-NN.md`) against its lab (`smolagents-course-labs/module-NN/`)
and the **installed smolagents 1.26.0 source** (not just the spec) for: API correctness,
article↔lab snippet fidelity, module-to-module continuity, and dev-grade quality.

## Suite result

```
1891 passed, 126 skipped, 0 failed   (offline, ~57s)   exit 0
```

Up from Stage A's 1881 — the +10 are a new reset=False regression test added to the prune
suite of every module that ships it (M06–M15). 126 skipped = `live` + `sandbox` tests, gated
as designed.

## Verdicts

SHIP-READY out of the box: **M02, M04, M05, M07, M12, M13, M14, M15**.
Fixed this pass: **M01, M03, M06, M08, M09, M10, M11**.

## The one real bug (HIGH) — fixed

`prune_old_observations` (the M06 headline callback) keyed step age on `step_number`. But
`run(reset=False)` restarts `step_number` at 1 each turn, so a previous turn's big DataFrame
dumps were **never** pruned — i.e. the callback failed in the exact long multi-turn session it
exists for. The synthetic unit tests faked the age-gap via `step_number` without populating
`memory.steps`, so they masked it.

Fix: prune by **list position** in `agent.memory.steps` (globally monotonic across turns), keep
the last `KEEP_LAST` action steps. Propagated to the cumulative copies in M06–M15, plus the
twin `prune_old_screenshots` (M11–M15), the misleading source comment, the M06 article snippet +
field note, and an added regression test (`test_prune_works_across_a_reset_false_turn_boundary`)
that fails under the old logic.

## Article factual corrections (MEDIUM) — fixed, verified against source

- **M08** — "only `AgentMaxStepsError` ends a run" was wrong; `AgentGenerationError` is re-raised
  and also ends it (agents.py:594). Corrected in 3 places.
- **M03** — `WebSearchTool`'s default engine was said to need `ddgs`; it uses `requests` +
  stdlib (default_tools.py:374). `ddgs` belongs to the separate `DuckDuckGoSearchTool`. Fixed in
  the article (table + prose + pitfall), both lab spots, the M04–M09 agent.py comments, and the
  spec §4. Also softened the `validate_arguments` claim and the docstring→`description` mapping.
- **M11** — images were said to "vanish silently" with `flatten_messages_as_text=True`; the
  source raises a loud `AssertionError` (models.py:364, verified empirically). Corrected in 2
  places. (The `additional_args` silent case is genuinely silent — left as is.)
- **M09** — `structured_output` default shown as `False`; the signature default is `None` (emits
  a `FutureWarning`, then falls back to `False`). Article now matches its own lab; both signatures
  fixed plus `from_space(description="")`.
- **M01** — setup described a per-module `pyproject.toml`/`uv.lock` that don't exist (pins +
  lock are at the repo root, one shared venv). Reworded; restored the `__main__` entry-point
  guard in the snippet.

## Minor — fixed

- **M10** — `team.py` comment falsely claimed a `ToolCallingAgent` includes `python_interpreter`
  (it doesn't); stale "160 passed" → real count.
- **M13** — `agent.interrupt()` was presented as a verbatim docstring quote (it isn't); reworded
  to describe the real flag-checked-at-step-boundary behavior (behavior was already correct).
- **M15** — the E2B Approach-2 path uploaded the dataset but not `data/corpus/`, so the M12
  RetrieverTool (on by default) would index an empty corpus inside an E2B sandbox. Now ships all
  of `data/` as bytes, mirroring the Docker path. (Docker path was already correct; E2B is the
  documented optional path, untested by the sandbox suite.)
- **M08** — stale `pyproject.toml` package name `...module-07` → `...module-08`.
- Article pass-count citations reconciled to actuals (M08 124→125, M10 →165); README total →1891.

## Residual / not changed (by design)

- `live` (real LLM) and `sandbox` (Docker/E2B) tests remain off by default — the offline suite is
  the green guarantee. The E2B corpus fix is not exercised offline (no E2B test); it mirrors the
  Docker path which is covered.
- A few LOW notes were judged non-defects on verification (e.g. `base_url`/`model_id` mutual
  exclusion lives in `huggingface_hub`; Phoenix `serve` subcommand is version-sensitive) and left
  with their existing "as of 1.26.0" / re-verify framing.
