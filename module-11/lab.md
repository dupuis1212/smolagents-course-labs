# Lab 11 — Quill gets eyes: re-reading its own charts (and, optionally, the web)

**Goal:** make Quill **re-read the charts it just saved** with a VLM (`run(images=[...])`) as a
visual sanity-check, and — optionally — give it a `vision_browser` sub-agent that screenshots
JS-heavy pages into its own memory via a `step_callback`. The verdict from the chart re-read lands
in `QuillReport.caveats` (no new field — the M8 schema is frozen).

**You'll see:** the two image channels (`run(images=...)` the model *looks at* vs `additional_args`
a sandbox *variable*) and why only the first makes the model see pixels; that vision needs a **VLM**
and NO extra (the `[vision]` extra is a browser, not image input); the `save_screenshot`
`step_callback` injecting a PNG into `observations_images` and pruning the old ones; and the
canonical `vision_browser` `CodeAgent` (helium + Chrome) staying `local`.

**Observable result:**

```bash
QUILL_MODEL_ID="Qwen/Qwen2-VL-72B-Instruct" \
  uv run python -m quill "Analyze data/sales.csv and chart monthly revenue, then check the chart yourself." --review
```

```text
[Quill] Backend: hf | Model: Qwen/Qwen2-VL-72B-Instruct
...                        # ← Quill writes pandas, draws matplotlib, save_chart -> outputs/monthly_revenue.png
===== CHART SELF-REVIEW (VLM re-reads the charts via run(images=...)) =====
  Chart review (outputs/monthly_revenue.png): Y-axis starts at 80, which exaggerates the trend — recommend starting at 0.
===== REPORT =====
# Analyze data/sales.csv and chart monthly revenue

## Findings
- Monthly revenue rises steadily through 2025.

## Charts
- `outputs/monthly_revenue.png`

## Caveats
- Chart review (outputs/monthly_revenue.png): Y-axis starts at 80, which exaggerates the trend — recommend starting at 0.
```

---

## Step 1 — Setup

Start from the cumulative Module 10 state (this lab extends it; M11's code still passes every
smoke test for modules ≤ 11).

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib
cp module-11/.env.example module-11/.env   # put your HF token in it
```

**For the chart re-read (the mandatory path): NO extra is required.** Image input depends ONLY on
the model being a **VLM** — point `make_model` at one via the env:

```bash
export QUILL_MODEL_ID="Qwen/Qwen2-VL-72B-Instruct"   # a VLM (re-verify availability on Inference Providers)
# or, via LiteLLM + an OpenAI key:
# export QUILL_MODEL_BACKEND=litellm QUILL_MODEL_ID="gpt-4o"
```

**Do NOT `pip install 'smolagents[vision]'` to re-read a chart.** That extra installs
**helium + selenium** (a browser); it does not enable image input. Install it ONLY for the optional
vision browser in Step 3:

```bash
uv pip install "smolagents[vision]==1.26.0"   # helium + selenium, plus a local Chrome
```

Run from inside `module-11/` so `data/`, `outputs/` and the `quill` package resolve.

## Step 2 — Chart re-read (the required path)

The job: after a run produces a `QuillReport` with `chart_paths`, load each PNG and ask a VLM to
critique it. Two ways to wire it; we recommend (b) for simplicity:

- **(a) a second call (what we ship).** `review_charts(report, model=...)` loads each chart as a PIL
  image and calls a bare reviewer agent via `run(images=[pil_image])`, then appends the verdict to
  `report.caveats`. The image is **vision content the model looks at** — that is the `images=`
  channel, not `additional_args`.
- **(b) a vision tool Quill calls itself (`review_chart`).** Quill could call a tool with the chart
  path. The catch — and the whole point of the module — is that for the *model* to **see** the
  image, it must reach it via `images=`, NOT just as an `additional_args` variable a tool receives.
  Putting the path in `additional_args` only hands the sandbox a string/object; the model never
  sees pixels. So even the tool route has to round-trip the image back through `run(images=...)`.

The reviewer model comes from `make_model` (the frozen M4 contract — ONE place), so a single
`QUILL_MODEL_ID` env var selects the VLM for the whole project. From code:

```python
from quill.agent import build_quill, build_report_task, review_charts

agent = build_quill()
report = agent.run(build_report_task("data/sales.csv", "Chart monthly revenue."))
report = review_charts(report)        # VLM re-reads each chart via run(images=[...])
print(report.to_markdown())           # the verdict is now a caveat
```

The CLI does this for you with `--review` (see the observable result above).

## Step 3 — The optional `vision_browser`

In `quill/team.py`, `build_vision_browser(model) -> CodeAgent` builds a sub-agent named EXACTLY
`vision_browser` (06 §2 canonical name) that drives helium/Chrome and reads screenshots:

```python
CodeAgent(
    tools=[go_back, close_popups, search_item_ctrl_f],   # simple @tool helium helpers
    model=<VLM via make_model>,
    additional_authorized_imports=["helium"],            # its code does `from helium import *`
    step_callbacks=[save_screenshot],                    # the screenshot callback
    max_steps=15,                                         # bounded: a screenshot per step is pricey
)
```

`save_screenshot` (in `quill/callbacks.py`) is the **exact** canonical callback: it screenshots the
page each step into `memory_step.observations_images`, **prunes** the screenshots from steps
`<= current - 2` (the cost guard — a screenshot ≈ hundreds of text tokens), and appends the current
URL to `observations`. Before the first run you preload helium into the sandbox:

```python
agent.python_executor("from helium import *", agent.state)
```

Wire it onto the manager only in `--browse` mode, and keep it `executor_type="local"` — helium needs
a local Chrome, and a remote executor + `managed_agents` raises the Module 10 exception
(`# M15: Approach 2`):

```bash
QUILL_MODEL_ID="Qwen/Qwen2-VL-72B-Instruct" uv run python -m quill "..." --browse
```

## Step 4 — Run + read the trajectory

```bash
uv run python -m quill.agent "Chart monthly revenue, then check the chart yourself." --review
```

`python -m quill.agent` prints the full ReAct trajectory (`agent.replay()` — Module 6), the chart
self-review verdicts, and the rendered report. The trajectory is read via `agent.replay()` /
`agent.memory.steps`, never the removed `agent.logs`.

## Step 5 — Test it

```bash
uv run pytest module-11/tests/                    # offline (no token, no network, no VLM)
QUILL_LIVE_TESTS=1 uv run pytest module-11/tests/ # + the real VLM chart re-read (needs HF_TOKEN + a VLM)
```

The offline tests assert (fake model + a real PNG, no VLM): `review_charts` passes the PNG via
`images=` (the image reaches the model as a vision block, NOT `additional_args`); the final
`QuillReport` gets a non-empty caveat when a chart is reviewed; the `vision_browser` is a `CodeAgent`
named EXACTLY `vision_browser` with `helium` in `additional_authorized_imports` and `save_screenshot`
in `step_callbacks`; and `save_screenshot`/`prune_old_screenshots` prune `observations_images` on the
old steps. The `live` VLM call costs MORE than a text call (one image ≈ hundreds of text tokens —
budget the HF free tier accordingly, $0.10/month as of smolagents 1.26.0); the browser test is
`sandbox`-marked (Chrome/selenium required, skips cleanly otherwise).

## Try it yourself (not graded)

1. Give Quill two versions of the SAME chart — one with a truncated Y-axis (`plt.ylim(80, …)`) and
   one starting at zero — and ask `review_charts` which is misleading. Check the VLM flags the
   truncation.
2. Run the packaged CLI on a chart-heavy page and compare the default model to a HF VLM:
   `webagent "..." --model-type InferenceClientModel --model-id Qwen/Qwen2-VL-72B-Instruct` vs the
   default `gpt-4o` (quality, cost, latency).

## What this lab does NOT do

- No RAG / `RetrieverTool` / corpus citations (Module 12) — vision does not do retrieval.
- No Approach 2 (the `vision_browser` inside a remote sandbox) — it stays `local`; Module 15.
- No telemetry / vision-span traces (Module 14).
- No audio / video (`AgentAudio`, `SpeechToTextTool`).
- No image-upload UI (Module 13) — the image comes from `chart_paths` / a local path.
- No `[vision]` extra for the chart re-read — the required path needs only a VLM (the T11.4 point).

Verified against **smolagents 1.26.0**.
