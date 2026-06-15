# Lab 12 — Quill stops guessing: a knowledge base it retrieves from and cites

**Goal:** give Quill a **`RetrieverTool`** (BM25 over a docs corpus in `data/corpus/`) that it
queries to interpret ambiguous columns, and make it **cite** the retrieved passages in
`QuillReport.sources`. This is **agentic RAG** — the retriever is a tool the agent *decides* to call
(and can reformulate), not a fixed retrieve-then-generate pipeline.

**You'll see:** a retriever built the canonical smolagents way (a `Tool` subclass, index in
`setup()`, `forward(query)` returning citable passages); why the index belongs in `setup()`
(lazy init) and not in `forward`; BM25 (lexical, what we ship) vs embeddings (semantic, prod); the
retriever **pushable by construction** (no `__init__` args, imports inside methods); and the
retrieved `title`+`url` mapped to a `Source` with a `[n]` citation — the same `QuillReport.sources`
field as the web sources (M10), no new field.

**Observable result:**

```bash
uv run python -m quill "What was net_rev growth in data/sales.csv last quarter, and define net_rev." --retrieve
```

```text
[Quill] Backend: hf | Model: Qwen/Qwen2.5-Coder-32B-Instruct
...                        # ← Quill profiles, then calls retriever("net_rev definition")
...                        #   reads the data dictionary passage, computes growth, save_chart
===== REPORT =====
# What was net_rev growth in data/sales.csv last quarter, and define net_rev.

## Findings
- net_rev is net revenue in USD = gross subscription revenue minus refunds [1].
- net_rev grew +X% from the previous quarter.

## Charts
- `outputs/net_rev_growth.png`

## Sources
[1] [Data dictionary — sales.csv](data/corpus/data_dictionary.md)
```

The definition carries a `[1]` that resolves to the corpus doc — Quill **retrieved and sourced** it
instead of guessing.

---

## Step 1 — Setup

Start from the cumulative Module 11 state (this lab extends it; M12's code still passes every smoke
test for modules ≤ 12). Add **`rank-bm25`** (the BM25 lib) — and nothing else; the retriever is a
home-grown `Tool`, there is no smolagents RAG extra.

```bash
uv venv --python 3.11
uv pip install "smolagents[toolkit]==1.26.0" "huggingface_hub>=1.0,<2" "pandas>=2.2.3" matplotlib rank-bm25
cp module-12/.env.example module-12/.env   # put your HF token in it (only needed for a live run)
```

We deliberately do **not** use the official RAG example's `langchain_community.BM25Retriever` — that
drags in LangChain. `rank-bm25` (`BM25Okapi`) is the whole dependency. Run from inside `module-12/`
so `data/`, `data/corpus/`, `outputs/` and the `quill` package resolve.

## Step 2 — Create the corpus (`data/corpus/`)

A small set of citable Markdown business docs. The most important is the **data dictionary**, which
defines the ambiguous columns of `data/sales.csv` (`net_rev`, `region_code`, `churn_flag`). Each doc
opens with a `# Heading` (its `title`) and is cited by its file path (its `url`).

```text
data/corpus/
├── data_dictionary.md     # defines net_rev (= gross minus refunds), region_code, churn_flag, …
├── revenue_policy.md       # revenue-recognition rules (Free-tier net_rev = 0, compare via net_rev)
├── metrics_glossary.md     # growth / QoQ / ARPU / retention definitions
└── segmentation.md         # what a "segment" is; when to use region_code vs category
```

`data/sales.csv` is **unchanged** (the frozen fil-rouge dataset, 06 §2). The corpus is the new,
citable knowledge — not new data.

## Step 3 — Write `quill/retriever.py`

A `RetrieverTool(Tool)` and a small `load_corpus` helper. The whole point is the **`Tool` contract**:
class attributes for `name`/`description`/`inputs`/`output_type`, the BM25 index built **once** in
`setup()`, and `forward(self, query)` whose parameter matches the `inputs` key.

```python
from smolagents import Tool

DEFAULT_CORPUS_DIR, DEFAULT_K = "data/corpus", 5

def load_corpus(corpus_dir=DEFAULT_CORPUS_DIR) -> list[dict]:
    import pathlib
    docs = []
    for p in sorted(pathlib.Path(corpus_dir).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        first = text.lstrip().splitlines()[0] if text.strip() else p.stem
        title = first.lstrip("# ").strip() if first.startswith("#") else p.stem
        docs.append({"title": title, "url": p.as_posix(), "text": text})
    return docs

class RetrieverTool(Tool):
    name = "retriever"
    description = (
        "Looks up the project's data dictionary and domain docs to explain what a column means or "
        "how a metric is defined (net_rev, churn_flag, region_code, growth). Use it BEFORE "
        "interpreting an ambiguous column instead of guessing, and cite the returned source. Use "
        "the affirmative form, e.g. 'net_rev definition'. If it does not answer, rewrite and retry."
    )
    inputs = {"query": {"type": "string", "description": "What to look up in the docs."}}
    output_type = "string"
    corpus_dir = DEFAULT_CORPUS_DIR   # CLASS attribute — NOT an __init__ arg (pushable)
    k = DEFAULT_K

    def setup(self):
        from rank_bm25 import BM25Okapi               # import INSIDE the method (pushable)
        self.documents = load_corpus(self.corpus_dir)
        tokenized = [d["text"].lower().split() for d in self.documents]
        self.index = BM25Okapi(tokenized) if tokenized else None
        super().setup()                               # flips is_initialized — index built ONCE

    def forward(self, query: str) -> str:
        if not self.documents:
            return f"No documents found in the corpus at {self.corpus_dir!r}."
        scores = self.index.get_scores(query.lower().split())
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]
        return "\n\n".join(
            f"===== [{n}] {self.documents[j]['title']} ({self.documents[j]['url']}) =====\n"
            f"{self.documents[j]['text']}"
            for n, j in enumerate(top, start=1)
        )
```

Why `setup()` and not `forward`? `Tool.__call__` runs `setup()` **lazily on the first call**
(`is_initialized`) and never again, while `forward` runs every step — build the index once and reuse
it. (The `super().setup()` is what sets `is_initialized=True`; forget it and you rebuild the index
on every query.) Why no `__init__` args? **Pushable** — Module 13 can `agent.push_to_hub(...)` the
whole agent only if its tools take no constructor args and import inside their methods.

## Step 4 — Wire it into `build_quill` (`quill/agent.py`)

Add the retriever to the manager's own `tools=` list, next to the data tools and `save_chart`. The
manager stays a `CodeAgent`, `managed_agents=[web_researcher]` is unchanged, and `executor_type` is
still read from `QUILL_EXECUTOR` (stays `local`):

```python
from .retriever import RetrieverTool

def build_quill(model=None, *, ..., retrieve: bool = True):
    tools = [load_dataset, profile_dataframe, save_chart()]
    if retrieve:
        tools.append(RetrieverTool())   # Tool subclass — instantiate; setup() runs lazily
    ...
    agent = CodeAgent(tools=tools, managed_agents=[web_researcher], executor_type="local", ...)
```

**Do NOT add `rank_bm25` to `additional_authorized_imports`.** The retriever does its BM25 scoring
inside its own `forward` (it is a `Tool`, run by the framework around the sandbox); the agent's
generated code only ever writes `retriever("...")`. So the frozen least-privilege import lock
(`["pandas","numpy","matplotlib.*","json","statistics"]`) is **untouched** — never the `"*"`
wildcard. (This is the clean alternative the brief calls out: scoring lives in the tool, not in the
agent's sandboxed imports.)

## Step 5 (optional) — Strengthen a `final_answer_check`

The M8 `final_answer_checks` are 3-arg `(final_answer, memory, agent)` validators. You can add one
that refuses a finding which *defines a column* but carries no matching `Source` — Quill then
self-corrects by retrieving and citing. This is a **use** of the M8 check mechanism, not a new
feature; keep it optional (the default checks already enforce a chart + web sources).

## Step 6 — Run + read the trajectory

```bash
uv run python -m quill "What was net_rev growth in data/sales.csv last quarter, and define net_rev." --retrieve
# step-by-step trajectory (the retriever call + any reformulation):
uv run python -m quill.agent data/sales.csv "Define net_rev and report its quarterly growth."
```

`python -m quill.agent` prints the full ReAct trajectory (`agent.replay()` — Module 6) so you can see
Quill call `retriever("net_rev definition")`, read the passage, and (if the first query missed)
rewrite it and retrieve again. Read the trajectory via `agent.replay()` / `agent.memory.steps`,
never the removed `agent.logs`.

## Step 7 — Test it

```bash
uv run pytest module-12/tests/                    # offline (no token, no network, no LLM)
QUILL_LIVE_TESTS=1 uv run pytest module-12/tests/ # + the real-model retrieve-and-cite run (needs HF_TOKEN)
```

The RAG core is **fully offline** — BM25 calls no LLM. The offline tests assert: `RetrieverTool`
follows the `Tool` contract and is **pushable** (no required `__init__` arg); `forward("net_rev
definition")` returns a non-empty string mentioning **refunds** / the data dictionary content with
the corpus **url + title**; the BM25 index is built **once** in `setup()` (lazy) and reused; the
retriever is on the manager by default and does NOT widen the frozen import lock; and an end-to-end
fake-model run where the manager calls `retriever(...)`, then returns a `QuillReport` whose finding
cites `[1]` mapped to a corpus `Source`. The `live` test runs the whole thing with a real model
(skips cleanly without `HF_TOKEN`); budget 5–15 LLM calls. (The RAG path needs no `sandbox` marker
unless you test `QUILL_EXECUTOR=docker`.)

## Try it yourself (not graded)

1. **Prove the RAG is agentic.** Ask Quill a question that touches **no** ambiguous column ("chart
   the first 10 `units` values") and confirm it **never** calls the retriever (a fixed pipeline would
   retrieve regardless). Then ask one about `net_rev` and watch it retrieve.
2. **Swap BM25 → embeddings.** Replace the `BM25Okapi` index in `setup()` with a
   `sentence-transformers` encoder + cosine similarity, and compare on a **synonym** query
   ("customer loss" vs "churn") — BM25 misses it, embeddings catch it. Because `RetrieverTool` is
   already **pushable** (corpus via `corpus_dir`, imports inside `setup()`), the Module 13 Hub push
   keeps working as-is.

## What this lab does NOT do

- No **embeddings / vector store** on the required path — BM25 lexical (embeddings = the option above).
- No **vision** / `run(images=...)` / browser — Module 11 (already acquired), not redone here.
- No **MCP** as the corpus source — the retriever is a home-grown `Tool` (Module 9 for MCP).
- No **`GradioUI`** / corpus upload / Space / CLI `smolagent` — Module 13.
- No **telemetry** / trace of the retriever span — Module 14.
- No **eval** / scoring of the citations — Module 14 (which reuses `QuillReport`).
- No **Approach 2** (retriever inside a remote sandbox) — Quill stays `local`; Module 15.
- No change to `data/sales.csv` or the `QuillReport` schema — frozen contracts.

Verified against **smolagents 1.26.0**.
