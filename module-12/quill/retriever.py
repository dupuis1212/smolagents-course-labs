"""Quill's knowledge base — the ``RetrieverTool`` (agentic RAG, Module 12).

Until now, when Quill met an ambiguous column (`net_rev`, `churn_flag`, `region_code`) it
*guessed* the meaning from the name and answered with confidence — no source, no guarantee. This
module gives Quill a place to LOOK THINGS UP instead of inventing them: a ``RetrieverTool`` over a
small corpus of business docs (`data/corpus/*.md` — a data dictionary plus revenue/segmentation
rules). Quill calls it when (and only when) it needs to, can REFORMULATE its query and retrieve
again, and CITES the passage it used in ``QuillReport.sources``.

**Agentic RAG, not a fixed pipeline.** The retriever is a *tool in the agent's toolbox* — the
agent DECIDES to call it (`docs = retriever("net_rev definition")`), reads the passages, and may
rewrite the query and call again if they miss. That is the opposite of a fixed
`query → retrieve top-k → stuff prompt → generate` pipeline that retrieves once, always, before
the model ever runs. The agent's code-as-action loop makes iterative query reformulation
(HyDE/self-query) EMERGE for free — those are behaviours, not smolagents APIs.

**It is just a ``Tool`` (the M3 contract).** ``RetrieverTool`` subclasses ``smolagents.Tool`` with
the standard class attributes (``name``/``description``/``inputs``/``output_type``), builds its
BM25 index ONCE in ``setup()`` (lazy init — run on the first ``__call__``, reused on every
``forward``), and exposes ``forward(self, query: str) -> str`` whose ``query`` parameter matches
the single key of ``inputs`` (validated by smolagents). Same shape, same lazy-init discipline as
``save_chart`` (M3).

**BM25, not embeddings (06 §4).** We ship lexical retrieval via ``rank-bm25`` (``BM25Okapi``):
zero model, zero inference cost, deterministic — it ranks docs by word overlap. It is the right
default for a small, homogeneous corpus like a data dictionary. It MISSES synonyms (a user asking
"customer loss" will not match a doc that says "churn"); that is when production swaps in
**embeddings** (semantic search). "BM25 for the demo, embeddings for prod" — not the reverse.

**Pushable by construction (06 §2, research-02 §8).** This is deliberate so Module 13 can
``agent.push_to_hub(...)`` the WHOLE Quill (tools included) without a rewrite: every import lives
INSIDE a method, and ``__init__`` takes NO argument beyond ``self`` (the corpus comes from the
CLASS attribute ``corpus_dir``, read in ``setup()``). The official smolagents RAG example passes
``docs`` to ``__init__`` and uses ``langchain_community.BM25Retriever``; we deviate on both —
no constructor args (pushable) and no LangChain dependency (``rank-bm25`` directly).

Grounding (Module 8 contract, reused — no new field): each retrieved passage embeds its ``title``
and ``url`` so the agent can map a used definition to a ``Source(url=..., title=...)`` in
``QuillReport.sources`` and cite it ``[n]`` in the matching finding. The corpus sources and the
web sources (M10) live in the SAME ``sources`` list.

Verified against smolagents 1.26.0. NOTE: this retriever does NOT call an LLM — BM25 is purely
lexical, so the whole retrieval path is testable fully offline.
"""
from __future__ import annotations

from smolagents import Tool

# Where the corpus lives, relative to the working directory Quill is run from (the module-12/
# root). It is a CLASS attribute on RetrieverTool (below) — NOT an __init__ argument — so the
# tool stays pushable; this constant is exported for the lab/tests to reference the same default.
DEFAULT_CORPUS_DIR = "data/corpus"

# How many passages forward() returns by default (the BM25 top-k). Bounded on purpose: every
# retrieved passage is injected into the agent's context as tokens (the T7 production note), so a
# small k keeps the context — and the cost — under control. The agent reformulates rather than
# asking for a huge k.
DEFAULT_K = 5


def load_corpus(corpus_dir: str = DEFAULT_CORPUS_DIR) -> list[dict]:
    """Read ``corpus_dir/*.md`` into a list of ``{"title", "url", "text"}`` dicts (Module 12).

    A small pure helper (no smolagents, no index) so the corpus loading is testable on its own and
    reusable by an embeddings swap (the "Try it yourself"). Each Markdown file becomes one document:

    - ``title``: the first ``# Heading`` line if present, else the file stem — what a reader sees
      in a citation.
    - ``url``: the file's POSIX path (a logical, citable location). In production this would be the
      doc's canonical URL; here the path IS the source identifier.
    - ``text``: the full file contents, used both for BM25 scoring and for the returned passage.

    Args:
        corpus_dir: directory of ``.md`` docs (default ``data/corpus``).

    Returns:
        A list of document dicts, sorted by filename for a deterministic order/citation numbering.
    """
    # Imported here (not at module top) to keep RetrieverTool pushable: the same discipline lets
    # setup() reuse this without dragging a top-level import into the saved tool.
    import pathlib

    docs: list[dict] = []
    for path in sorted(pathlib.Path(corpus_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # Title = the first Markdown heading if the file opens with one, else the file stem.
        first_line = text.lstrip().splitlines()[0] if text.strip() else path.stem
        title = first_line.lstrip("# ").strip() if first_line.startswith("#") else path.stem
        docs.append({"title": title, "url": path.as_posix(), "text": text})
    return docs


class RetrieverTool(Tool):
    """A BM25 retriever over Quill's docs corpus — a standard ``Tool`` (Module 12, T11.7).

    The agent calls it like any tool from inside the Python it writes::

        passages = retriever("net_rev definition")

    and gets back the top-``k`` corpus passages, each prefixed with its ``title`` and ``url`` so the
    agent can cite it. The index is built ONCE in ``setup()`` (lazy init) and reused on every
    ``forward`` — never rebuilt per query.

    The class attributes below ARE the tool contract (M3): smolagents bakes ``name``/``description``/
    ``inputs`` into the system prompt at init, so the ``description`` is what teaches the agent WHEN
    and HOW to call this (note the "affirmative form" hint — the official RAG example's wording).
    """

    name = "retriever"
    description = (
        "Looks up the project's data dictionary and domain docs to explain what a dataset column "
        "means or how a metric is defined (e.g. net_rev, churn_flag, region_code, growth). Use it "
        "BEFORE interpreting any ambiguous column instead of guessing from the name, and cite the "
        "returned source. Use the affirmative form, e.g. 'net_rev definition' rather than a "
        "question. If the passages do not answer you, REWRITE the query and call again."
    )
    inputs = {
        "query": {
            "type": "string",
            "description": "What to look up in the docs (affirmative form, e.g. 'net_rev definition').",
        }
    }
    output_type = "string"

    # CLASS attributes — NOT __init__ args (06 §2 pushable contract). The corpus location and the
    # top-k live here so __init__ stays argument-free and the tool can be pushed to the Hub whole.
    corpus_dir = DEFAULT_CORPUS_DIR
    k = DEFAULT_K

    def setup(self) -> None:
        """Build the BM25 index ONCE, lazily, on the first call (Module 12, T3.4 / T11.7).

        ``Tool.__call__`` runs ``setup()`` the FIRST time the tool is invoked (guarded by
        ``is_initialized``) and never again, so the index is built once and reused on every
        ``forward`` — building it in ``forward`` instead would re-tokenise the whole corpus on every
        query (slow, wasteful). All imports are INSIDE this method (the pushable rule): the saved
        tool carries no top-level ``rank_bm25`` import.

        We tokenise each doc by lowercasing and splitting on whitespace — a plain lexical
        tokeniser, which is exactly what BM25 (a bag-of-words model) wants. ``self.documents`` keeps
        the ``{title, url, text}`` dicts so ``forward`` can return citable passages.

        The final ``super().setup()`` flips ``is_initialized`` to ``True`` — THE point of the
        module: without it, ``Tool.__call__`` would run ``setup()`` (rebuilding the BM25 index) on
        EVERY ``retriever("...")`` call instead of once. Build the index here, reuse it in
        ``forward``.
        """
        from rank_bm25 import BM25Okapi

        self.documents = load_corpus(self.corpus_dir)
        # Tokenised corpus for BM25 (one token list per doc). Lowercase so the match is
        # case-insensitive; whitespace split is the simple lexical tokeniser BM25 expects.
        self._tokenized = [doc["text"].lower().split() for doc in self.documents]
        # Guard the empty-corpus case: BM25Okapi divides by the corpus size, so it raises
        # ZeroDivisionError on an empty list. We keep index=None and let forward() return a clear
        # "no docs" message instead of crashing the agent loop on a missing/empty corpus.
        self.index = BM25Okapi(self._tokenized) if self._tokenized else None
        # Mark the tool initialised so the lazy-init guard in Tool.__call__ never rebuilds the
        # index. (Tool.setup() does exactly this; we call super() so the contract is honoured.)
        super().setup()

    def forward(self, query: str) -> str:
        """Return the top-``k`` corpus passages for ``query``, each with its ``title`` + ``url``.

        ``query`` matches the single key of ``inputs`` (smolagents validates this). We score every
        doc with the prebuilt BM25 index, take the ``k`` highest, and format each as::

            ===== [n] <title> (<url>) =====
            <full passage text>

        The ``title`` + ``url`` are embedded ON PURPOSE: that is what lets the agent map a used
        definition to a ``Source(url, title)`` and cite it ``[n]`` in ``QuillReport`` (grounding,
        H2-4). The ``[n]`` here is a passage index within THIS result, a hint for the agent — the
        report's final citation numbering is decided when the agent assembles ``sources``.

        Args:
            query: what to look up (affirmative form works best — see ``description``).

        Returns:
            A formatted string of the top-``k`` passages, or a clear "no docs" message if the
            corpus is empty (so the agent can reformulate or fall back rather than crash).
        """
        if not self.documents:
            return (
                f"No documents found in the corpus at {self.corpus_dir!r}. "
                "There is nothing to retrieve — answer from the data alone and note the missing "
                "data dictionary as a caveat."
            )

        scores = self.index.get_scores(query.lower().split())
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top = ranked[: self.k]
        passages = []
        for n, doc_index in enumerate(top, start=1):
            doc = self.documents[doc_index]
            passages.append(
                f"===== [{n}] {doc['title']} ({doc['url']}) =====\n{doc['text']}"
            )
        return "\n\n".join(passages)


__all__ = [
    "DEFAULT_CORPUS_DIR",
    "DEFAULT_K",
    "RetrieverTool",
    "load_corpus",
]
