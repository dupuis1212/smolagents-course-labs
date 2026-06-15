"""Quill's data tools — the frozen contract (06-FIL-ROUGE-SPEC §2).

Three reusable, validated, documented capabilities the agent CALLS instead of
re-inventing pandas/matplotlib boilerplate on every run:

- ``load_dataset``      — load a CSV/Parquet file, print a summary, return it as text.
- ``profile_dataframe`` — schema, dtypes, describe(), and missing-value counts as text.
- ``save_chart``        — a ``Tool`` subclass that boots matplotlib lazily in ``setup()``
                          and saves the current figure to ``outputs/``, returning its path.

Two engineering rules are applied here on purpose:

1. **"pushable" rules (Module 9, hardened for the agent-level push of Module 13).** Every
   import lives INSIDE the function/method, ``save_chart.__init__`` takes no argument other
   than ``self``, AND each ``@tool`` function is SELF-CONTAINED — its body references no
   module-level helper. The last point is the real friction Module 13 surfaced: when you push
   the WHOLE agent, ``Tool.to_dict``/``_get_tool_code`` serialise each tool by reading ONLY
   its own function source, so a ``@tool`` that called a module-level ``_read_table`` helper
   failed validation with "Name '_read_table' is undefined". So the read-table logic is now
   INLINED into ``load_dataset`` and ``profile_dataframe`` (their FROZEN M3 signatures, the
   ``print`` summaries and the ``ValueError`` messages are byte-for-byte unchanged — only the
   single helper call became an inline body). The stand-alone ``_read_table`` stays for
   non-tool callers and tests, but NO ``@tool`` depends on it — that is what lets M13 do
   ``agent.push_to_hub`` / ``agent.save`` over the whole Quill without a rewrite.

2. **Good-tool habits (the "write better tools" principle — Module 7, T3.12).** The three
   golden rules are applied here and the docstrings are SHARPENED in M7 (their FROZEN M3
   signatures are untouched — only the prose, the ``print()`` summaries and the ``ValueError``
   messages changed):
   - **A precise docstring.** The ``description``/``Args:`` is injected into the system prompt
     — it IS the interface the model reads — so it names the supported formats, the expected
     date format ``'%Y-%m'`` for this dataset, an example, and EXACTLY what the tool returns.
   - **Print what helps the LLM.** Each tool ``print()``s a readable one-line summary (rows ×
     cols, dtypes, total missing), never a raw multi-kB dump, so the ``Observation`` the model
     reasons over is signal, not noise.
   - **Raise informative ``ValueError``s.** On a bad path or unsupported format the tool raises
     a ``ValueError`` whose message tells the agent how to fix it, so it self-corrects on the
     next step instead of crashing the run.
**Module 15 change (idempotence — by ADDITION, the FROZEN M3 signature is untouched).** A
production run that retries (a network blip, a UI re-submit) must not double-WRITE charts: two
identical runs should overwrite the same file, not accumulate ``chart-<timestamp>.png`` litter.
We add a module-level **run signature** (``set_run_signature`` / ``clear_run_signature``) that
``save_chart`` reads ONLY when the agent calls it WITHOUT a ``filename`` — instead of the
timestamp auto-name it then derives a deterministic stem from the active signature, so a re-run of
the same (question, dataset) writes ``outputs/quill-<sig>.png`` every time. This is a pure
ADDITION: the tool ``name``, ``inputs``, ``output_type``, ``description`` and the explicit-filename
behaviour are byte-for-byte unchanged; an agent that passes an explicit ``filename`` (and every
prior test) is completely unaffected. ``quill/runtime.py`` owns the signature derivation
(``run_signature``); ``quill/agent.py`` sets it around a run.
"""
from __future__ import annotations

from smolagents import Tool, tool

# Module 15 (idempotence): the env var that carries the ACTIVE run signature, set by quill/agent.py
# around a run. When present, save_chart() — called WITHOUT an explicit filename — auto-names
# deterministically as `quill-<signature>` instead of `chart-<timestamp>`, so a re-run of the same
# (question, dataset) OVERWRITES the same PNG rather than accumulating files. We use an ENV var (not
# a module global) so save_chart stays fully SELF-CONTAINED and pushable (M9/M13): the serialized
# tool reads os.environ, with no reference back into quill.tools.data.
RUN_SIGNATURE_ENV = "QUILL_RUN_SIGNATURE"


def set_run_signature(signature: str | None) -> None:
    """Set the active run signature for deterministic, idempotent ``save_chart`` auto-naming (M15).

    ``quill/agent.py`` calls this with ``runtime.run_signature(question, dataset)`` before a run and
    clears it after. ONLY affects ``save_chart`` calls that pass NO ``filename`` — the FROZEN tool
    signature is untouched. Stored in ``QUILL_RUN_SIGNATURE`` so the (possibly serialized) tool reads
    it from the environment. Pass ``None`` (or empty) to clear.
    """
    import os

    if signature:
        os.environ[RUN_SIGNATURE_ENV] = signature
    else:
        os.environ.pop(RUN_SIGNATURE_ENV, None)


def clear_run_signature() -> None:
    """Clear the active run signature (M15) — back to the timestamp auto-name for un-named charts."""
    import os

    os.environ.pop(RUN_SIGNATURE_ENV, None)


def _read_table(path: str):
    """Read a CSV or Parquet file into a DataFrame, with an informative error on failure.

    Imports live inside the function (pushable rule). Raises ``ValueError`` (not a bare
    crash) so the agent sees a readable message and can correct the path/format itself.
    """
    import os

    import pandas as pd

    if not os.path.exists(path):
        raise ValueError(
            f"No file at {path!r}. The path is relative to the current working directory; "
            "list the data/ folder, fix the path, then call this tool again."
        )
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported format for {path!r}. Supported extensions: .csv, .parquet, .pq. "
        "Convert the file to one of these, or point me at a supported one."
    )


@tool
def load_dataset(path: str) -> str:
    """Load a tabular dataset (CSV or Parquet) and return a short text summary of it.

    Call this ONCE, before writing any analysis code, to discover a dataset's shape and exact
    column names so you never have to guess the schema. Do not call it again for a file you have
    already loaded. For a full schema with dtypes, statistics and missing-value counts, use
    ``profile_dataframe`` instead.

    Returns a one-line summary string (also printed as the observation), e.g.
    ``"Loaded data/sales.csv: 108 rows x 6 columns. Columns: ['month', 'region_code',
    'category', 'units', 'net_rev', 'churn_flag']."`` For this dataset, ``month`` is a string in
    ``'%Y-%m'`` format (e.g. ``'2025-01'``); parse it with ``pd.to_datetime(df['month'])`` if
    you need date arithmetic.

    Args:
        path: Filesystem path to the dataset, relative to the working directory; must end in
            ``.csv``, ``.parquet`` or ``.pq`` (e.g. ``"data/sales.csv"``). Raises ``ValueError``
            with a fix-it message if the file is missing or the extension is unsupported.
    """
    # M13 pushable rule: the read-table body is INLINED (no module-level _read_table call) so
    # Tool.to_dict can serialise this @tool from its own source for agent.push_to_hub/save.
    import os

    import pandas as pd

    if not os.path.exists(path):
        raise ValueError(
            f"No file at {path!r}. The path is relative to the current working directory; "
            "list the data/ folder, fix the path, then call this tool again."
        )
    lower = path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower.endswith(".parquet") or lower.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported format for {path!r}. Supported extensions: .csv, .parquet, .pq. "
            "Convert the file to one of these, or point me at a supported one."
        )
    summary = (
        f"Loaded {path}: {df.shape[0]} rows x {df.shape[1]} columns. "
        f"Columns: {list(df.columns)}."
    )
    print(summary)
    return summary


@tool
def profile_dataframe(path: str) -> str:
    """Profile a tabular dataset in ONE call: schema, dtypes, statistics, and missing values.

    Call this ONCE at the start of an analysis to understand a dataset fully before writing any
    code — it returns, in a single observation, every column's dtype, ``describe()`` statistics
    for the numeric columns, and the per-column missing-value counts. Calling it once replaces a
    column-by-column inspection loop (the "reduce LLM calls" principle), so do NOT re-profile a
    file you have already profiled in this run.

    Returns a multi-line profile string (also printed as a one-line summary), e.g. a block
    starting ``"Profile of data/sales.csv\\nShape: 108 rows x 6 columns\\nDtypes:\\n  - month:
    object\\n  ..."``. For this dataset ``month`` is a ``'%Y-%m'`` string and ``net_rev`` /
    ``units`` / ``churn_flag`` are numeric.

    Args:
        path: Filesystem path to the dataset, relative to the working directory; must end in
            ``.csv``, ``.parquet`` or ``.pq`` (e.g. ``"data/sales.csv"``). Raises ``ValueError``
            with a fix-it message if the file is missing or the extension is unsupported.
    """
    # M13 pushable rule: the read-table body is INLINED (no module-level _read_table call) so
    # Tool.to_dict can serialise this @tool from its own source for agent.push_to_hub/save.
    import os

    import pandas as pd

    if not os.path.exists(path):
        raise ValueError(
            f"No file at {path!r}. The path is relative to the current working directory; "
            "list the data/ folder, fix the path, then call this tool again."
        )
    lower = path.lower()
    if lower.endswith(".csv"):
        df = pd.read_csv(path)
    elif lower.endswith(".parquet") or lower.endswith(".pq"):
        df = pd.read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported format for {path!r}. Supported extensions: .csv, .parquet, .pq. "
            "Convert the file to one of these, or point me at a supported one."
        )

    # M13 pushable rule: LIST comprehensions (not bare generator expressions) inside the join —
    # smolagents' Tool.to_dict static validator registers list/dict/set-comp targets but not
    # GeneratorExp targets, so a `join(... for ...)` would fail with "Name 'col' is undefined".
    # `str.join` over a list is behaviourally identical; the output is byte-for-byte unchanged.
    dtypes = "\n".join([f"  - {col}: {dtype}" for col, dtype in df.dtypes.items()])
    missing = df.isna().sum()
    missing_lines = "\n".join(
        [f"  - {col}: {int(count)}" for col, count in missing.items() if count]
    ) or "  (none)"

    numeric = df.select_dtypes("number")
    stats = numeric.describe().to_string() if not numeric.empty else "(no numeric columns)"

    profile = (
        f"Profile of {path}\n"
        f"Shape: {df.shape[0]} rows x {df.shape[1]} columns\n"
        f"Dtypes:\n{dtypes}\n"
        f"Missing values:\n{missing_lines}\n"
        f"Numeric summary:\n{stats}"
    )
    print(f"Profiled {path}: {df.shape[0]}x{df.shape[1]}, "
          f"{int(missing.sum())} missing values total.")
    return profile


class save_chart(Tool):  # noqa: N801 — canonical tool name is `save_chart` (06 §2)
    """Save the current matplotlib figure to ``outputs/`` and return its file path.

    Implemented as a ``Tool`` SUBCLASS (not ``@tool``) because it needs an expensive,
    one-time initialization — selecting matplotlib's non-interactive "Agg" backend — that
    must happen lazily in ``setup()``, not when the tool is merely constructed. The path it
    returns feeds ``chart_paths`` of the future ``QuillReport`` (Module 8).

    Pushable rules (Module 9): ``__init__`` takes no argument other than ``self`` (we use
    the base one), and every import is inside a method.
    """

    name = "save_chart"
    description = (
        "Save the CURRENT matplotlib figure (the one you just drew) to the outputs/ "
        "directory as a PNG and RETURN its saved file path as a string, e.g. "
        "'outputs/category_revenue.png'. Draw your chart FIRST with matplotlib "
        "(e.g. df.plot(kind='bar') or plt.plot(...)), then call this — do NOT use plt.show() "
        "(it saves nothing and the path you need is the return value of THIS tool). Pass an "
        "optional base filename like 'category_revenue'; omit it to auto-name with a "
        "timestamp. Raises ValueError if no figure has been drawn yet."
    )
    inputs = {
        "filename": {
            "type": "string",
            "description": "Optional base filename WITHOUT a directory, e.g. "
                           "'category_revenue' (saved under outputs/). '.png' is appended if "
                           "missing. Omit to auto-name with a timestamp.",
            "nullable": True,
        }
    }
    output_type = "string"

    def setup(self) -> None:
        """Lazy, one-time init: force the non-interactive 'Agg' backend.

        Runs on the FIRST call only (smolagents calls ``setup()`` when
        ``not self.is_initialized``). 'Agg' renders to a file with no display, which is
        what you want on a server/in a sandbox. ``super().setup()`` flips
        ``is_initialized`` so this never runs twice.
        """
        import matplotlib

        matplotlib.use("Agg")  # non-interactive: write PNGs, never open a window
        super().setup()

    def forward(self, filename: str | None = None) -> str:
        """Save the current figure to ``outputs/<filename>.png`` and return the path."""
        import datetime
        import os

        import matplotlib.pyplot as plt

        os.makedirs("outputs", exist_ok=True)

        if not filename:
            # M15 idempotence (by ADDITION): if a run signature is active in the environment (set by
            # quill/agent.py around a run, via QUILL_RUN_SIGNATURE), auto-name deterministically as
            # `quill-<sig>` so a re-run of the same (question, dataset) OVERWRITES the same PNG.
            # Reading the env keeps save_chart fully SELF-CONTAINED (pushable, M9/M13 — no
            # module-level reference). Otherwise keep the original timestamp auto-name, so nothing
            # changes for callers that did not opt in. An EXPLICIT filename always wins (the FROZEN
            # behaviour is untouched).
            run_signature = os.environ.get("QUILL_RUN_SIGNATURE")
            if run_signature:
                filename = f"quill-{run_signature}"
            else:
                filename = f"chart-{datetime.datetime.now():%Y%m%d-%H%M%S-%f}"
        if not filename.endswith(".png"):
            filename = f"{filename}.png"

        out_path = os.path.join("outputs", filename)
        fig = plt.gcf()
        if not fig.get_axes():
            raise ValueError(
                "No figure to save — draw a chart with matplotlib (e.g. df.plot(...) or "
                "plt.plot(...)) BEFORE calling save_chart."
            )
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved chart to {out_path}")
        return out_path
