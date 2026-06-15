"""Quill's data tools — the frozen contract (06-FIL-ROUGE-SPEC §2).

Three reusable, validated, documented capabilities the agent CALLS instead of
re-inventing pandas/matplotlib boilerplate on every run:

- ``load_dataset``      — load a CSV/Parquet file, print a summary, return it as text.
- ``profile_dataframe`` — schema, dtypes, describe(), and missing-value counts as text.
- ``save_chart``        — a ``Tool`` subclass that boots matplotlib lazily in ``setup()``
                          and saves the current figure to ``outputs/``, returning its path.

Two engineering rules are applied here on purpose:

1. **"pushable" rules (for Module 9).** Every import lives INSIDE the function/method, and
   ``save_chart.__init__`` takes no argument other than ``self``. We do NOT push to the Hub
   yet (that is Module 9) — we just write the tools so M9 can push them with no rewrite.

2. **Good-tool habits (the full theory lands in Module 7).** Each tool has a precise
   docstring (the ``description``/``Args:`` is injected into the system prompt — it IS the
   interface the model reads), ``print()``s a one-line summary so the model gets useful
   ``Observation`` to reason over, and raises informative ``ValueError``s so the agent can
   self-correct instead of crashing the run.
"""
from __future__ import annotations

from smolagents import Tool, tool


def _read_table(path: str):
    """Read a CSV or Parquet file into a DataFrame, with an informative error on failure.

    Imports live inside the function (pushable rule). Raises ``ValueError`` (not a bare
    crash) so the agent sees a readable message and can correct the path/format itself.
    """
    import os

    import pandas as pd

    if not os.path.exists(path):
        raise ValueError(
            f"No file at {path!r}. Check the path (it is relative to the working "
            "directory) and try again."
        )
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path)
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return pd.read_parquet(path)
    raise ValueError(
        f"Unsupported format for {path!r}. Supported: .csv, .parquet. "
        "Convert the file or point me at a supported one."
    )


@tool
def load_dataset(path: str) -> str:
    """Load a tabular dataset (CSV or Parquet) and return a short text summary.

    Use this first to discover a dataset's shape and column names before writing any
    analysis code, so you never have to guess the schema. Prints a one-line summary and
    returns the same summary as the observation.

    Args:
        path: Filesystem path to the dataset (a ``.csv`` or ``.parquet`` file).
    """
    df = _read_table(path)
    summary = (
        f"Loaded {path}: {df.shape[0]} rows x {df.shape[1]} columns. "
        f"Columns: {list(df.columns)}."
    )
    print(summary)
    return summary


@tool
def profile_dataframe(path: str) -> str:
    """Profile a tabular dataset: schema, dtypes, summary statistics, and missing values.

    Use this to understand a dataset before analyzing it — it reports each column's dtype,
    ``describe()`` statistics for numeric columns, and how many values are missing per
    column. Prints the profile and returns it as the observation.

    Args:
        path: Filesystem path to the dataset (a ``.csv`` or ``.parquet`` file).
    """
    df = _read_table(path)

    dtypes = "\n".join(f"  - {col}: {dtype}" for col, dtype in df.dtypes.items())
    missing = df.isna().sum()
    missing_lines = "\n".join(
        f"  - {col}: {int(count)}" for col, count in missing.items() if count
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
        "directory as a PNG and return the saved file path. Draw your chart first with "
        "matplotlib, then call this — do NOT use plt.show() (it saves nothing). Pass an "
        "optional base filename; otherwise a timestamped name is used."
    )
    inputs = {
        "filename": {
            "type": "string",
            "description": "Optional base filename (without directory). '.png' is added "
                           "if missing. Omit to auto-name with a timestamp.",
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
