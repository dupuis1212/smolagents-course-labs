"""Build ``data/sales.db`` (SQLite) from ``data/sales.csv`` for the MCP SQLite server (M9).

The stdio MCP data server (``uvx mcp-server-sqlite --db-path data/sales.db``) serves a SQLite
file; this script materializes that file from Quill's canonical CSV. The frozen dataset
``data/sales.csv`` (06 §2) is the source of truth — the DB is a derived artifact (one ``sales``
table with the same columns), so re-running this is idempotent.

Run it once after setup (or whenever ``sales.csv`` changes):

    uv run python -m quill.scripts.build_sales_db          # writes data/sales.db
    uv run python -m quill.scripts.build_sales_db --csv data/sales.csv --db data/sales.db

No network, no token — pure local pandas + sqlite3 (both stdlib/already-installed). Imports live
inside the function (the "pushable" habit) so importing this module is side-effect-free.
"""
from __future__ import annotations

DEFAULT_CSV = "data/sales.csv"
DEFAULT_DB = "data/sales.db"
TABLE_NAME = "sales"


def build_sales_db(csv_path: str = DEFAULT_CSV, db_path: str = DEFAULT_DB,
                   table: str = TABLE_NAME) -> str:
    """Load ``csv_path`` and write it as a single ``table`` in the SQLite file ``db_path``.

    Args:
        csv_path: source CSV (defaults to Quill's frozen ``data/sales.csv``).
        db_path: destination SQLite file (defaults to ``data/sales.db``).
        table: the table name the MCP server queries (defaults to ``sales``).

    Returns:
        The ``db_path`` written.

    Raises:
        FileNotFoundError: if ``csv_path`` does not exist (build it / fix the path first).
    """
    import os
    import sqlite3

    import pandas as pd

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No CSV at {csv_path!r}. Run from the module-09/ directory so data/sales.csv resolves."
        )
    df = pd.read_csv(csv_path)
    con = sqlite3.connect(db_path)
    try:
        df.to_sql(table, con, index=False, if_exists="replace")
        con.commit()
    finally:
        con.close()
    print(f"Built {db_path}: table {table!r} with {len(df)} rows, columns {list(df.columns)}.")
    return db_path


def main() -> int:
    """CLI: ``python -m quill.scripts.build_sales_db [--csv PATH] [--db PATH]``."""
    import argparse

    parser = argparse.ArgumentParser(description="Build data/sales.db from data/sales.csv.")
    parser.add_argument("--csv", default=DEFAULT_CSV, help="source CSV path")
    parser.add_argument("--db", default=DEFAULT_DB, help="destination SQLite path")
    parser.add_argument("--table", default=TABLE_NAME, help="table name to write")
    args = parser.parse_args()
    build_sales_db(args.csv, args.db, args.table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
