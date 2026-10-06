# Creating pipelines from SQL

Under construction

## Snapshots

A device can publish immutable snapshots of its shot database — one
directory per snapshot, `manifest.json` plus Parquet per table — and
declare where they live with a `sql_snapshot` locator. `toksearch.sql.snapshot`
reads one remotely, by HTTP range request, through DuckDB, and rewrites
your T-SQL so it runs there unchanged:

```python
from toksearch.sql import snapshot

with snapshot.connect_tokamak("d3d", "d3drdb") as conn:
    df = pd.read_sql("SELECT TOP 50 shot, entered FROM shots ORDER BY shot DESC", conn)
print(conn.snapshot)          # e.g. d3drdb_20261005T120000Z
```

Which snapshot a process reads is settled once and exported as
`FDP_SQL_SNAPSHOT_<NAME>` (here `FDP_SQL_SNAPSHOT_D3DRDB`), so every
worker a pipeline starts reads the same one. Precedence: `snapshot=` in
code, then that variable, then the pairing the process's catalog
records, then the newest published. Code and environment naming
different snapshots is an error, not a choice.

Result column names follow the query's spelling, as SQL Server's do:
`SELECT shot FROM shots` returns a column `shot`, though the table
stores it as `SHOT`, so `df["shot"]` works on both. `SELECT *` returns
the stored names.

Views are created on first use, so connecting is cheap and the first
query on each table pays its Parquet footer read.

What is **not** done: bytes are not checked against the manifest's
hashes (a range read cannot hash a file; `fdp snapshot verify` does), a
missing snapshot is never replaced by another, and nothing falls back
to the live database. The first connection in a process issues a
`SnapshotNotice` naming the snapshot; silence it with
`warnings.filterwarnings("ignore", category=snapshot.SnapshotNotice)`.

Requires `python-duckdb`, `duckdb-extension-httpfs` and `sqlglot`
(conda-forge). `toksearch` does not declare them; the device package
that ships the locator does.
