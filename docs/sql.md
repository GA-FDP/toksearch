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
code, then that variable, then the newest published. Code and
environment naming different snapshots is an error, not a choice.

Result column names follow the query's spelling, as SQL Server's do:
`SELECT shot FROM shots` returns a column `shot`, though the table
stores it as `SHOT`, so `df["shot"]` works on both. `SELECT *` returns
the stored names.

Views are created on first use, so connecting is cheap and the first
query on each table pays its Parquet footer read.

What is **not** done: bytes are not checked against the manifest's
hashes on read (a range read cannot hash a file; `verify_files` below
does), a missing snapshot is never replaced by another, and nothing
falls back to the live database. The first connection in a process
issues a `SnapshotNotice` naming the snapshot; silence it with
`warnings.filterwarnings("ignore", category=snapshot.SnapshotNotice)`.

### Replay and verification

A pipeline's `compute()` settles the snapshot of every registered
`sql_snapshot` locator before any worker starts — the one already in
`FDP_SQL_SNAPSHOT_<NAME>`, else the newest — exports it, and records it in
the run's provenance as `store["sql_snapshots"]`, so workers that connect
agree with each other and with the record even if the driver never
connected. When that cannot be done (no DuckDB, no token, origin
unreachable within 10 s) nothing is settled, the process stops trying, and
the run proceeds; the first connection raises with the real error. A saved snapshot (`fdp-snapshot/2`) carries
the ids as `sql_snapshots`, and `Pipeline.from_snapshot` pins exactly
those for the replay; an environment naming a different id is a
`SnapshotConflict`, not a choice. `snapshot.verify_files(locator, id,
sample=None)` streams each Parquet file of a published snapshot through
SHA-256 over the same path the client reads, compares it with the
manifest, and returns `(checked, total, failures)`; `fdp snapshot verify`
calls it for each id a saved snapshot names.

Requires `python-duckdb`, `duckdb-extension-httpfs` and `sqlglot`
(conda-forge). `toksearch` does not declare them; the device package
that ships the locator does.
