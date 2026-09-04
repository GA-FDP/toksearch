# Provenance and CMF

A computed result is only reusable if you can say what produced it. TokSearch
derives that description itself — the shot source, the specification of every
signal fetched, the sequence of operations, the compute backend, and the git
commit of the script that ran — and hands it to a provenance backend. You do
not hand-write metadata.

## The interface

`toksearch.provenance` defines the contract and nothing else. It has no
dependency on any metadata service; the CMF implementation lives in the
separate `toksearch_cmf` package.

**`RunContext`** is the derived description of a run. Its parts:

| Component | What it carries |
|---|---|
| `SourceSpec` | Where the shots came from: `kind` (`shotlist`, `sql`, `recordset`), `count`, a hash of the shot list, and for SQL sources the `query` and `params` |
| `OpSpec` | One per pipeline operation, in order: `fetch`, `map`, `keep`, `where`, `align`, `write`, each with its detail |
| `BackendSpec` | Which compute backend ran it, and its configuration |
| `CodeSpec` | The executing script, its `argv`, the git `commit`, and whether the tree was `dirty` |
| `signals` | The `Signal.spec()` of every fetched signal, keyed by record field |

`RunContext.input_identity()` hashes only the source, the signals, and the
device — deliberately excluding operations, backend, and code. Two runs that
read the same data have the same input identity even if they process it
differently.

**`Provenance`** is the ABC a backend implements:

| Hook | When |
|---|---|
| `on_compute_start(ctx)` | Before the backend runs |
| `on_compute_end(ctx, recordset)` | After compute returns |
| `output(*paths, **custom_properties)` | Declare artifacts toksearch did not write itself |
| `metrics(name, values)` | Record a named set of metrics |
| `finalize()` | Flush and close the record |

Hooks are invoked through `safe_call`, which converts any exception into a
`RuntimeWarning` and carries on. Losing a provenance record is bad; losing a
completed multi-hour compute is worse. Set `strict=True` on the backend to make
failures propagate instead — appropriate in CI, not in production runs.

## Recording a run

Pass a backend to any of the four `compute_*` methods. `JsonProvenance` needs
no external services, which makes it the right thing to try first:

```python
import xarray as xr
from toksearch import Pipeline
from toksearch.provenance import JsonProvenance

def peaks(rec):
    rec["peaks"] = xr.Dataset(
        {"double": ("shot", [rec.shot * 2])},
        coords={"shot": ("shot", [rec.shot])},
    )

pipeline = Pipeline([180000, 180001])
pipeline.map(peaks)
pipeline.keep(["peaks"])
pipeline.write("out", field="peaks", fmt="netcdf")

run = JsonProvenance("demo", stage="explore", path="run.json")
results = pipeline.compute_serial(provenance=run)
run.metrics("coverage", {"requested": 2, "returned": len(results)})
run.finalize()
```

That writes `out/180000.nc` and `out/180001.nc`, and this `run.json`:

```json
{
  "context": {
    "backend": {"config": {}, "kind": "SerialRecordSet"},
    "code": {
      "argv": ["demo.py"],
      "commit": "b06208582106642adfc5666eef5c3c43628204d8",
      "dirty": false,
      "repo_root": "/path/to/demo",
      "script": "demo.py"
    },
    "device": null,
    "ops": [
      {"op": "map",
       "detail": {"func": {"module": "__main__", "name": "peaks",
                           "source_sha256": "30b447cd1662a427..."}}},
      {"op": "keep", "detail": {"fields": ["peaks"]}},
      {"op": "write",
       "detail": {"directory": "/path/to/demo/out", "fields": ["peaks"],
                  "fmt": "netcdf", "func": null, "name": null,
                  "on_error": "skip", "path_field": "output_path",
                  "track": "directory"}}
    ],
    "parent_run": null,
    "signals": {},
    "source": {"count": 2, "hash": "8964cb18ba55ec4e...", "kind": "shotlist",
               "params": null, "query": null}
  },
  "input_identity": "e320ff068b07e8d0aa91bdcd7c7b8c2c454cf1dee2879fab1ef8718668e15d36",
  "metrics": {"coverage": {"requested": 2, "returned": 2}},
  "outputs": [{"path": "/path/to/demo/out", "source": "pipeline_write"}],
  "pipeline_name": "demo",
  "run_id": "bd9348f4471b4f2fb14ca64aea5fcdf6",
  "stage": "explore"
}
```

`source_sha256` is the hash of the mapped function's source, so a change to
`peaks` produces a different record. `commit` and `dirty` come from the git
repository containing the script; outside a repository both are `null`.

Add a `fetch` and the `signals` block fills in with that signal's full
specification — class, module, expression, tree name, dims, and every
constructor field — which is what makes the input identity meaningful.

### Chaining runs

`Pipeline.compute` copies the backend's `run_id` onto the returned
`RecordSet`. Building a new pipeline from that recordset —
`Pipeline(previous_results)` — reads it back as `RunContext.parent_run`, so a
multi-stage analysis links itself without any bookkeeping on your part.

## Getting data out

### `Pipeline.write`

Write one file per record, in the worker that produced it. This is the
recommended way to get data out of a pipeline: writing per shot in the workers
is faster than concatenating on the driver, and more honest — concatenation is
a transformation and deserves its own stage rather than hiding inside a writer.

Two forms. **Declarative**, which appends the operation immediately:

```python
pipeline.write("out/peaks", field="ds", fmt="netcdf")
pipeline.write("out/peaks", fields=["ip", "betan"], fmt="netcdf")   # xr.merge
```

**Decorator**, when the file's content needs computing:

```python
@pipeline.write("out/peaks", fmt="netcdf")
def shot_file(rec):
    return rec["ds"]
```

A two-argument function takes `(record, path)`, writes the file itself, and
returns the path it wrote.

| Argument | Meaning |
|---|---|
| `directory` | Output directory; created if absent |
| `field` / `fields` | One record field, or several merged with `xarray.merge` |
| `fmt` | `netcdf`, `parquet`, `npy`, `npz`, `json`; inferred from the object when omitted |
| `name` | `(record) -> str` basename without extension; defaults to the shot number |
| `track` | `directory` (one artifact for the whole directory) or `file` (one per shot) |
| `exist_ok` | Off by default — two runs interleaving into one directory silently corrupt the directory's content hash, and `flock` is not cross-client on BeeGFS, so nothing else prevents it |
| `path_field` | Record field that receives the written path |
| `on_error` | `skip` (default) writes no file for a record that already failed, so the directory — and the provenance hash over it — covers exactly the shots that completed; `write` writes anyway |

Read the results back with `xarray.open_mfdataset('out/peaks/*.nc')` where
`dask` is installed. Without it, open per file:

```python
import glob
import xarray as xr

ds = xr.concat(
    [xr.open_dataset(f) for f in sorted(glob.glob("out/peaks/*.nc"))],
    dim="shot",
)
```

Without `netCDF4`/`h5netcdf`, xarray writes NetCDF3 via scipy, which has no
int64 — integer coordinates read back as int32.

### Driver-side collection

When the per-shot results are small enough to gather, `RecordSet` collapses
them directly:

```python
results = pipeline.compute_multiprocessing(num_workers=16)
df = results.to_dataframe()                       # all scalar fields
df = results.to_dataframe(fields=["ip_max_ma"])   # a subset
results.to_parquet("peaks.parquet")
```

## Recording to CMF

[CMF](https://github.com/HewlettPackard/cmf) (Common Metadata Framework) is a
git- and DVC-backed metadata store. `toksearch_cmf` provides `CmfRun`, a
`Provenance` implementation that records a run there. It owns every `cmflib`
and `dvc` dependency; TokSearch core knows about neither.

`toksearch_cmf` is part of the `fdp-core` metapackage, so an FDP environment
already has it.

```python
from toksearch import MdsSignal, Pipeline
from toksearch_d3d import PtDataSignal
from toksearch_cmf import CmfRun

run = CmfRun("betan-ip-study", stage="assemble", work_dir=".")

pipeline = Pipeline(shots)
pipeline.fetch("ip", PtDataSignal("ip"))
pipeline.fetch(
    "betan",
    MdsSignal(r"\betan", "efit01", location="remote://atlas.gat.com"),
)
pipeline.map(peaks)
pipeline.keep(["peaks"])
pipeline.write("peaks", field="peaks", fmt="netcdf")

results = pipeline.compute_multiprocessing(num_workers=8, provenance=run)

run.metrics("coverage", {"requested": len(shots), "returned": len(results)})
run.finalize()
```

Nothing there hand-writes a `cmflib.log_dataset` call. TokSearch derives the
run description and `CmfRun` records it.

**Prerequisites.** cmflib records the executing script's commit and hands
output paths to DVC, so the script must run from inside a git repository that
has a remote and DVC initialised, with the script itself committed there.
`CmfRun` checks for the repository up front rather than failing after a long
compute.

**Run it with `python -m fdp run`, not `fdp run`.** In any environment
carrying cmflib, graphviz arrives transitively (`cmflib → dvc → pydot →
graphviz`) and installs its own layout engine at `bin/fdp`. `fdp` 0.6.0
declared graphviz as a dependency so the installer's link order gives the FDP
CLI the file back (verified on pixi/rattler and micromamba 2.9.0). That still
leaves two ways to lose the collision: an `fdp` older than 0.6.0, or an
installer whose link order isn't guaranteed the way those two are.
`python -m fdp` sidesteps the question either way:

```bash
python -m fdp run python betan_ip_peaks_cmf.py
```

A complete working example is
[`examples/betan_ip_peaks_cmf.py`](https://github.com/GA-FDP/toksearch_cmf/blob/main/examples/betan_ip_peaks_cmf.py)
in the `toksearch_cmf` repository. The prerequisite passage above is kept
deliberately in sync with the same passage in that repository's README; if you
change one, change the other.

## API reference

::: toksearch.provenance.Provenance
    handler: python
    options:
        show_root_heading: True
        members_order: source

::: toksearch.provenance.RunContext
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.SourceSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.OpSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.BackendSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.CodeSpec
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.JsonProvenance
    handler: python
    options:
        show_root_heading: True

::: toksearch.provenance.safe_call
    handler: python
    options:
        show_root_heading: True

### Pipeline output operations

`Pipeline.write` is rendered in full on the [Pipeline](pipeline.md) reference
page, and `RecordSet.to_dataframe` / `RecordSet.to_parquet` on the
[RecordSet](record_set.md) page. Their arguments are tabulated under
[Getting data out](#getting-data-out) above.
