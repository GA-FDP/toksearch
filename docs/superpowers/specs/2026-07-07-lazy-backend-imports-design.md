# Lazy-load Ray and Spark backends in TokSearch

**Date:** 2026-07-07
**Status:** Approved design
**Scope:** `toksearch` source repo only (published to `ga-fdp` via a `release-*` tag separately)

## Problem

`import toksearch` costs ~4.3s warm (and far worse cold on the `/fusion` NFS
mount — a first-run `import fdp.cli` alone was measured at 4.8s). Profiling with
`python -X importtime` attributes the cost as:

| Component | Cumulative (warm) | On import path via |
|-----------|-------------------|--------------------|
| `ray` backend | ~2.6s | `record_pipeline.py:65` (eager) |
| `pyspark` / Spark backend | ~0.34s | `record_pipeline.py:67,68` (eager) |
| `xarray` | ~0.6–0.9s | `record_pipeline.py:35`, signal path |
| `MDSplus` | ~0.3s | `signal/mds.py` via public `MdsSignal` |

Within Ray, the cost is dominated by a single pathological transitive
dependency, `rfc3987_syntax.syntax_helpers` (pulled in by
`ray.runtime_env → jsonschema`), which compiles a large grammar at import time
(~1.0–2.2s of CPU, largely cache-independent).

The Ray and Spark backends are imported **eagerly** at module top level, so
their combined ~2.9s is paid on *every* `import toksearch` — even though the
common FDP workflows use `compute_multiprocessing()` or `compute_serial()` and
never touch Ray or Spark.

## Goal

Defer the Ray and Spark backend imports so they are only paid when a user
actually calls `compute_ray()` or `compute_spark()`. Target: `import toksearch`
drops from ~4.3s → ~1.6s warm.

Out of scope (deliberately): deferring `xarray`, `MDSplus`, or `numpy`. These
sit on the core signal/dataset path (`xarray` is used pervasively in
`record_pipeline.py`; `MDSplus` is reached through the public `MdsSignal`
re-export in `toksearch/__init__.py`) and are far riskier to defer for a smaller
marginal win.

## Current state (verified)

- `toksearch/pipeline/record_pipeline.py` eagerly imports all four backends:
  - line 64: `from ..backend.multiprocessing import MultiprocessingRecordSet, MultiprocessingConfig`
  - line 65: `from ..backend.ray import RayRecordSet, RayConfig`
  - line 67: `from ..backend.spark import SparkRecordSet, ToksearchSparkConfig`
  - line 68: `from pyspark.context import SparkContext`
  - line 70: `from ..backend.serial import SerialRecordSet`
- The backend symbols are used only inside the `compute_*` methods and in their
  signatures:
  - `compute_ray(...) -> RayRecordSet` uses `RayConfig`, `RayRecordSet`
  - `compute_spark(sc: Optional[SparkContext] = None, ...) -> SparkRecordSet`
    uses `ToksearchSparkConfig`, `SparkRecordSet`, `SparkContext`
  - `compute_multiprocessing`, `compute_serial` use the cheap backends
- `toksearch/__init__.py` re-exports **none** of the backend RecordSet/Config
  classes (only `Signal`, `ZarrSignal`, `MdsSignal`, `MdsTreePath`,
  `XarrayAligner`, `Pipeline`). Making backends lazy therefore changes no public
  import surface.
- The only other eager `import ray` / `import pyspark` sites are in
  `toksearch/slurm/` (`ray_cluster.py`, `spark_cluster.py`), which are **not** on
  the `import toksearch` path (`__init__.py` imports only `signal` and
  `pipeline`). No change needed there.
- `backend/multiprocessing` and `backend/serial` do not transitively import
  Ray or Spark.

## Design

All changes are contained in `toksearch/pipeline/record_pipeline.py`.

1. **Enable deferred annotations.** Add `from __future__ import annotations`
   (PEP 563) at the top of the file so all annotations become un-evaluated
   strings. This lets `compute_ray()`/`compute_spark()` keep their
   `-> RayRecordSet` and `sc: Optional[SparkContext]` annotations without those
   names being resolvable at function-definition time.

   Safe here: `Pipeline` is a plain class — not a dataclass or pydantic model —
   and nothing calls `typing.get_type_hints()` on it, so no runtime code depends
   on the annotations being live objects.

2. **Remove the eager backend imports.** Delete the three top-level lines:
   - `from ..backend.ray import RayRecordSet, RayConfig`
   - `from ..backend.spark import SparkRecordSet, ToksearchSparkConfig`
   - `from pyspark.context import SparkContext`

3. **Add a `TYPE_CHECKING` block** re-declaring those names so static type
   checkers and IDEs still resolve the annotations:

   ```python
   from typing import TYPE_CHECKING
   if TYPE_CHECKING:
       from ..backend.ray import RayRecordSet, RayConfig
       from ..backend.spark import SparkRecordSet, ToksearchSparkConfig
       from pyspark.context import SparkContext
   ```

4. **Move the imports into the method bodies:**
   - `compute_ray()`: add `from ..backend.ray import RayRecordSet, RayConfig` at
     the top of the method body.
   - `compute_spark()`: add
     `from ..backend.spark import SparkRecordSet, ToksearchSparkConfig` at the
     top of the method body. `SparkContext` is referenced **only** in the
     `sc: Optional[SparkContext] = None` annotation (verified — the default value
     is `None`, not a `SparkContext()` call), so under PEP 563 it needs no
     runtime import inside the method.

5. **Leave `multiprocessing` and `serial` backend imports eager** — they are
   cheap and pull in nothing heavy.

Because Python caches imported modules in `sys.modules`, the first call to
`compute_ray()`/`compute_spark()` pays the import once; subsequent calls are
free. Behavior is otherwise identical.

## Regression guard

Add a pytest that guards the laziness so an eager import cannot silently creep
back in:

- Spawn a **subprocess** with a clean interpreter (`subprocess.run([sys.executable,
  "-c", "import toksearch; import sys; ...])`) — a subprocess is required
  because pytest itself may already have imported Ray/Spark into the parent's
  `sys.modules`.
- Assert that after `import toksearch`, neither `ray` nor `pyspark` is present in
  `sys.modules`.
- Optionally, a second assertion that after constructing a `Pipeline` (without
  calling a compute method) Ray/Spark are still absent.

This test lives alongside the existing backend tests in the toksearch test
suite.

## Verification

- Run the toksearch test suite (`pixi run pytest`) — existing backend tests must
  still pass, confirming `compute_ray`/`compute_spark` work with the deferred
  imports.
- Measure `import toksearch` warm wall time before/after in the toksearch repo's
  own pixi env; expect ~4.3s → ~1.6s.
- Confirm the new regression test passes.

## Non-goals / follow-ups

- Deferring `xarray` and `MDSplus` (separate, more invasive effort).
- The upstream `rfc3987_syntax` slowness (a jsonschema/ray dependency) — sidesteps
  itself once Ray is lazy; not chased directly.
- Cutting the `ga-fdp` release: done separately by tagging `release-*`.
