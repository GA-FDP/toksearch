# Lazy-load Ray and Spark Backends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `import toksearch` ~2.9s faster by deferring the Ray and Spark backend imports until `compute_ray()`/`compute_spark()` are actually called.

**Architecture:** The Ray (~2.6s) and Spark/pyspark (~0.34s) backends are imported eagerly at the top of `toksearch/pipeline/record_pipeline.py`, so their cost is paid on every `import toksearch` even though most workflows use `compute_multiprocessing()`/`compute_serial()`. Enable PEP 563 deferred annotations (`from __future__ import annotations`) so the `-> RayRecordSet` / `sc: Optional[SparkContext]` signatures don't force those names to resolve at definition time, move the heavy imports into the two `compute_*` method bodies, and keep a `TYPE_CHECKING` block for static tooling. A subprocess-based regression test locks in the win.

**Tech Stack:** Python 3.12, `unittest` (test suite discovered by `tests/testit.py`), `pixi` for the dev environment.

**Spec:** `docs/superpowers/specs/2026-07-07-lazy-backend-imports-design.md`

---

## File Structure

- **Modify:** `toksearch/pipeline/record_pipeline.py` — the only file with eager Ray/Spark imports on the `import toksearch` path. Add `from __future__ import annotations`, remove the three eager backend imports, add a `TYPE_CHECKING` block, and move the imports into `compute_ray()`/`compute_spark()`.
- **Create:** `tests/test_lazy_backend_imports.py` — a `unittest` regression test that runs `import toksearch` in a clean subprocess and asserts `ray`/`pyspark` are absent from `sys.modules`.

No public API changes: `toksearch/__init__.py` re-exports none of the backend classes.

---

## Task 1: Add the failing regression test

**Files:**
- Create: `tests/test_lazy_backend_imports.py`

- [ ] **Step 1: Write the regression test**

Create `tests/test_lazy_backend_imports.py` with exactly this content. The test spawns a **fresh** subprocess (required — the pytest/unittest parent process may already have Ray/Spark in `sys.modules`, e.g. `tests/test_pipeline.py` imports `ray` at module top level).

```python
# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression guard: importing toksearch must not eagerly import the heavy
Ray and Spark backends. See
docs/superpowers/specs/2026-07-07-lazy-backend-imports-design.md.
"""

import json
import subprocess
import sys
import unittest


HEAVY_PREFIXES = ("ray", "pyspark")


def _heavy_modules_after(snippet):
    """Run `snippet` in a clean interpreter, return the sorted list of loaded
    ray/pyspark modules."""
    code = (
        f"{snippet}\n"
        "import sys, json\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'ray' or m == 'pyspark'\n"
        "    or m.startswith('ray.') or m.startswith('pyspark.')\n"
        ")\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestLazyBackendImports(unittest.TestCase):
    def test_import_toksearch_does_not_import_ray_or_spark(self):
        heavy = _heavy_modules_after("import toksearch")
        self.assertEqual(
            heavy, [], f"`import toksearch` eagerly loaded: {heavy}"
        )

    def test_building_pipeline_does_not_import_ray_or_spark(self):
        heavy = _heavy_modules_after(
            "from toksearch import Pipeline\np = Pipeline([1, 2, 3])"
        )
        self.assertEqual(
            heavy, [], f"Building a Pipeline eagerly loaded: {heavy}"
        )
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run:
```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
pixi run python -m pytest tests/test_lazy_backend_imports.py -v
```
Expected: **FAIL**. Both tests fail because `import toksearch` currently pulls in Ray (and pyspark) via `record_pipeline.py`. The assertion message will list `ray`, `ray.*`, `pyspark`, `pyspark.*` modules.

- [ ] **Step 3: Commit the failing test**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
git add tests/test_lazy_backend_imports.py
git commit -m "test: guard against eager Ray/Spark import in toksearch"
```

---

## Task 2: Make the Ray and Spark imports lazy

**Files:**
- Modify: `toksearch/pipeline/record_pipeline.py`

Current relevant state (line numbers approximate):
- module docstring ends at the `"""` on line ~21, then `import os` on line ~23
- `from typing import List, Optional, Callable, Union, Type, Iterable` on line ~37
- eager backend imports on lines ~64–70
- `compute_ray()` body builds `RayConfig(...)` and calls `self.compute(RayRecordSet, ...)`
- `compute_spark()` body builds `ToksearchSparkConfig(...)` and calls `self.compute(SparkRecordSet, ...)`

- [ ] **Step 1: Add `from __future__ import annotations` as the first statement after the module docstring**

It MUST be the first statement after the docstring (PEP 563 requirement). Insert it immediately before `import os`:

```python
"""
... existing module docstring ...
"""

from __future__ import annotations

import os
import copy
import importlib
```

- [ ] **Step 2: Add `TYPE_CHECKING` to the typing import**

Change:
```python
from typing import List, Optional, Callable, Union, Type, Iterable
```
to:
```python
from typing import List, Optional, Callable, Union, Type, Iterable, TYPE_CHECKING
```

- [ ] **Step 3: Replace the eager backend import block**

Replace this block (lines ~64–70):
```python
from ..backend.multiprocessing import MultiprocessingRecordSet, MultiprocessingConfig
from ..backend.ray import RayRecordSet, RayConfig

from ..backend.spark import SparkRecordSet, ToksearchSparkConfig
from pyspark.context import SparkContext

from ..backend.serial import SerialRecordSet
```
with:
```python
from ..backend.multiprocessing import MultiprocessingRecordSet, MultiprocessingConfig
from ..backend.serial import SerialRecordSet

# Ray and Spark are heavy optional backends: Ray alone adds ~2.6s to
# `import toksearch` (via its jsonschema -> rfc3987 dependency chain), and
# pyspark ~0.3s. They are imported lazily inside compute_ray()/compute_spark()
# so the common serial/multiprocessing workflows don't pay that cost. These
# TYPE_CHECKING-only imports keep the method annotations resolvable for type
# checkers and IDEs without importing anything at runtime.
if TYPE_CHECKING:
    from ..backend.ray import RayRecordSet, RayConfig
    from ..backend.spark import SparkRecordSet, ToksearchSparkConfig
    from pyspark.context import SparkContext
```

- [ ] **Step 4: Add the lazy import inside `compute_ray()`**

In the body of `compute_ray()`, add the import as the first statement, immediately before `config = RayConfig(`:
```python
        from ..backend.ray import RayRecordSet, RayConfig

        config = RayConfig(
            numparts=numparts,
            placement_group_func=placement_group_func,
            memory_per_task=memory_per_shot,
            **ray_init_kwargs,
        )

        return self.compute(RayRecordSet, config=config)
```

- [ ] **Step 5: Add the lazy import inside `compute_spark()`**

In the body of `compute_spark()`, add the import as the first statement, immediately before `config = ToksearchSparkConfig(`:
```python
        from ..backend.spark import SparkRecordSet, ToksearchSparkConfig

        config = ToksearchSparkConfig(sc=sc, numparts=numparts, cache=cache)
        return self.compute(SparkRecordSet, config=config)
```

Note: `SparkContext` is referenced only in the `sc: Optional[SparkContext] = None` annotation (the default value is `None`), so under PEP 563 it needs no runtime import — the `TYPE_CHECKING` block covers it.

- [ ] **Step 6: Run the regression test to verify it PASSES**

Run:
```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
pixi run python -m pytest tests/test_lazy_backend_imports.py -v
```
Expected: **PASS** (both tests). `import toksearch` and building a `Pipeline` no longer load `ray`/`pyspark`.

- [ ] **Step 7: Commit**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
git add toksearch/pipeline/record_pipeline.py
git commit -m "perf: lazy-load Ray and Spark backends to speed up import

Defers the eager Ray (~2.6s) and Spark (~0.3s) backend imports in
record_pipeline.py into compute_ray()/compute_spark(). Uses PEP 563
deferred annotations so the method signatures still reference the backend
types. No public API change."
```

---

## Task 3: Verify existing backend behavior and measure the win

**Files:** none (verification only)

- [ ] **Step 1: Confirm `compute_ray` / `compute_spark` still work**

The existing `tests/test_pipeline.py` exercises the compute backends (it imports `ray` and uses `RayDD`). Run it to confirm the lazy imports didn't break the compute paths. This spins up a local Ray instance and may take a minute.

Run:
```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
pixi run python -m pytest tests/test_pipeline.py -v
```
Expected: **PASS** (same result as before the change — no new failures).

- [ ] **Step 2: Measure `import toksearch` before/after**

Warm the caches first, then time it:
```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
pixi run python -c "import toksearch"  # warm
for i in 1 2 3; do \
  /usr/bin/time -v pixi run python -c "import toksearch" 2>&1 | grep -i "wall clock"; \
done
```
Expected: warm wall-clock drops from ~4.3s (pre-change baseline) to ~1.6s. (Absolute numbers vary with NFS cache state and the constant `pixi run` overhead; the ~2.9s reduction is the signal.)

- [ ] **Step 3: Confirm no eager ray/spark imports remain on the import path**

Run:
```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch
pixi run python -c "import toksearch, sys; print(sorted(m for m in sys.modules if m.startswith(('ray','pyspark'))))"
```
Expected: `[]`

- [ ] **Step 4: Run the full mock test suite as a final sanity check**

Run:
```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/tests
pixi run --manifest-path ../pixi.toml python -B testit.py --mock
```
Expected: all tests pass (mock mode avoids the ptdata/d3drdb integration requirements). If unrelated pre-existing failures appear, confirm they also fail on the pre-change commit before attributing them to this work.

---

## Self-Review Notes

- **Spec coverage:** All spec design points are covered — `from __future__ import annotations` (Task 2 Step 1), remove eager imports + `TYPE_CHECKING` block (Task 2 Step 3), in-body imports (Steps 4–5), multiprocessing/serial left eager (Step 3), subprocess regression test (Task 1), verification + timing (Task 3).
- **Type consistency:** Symbol names used consistently throughout — `RayRecordSet`, `RayConfig`, `SparkRecordSet`, `ToksearchSparkConfig`, `SparkContext` match the current `record_pipeline.py` definitions and their `backend/ray/__init__.py` / `backend/spark/__init__.py` sources.
- **No placeholders:** every code and command step is concrete.
- **Out of scope (per spec):** deferring `xarray`/`MDSplus`; cutting the `ga-fdp` release (done separately via a `release-*` tag).
