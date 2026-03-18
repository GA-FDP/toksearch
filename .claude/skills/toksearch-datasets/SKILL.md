---
name: toksearch-datasets
description: TokSearch xarray dataset integration — fetch_dataset and align for multi-signal time-aligned analysis
user-invocable: false
license: Apache-2.0
compatibility: Claude Code
metadata:
  author: GA-FDP
  version: "1.0"
  url: https://ga-fdp.github.io/toksearch/
---

# TokSearch Datasets Skill

## Description

`Pipeline.fetch_dataset` and `Pipeline.align` allow multiple signals to be combined into an `xr.Dataset` per record and aligned to a common time base. This is the recommended pattern when working with several signals that need to be compared or processed together on a shared time grid.

## When to Use

- Combining multiple signals into a single xarray Dataset per shot
- Aligning signals with different native time bases to a common grid
- Passing multi-signal data downstream to xarray-aware analysis code

## `fetch_dataset`

Creates an `xr.Dataset` field in each record from a dict of signal objects. Each key becomes a data variable in the dataset, with its native time coordinate.

```python
from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal

shots = [202159, 202160, 202161]
pipeline = Pipeline(shots)

pipeline.fetch_dataset('eq_data', {
    'ip':   MdsSignal(r'\ipmhd', 'efit01'),
    'q95':  MdsSignal(r'\q95',   'efit01'),
    'betap': MdsSignal(r'\betap', 'efit01'),
})
```

After this step, `rec['eq_data']` is an `xr.Dataset` with data variables `'ip'`, `'q95'`, and `'betap'`, each with their own (possibly different) `times` coordinate.

### Appending to an existing dataset

By default (`append=True`), subsequent `fetch_dataset` calls with the same name add variables to the existing dataset:

```python
pipeline.fetch_dataset('signals', {'ip':  MdsSignal(r'\ipmhd', 'efit01')})
pipeline.fetch_dataset('signals', {'dens': PtDataSignal('dssdenest')})
# rec['signals'] now has both 'ip' and 'dens'
```

Pass `append=False` to replace any existing field with that name.

## `align`

Interpolates all variables in an `xr.Dataset` to a common coordinate grid (typically `'times'`).

```python
pipeline.align(
    ds_name='eq_data',     # name of the dataset field in the record
    align_with='ip',       # align to the time base of the 'ip' variable
    dim='times',           # which dimension to align along (default 'times')
    method='pad',          # interpolation method (default: zero-order hold)
    extrapolate=True,      # whether to extrapolate beyond data bounds
)
```

### `align_with` options

| Value | Meaning |
|-------|---------|
| `'variable_name'` | Use the coordinate of that variable in the dataset |
| `[t1, t2, t3, ...]` | Explicit list of target times |
| `np.ndarray` | Array of target times |
| `callable(ds, dim)` | Function returning target times |
| `float` (e.g. `1.0`) | Sample period in ms — creates uniform grid |

### `method` options

| Method | Description |
|--------|-------------|
| `'pad'` | Zero-order hold (step function); default |
| `'linear'` | Linear interpolation |
| `'cubic'` | Cubic spline interpolation |

## Complete Example

Fetch three signals, align to a uniform 1 ms grid, then compute correlations:

```python
import numpy as np
from toksearch import Pipeline, MdsSignal
from toksearch_d3d import PtDataSignal

shots = [202159, 202160, 202161]
pipeline = Pipeline(shots)

pipeline.fetch_dataset('data', {
    'ip':   MdsSignal(r'\ipmhd', 'efit01'),
    'q95':  MdsSignal(r'\q95',   'efit01'),
    'dens': PtDataSignal('dssdenest'),
})

# Align all variables to a uniform 1 ms time grid
pipeline.align('data', align_with=1.0, method='linear')

@pipeline.map
def select_flattop(rec):
    ds = rec['data']
    # Select times between 1000 and 4000 ms
    mask = (ds['times'] >= 1000) & (ds['times'] <= 4000)
    rec['flattop'] = ds.sel(times=ds['times'][mask])

pipeline.keep(['shot', 'flattop'])
records = pipeline.compute_serial()
```

## Accessing Dataset Fields in `map`

After `fetch_dataset`, the dataset lives at `rec['dataset_name']`:

```python
@pipeline.map
def analyze(rec):
    ds = rec['data']                        # xr.Dataset
    ip_arr  = ds['ip'].values               # numpy array
    times   = ds['times'].values            # time coordinate
    q95_mean = float(ds['q95'].mean())      # xarray operation
```

## `fetch_as_xarray` on Individual Signals

Signals can also be fetched directly as xarray objects (without using `fetch_dataset`):

```python
from toksearch import MdsSignal

sig = MdsSignal(r'\ipmhd', 'efit01')
da = sig.fetch_as_xarray(202161)
# <xarray.DataArray (times: N)>
# Coordinates:
#   * times  (times) float64 ...
# Attributes:
#     units: A
```

## Best Practices

- Use `fetch_dataset` + `align` when multiple signals need to be compared at the same time points
- `align_with=1.0` (uniform 1 ms grid) is a safe default for DIII-D signals that typically have ≤1 ms resolution
- Call `align` after all `fetch_dataset` steps, before any `map` that uses the aligned data
- After alignment, `rec['ds']['signal'].values` gives a regular numpy array with no NaN gaps
