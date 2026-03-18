---
name: toksearch-mds
description: MdsSignal for fetching MDSplus tree data — constructor args, location formats, TDI expressions, dims, and DIII-D efit01 signals via FDP/Pelican
user-invocable: false
license: Apache-2.0
compatibility: Claude Code
metadata:
  author: GA-FDP
  version: "1.0"
  url: https://ga-fdp.github.io/toksearch/
---

# TokSearch MdsSignal Skill

## Description

`MdsSignal` fetches data from MDSplus trees and integrates with the TokSearch `Pipeline`. It handles both local trees (via environment variable tree paths) and remote MDSplus servers, and is the standard way to access equilibrium, profile, and other tree-based data in FDP workflows.

## When to Use

- Fetching EFIT equilibrium quantities (Ip, q95, profiles, flux surfaces)
- Reading any MDSplus tree node using TDI expressions
- Accessing DIII-D data stored in MDSplus format via FDP/Pelican

## Constructor

```python
from toksearch import MdsSignal

MdsSignal(
    expression,          # TDI expression string (required)
    treename,            # MDSplus tree name (required)
    location=None,       # tree location (see below)
    dims=('times',),     # dimension names for axes
    data_order=None,     # axis order in raw data (if different from dims)
    fetch_units=True,    # whether to fetch units strings
)
```

## Return Value

`signal.fetch(shot)` returns a dict:

```python
{
    'data':  np.ndarray,          # signal values
    'times': np.ndarray,          # time axis in milliseconds (default)
    'units': {'data': str, 'times': str},  # present if fetch_units=True
}
```

For multi-dimensional signals, additional dimension arrays are present:

```python
{
    'data':  np.ndarray,   # shape (n_time, n_rho) for profiles
    'times': np.ndarray,   # shape (n_time,)
    'rho':   np.ndarray,   # shape (n_rho,) — only if dims=('times','rho') set
    'units': {...},
}
```

## TDI Expressions

MDSplus TDI expressions select nodes within a tree. For DIII-D, backslash-prefixed names are node references:

```python
MdsSignal(r'\ipmhd',  'efit01')   # plasma current
MdsSignal(r'\kappa',  'efit01')   # elongation
MdsSignal(r'\betap',  'efit01')   # poloidal beta
MdsSignal(r'\q95',    'efit01')   # q at 95% flux surface
MdsSignal(r'\betat',  'efit01')   # toroidal beta
MdsSignal(r'\wp',     'efit01')   # stored energy
MdsSignal(r'\ne',     'efit01')   # line-averaged electron density
MdsSignal(r'\t_e',    'efit01')   # electron temperature
MdsSignal(r'\psirz',  'efit01')   # poloidal flux (2D: time × (R,Z) grid)
```

TDI expressions can also be arbitrary TDI function calls:

```python
MdsSignal(r'abs(\ipmhd)', 'efit01')
```

## Location Formats

The `location` argument controls where MDSplus looks for tree files.

### `None` — use environment variables (recommended for FDP)

```python
sig = MdsSignal(r'\ipmhd', 'efit01')  # location defaults to None
```

When `location=None`, MdsSignal reads `${treename}_path` or `default_tree_path` from the environment. The `fdp run` command sets these automatically via the FDP/Pelican configuration. **This is the correct mode for FDP workflows.**

### String path — local tree files

```python
sig = MdsSignal(r'\ipmhd', 'efit01', location='/path/to/mdsplus/trees')
```

### Remote server

```python
sig = MdsSignal(r'\ipmhd', 'efit01', location='remote://atlas.gat.com')
```

### `MdsTreePath` — explicit multi-tree mapping

```python
from toksearch.signal.mds import MdsTreePath

tree_path = MdsTreePath(efit01='/data/efit01', magnetics='/data/magnetics')
sig = MdsSignal(r'\ipmhd', 'efit01', location=tree_path)
```

## FDP/Pelican Note

When running via `fdp run`, the environment variable `default_tree_path` is set to a semicolon-delimited list of Pelican URLs covering all DIII-D shot archives. **Only the `efit01` tree is confirmed available via FDP Pelican.** Do not assume other trees (magnetics, pcs, ece, etc.) are accessible.

```bash
# Correct usage with FDP
fdp run python my_analysis.py
```

```python
# In my_analysis.py — location=None works because fdp run set the env vars
from toksearch import Pipeline, MdsSignal

pipeline = Pipeline([202159, 202160, 202161])
pipeline.fetch('ip', MdsSignal(r'\ipmhd', 'efit01'))
records = pipeline.compute_serial()
```

## Multi-Dimensional Signals

For 2-D or higher-dimensional data, specify `dims` to label each axis:

```python
# psirz is a 2-D array: shape (n_time, n_r*n_z) or similar
sig = MdsSignal(r'\psirz', 'efit01', dims=('times',))
result = sig.fetch(202161)
print(result['data'].shape)   # (n_time, n_r, n_z) or similar
```

If the raw data axis order differs from the desired dim order, use `data_order`:

```python
# data is stored (n_rho, n_time) but you want dims labeled (times, rho)
sig = MdsSignal(r'\someprofile', 'efit01',
                dims=('times', 'rho'),
                data_order=('rho', 'times'))
```

## Disabling Units

```python
sig = MdsSignal(r'\ipmhd', 'efit01', fetch_units=False)
result = sig.fetch(202161)
# result has no 'units' key
```

## In a Pipeline

```python
from toksearch import Pipeline, MdsSignal
import numpy as np

shots = [202159, 202160, 202161]
pipeline = Pipeline(shots)

pipeline.fetch('ip',  MdsSignal(r'\ipmhd', 'efit01'))
pipeline.fetch('q95', MdsSignal(r'\q95',   'efit01'))

@pipeline.map
def flatten_profile(rec):
    # rec['ip']['data'] is the numpy array
    rec['max_ip'] = float(np.max(np.abs(rec['ip']['data'])))

records = pipeline.compute_serial()
```

## Best Practices

- Leave `location=None` in FDP workflows — `fdp run` configures the tree path
- Always use raw strings (`r'\signal'`) for TDI expressions to avoid backslash issues
- Use `fetch_units=False` if you do not need units, to reduce data transfer
- Check `rec.errors` if a fetch may fail (tree not open, node not found, etc.)
