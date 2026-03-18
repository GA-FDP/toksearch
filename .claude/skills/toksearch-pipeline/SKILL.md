---
name: toksearch-pipeline
description: Core TokSearch Pipeline workflow — creating pipelines, fetch/map/where/keep, record data access, and iterating results
user-invocable: false
license: Apache-2.0
compatibility: Claude Code
metadata:
  author: GA-FDP
  version: "1.0"
  url: https://ga-fdp.github.io/toksearch/
---

# TokSearch Pipeline Skill

## Description

`Pipeline` is the central abstraction in TokSearch for processing fusion experimental data across multiple shots. It applies a sequence of operations (fetch, map, where, keep) to `Record` objects — each `Record` holds all data for one shot. Pipelines can run in serial or parallel across multiple backends.

## When to Use

- Fetching one or more signals (MDSplus, PTData, IMAS) across a list of shots
- Filtering shots based on computed quantities
- Building derived quantities from fetched data
- Running the same analysis workflow over many shots

## Core Concepts

### Shot numbers

DIII-D shot numbers are **6-digit integers** (e.g. `202159`, `202160`, `202161`).

### Creating a Pipeline

```python
from toksearch import Pipeline

# From a list of shots
shots = [202159, 202160, 202161]
pipeline = Pipeline(shots)
```

```python
# From an SQL query (requires a DB-API compatible connection)
from toksearch import Pipeline
from toksearch.sql.mssql import connect_d3drdb

conn = connect_d3drdb()
query = "select shot from shots_type where shot_type = 'plasma' and shot > %d"
pipeline = Pipeline.from_sql(conn, query, 200000)
```

The `from_sql` query must produce a `shot` column. Extra columns are added to each record as initial fields.

### Records

Each shot is represented as a `Record` — a dict-like object:

```python
# Within map/where functions:
rec['field_name']            # access a field
rec.shot                     # shot number (integer)
rec.errors                   # dict of {field_name: exception} for failed fetches
rec.get('field_name', None)  # safe access with default (BOTH args required)
```

## Pipeline Operations

### `fetch` — retrieve signal data

```python
from toksearch import MdsSignal

# Direct call style (preferred)
pipeline.fetch('ip', MdsSignal(r'\ipmhd', 'efit01'))

# Decorator style (no function body needed)
@pipeline.fetch('ip', MdsSignal(r'\ipmhd', 'efit01'))
def _(): pass
```

After a fetch, `rec['ip']` is a **dict** with keys:
- `'data'`: numpy array of signal values
- `'times'`: numpy array of time points (milliseconds)
- `'units'`: dict with units strings (if `fetch_units=True`)

**Always access signal data via the dict keys:**

```python
@pipeline.map
def process(rec):
    data = rec['ip']['data']    # correct
    times = rec['ip']['times']  # correct
    # NOT: rec['ip']  ← this is the dict, not the array
```

### `map` — transform records in-place

```python
import numpy as np

@pipeline.map
def compute_max_ip(rec):
    rec['max_ip'] = float(np.max(np.abs(rec['ip']['data'])))
```

`map` functions must not return a value — they modify `rec` in place.

### `where` — filter shots

```python
@pipeline.where
def high_current(rec):
    return rec.get('max_ip', 0) > 1.0e6
```

Return `True` to keep the shot, `False` to drop it. Dropped shots do not appear in results.

### `keep` — drop unused fields

```python
pipeline.keep(['shot', 'max_ip', 'q95'])
```

Takes a **list**. Only the listed fields remain in each record. Reduces memory use before compute.

### `discard` — remove specific fields

```python
pipeline.discard(['raw_data', 'intermediate'])
```

## Running the Pipeline

```python
# Serial execution (default — good for development)
records = pipeline.compute_serial()

# Iterate results
for rec in records:
    print(rec['shot'], rec.get('max_ip', None))
```

Other backends: see `toksearch-backends` skill.

## Complete Example

```python
import numpy as np
from toksearch import Pipeline, MdsSignal

shots = [202159, 202160, 202161]
pipeline = Pipeline(shots)

pipeline.fetch('ip',  MdsSignal(r'\ipmhd', 'efit01'))
pipeline.fetch('q95', MdsSignal(r'\q95',   'efit01'))

@pipeline.map
def compute_stats(rec):
    ip_data = rec['ip']['data']
    rec['max_ip']  = float(np.max(np.abs(ip_data)))
    rec['mean_q95'] = float(np.mean(rec['q95']['data']))

@pipeline.where
def plasma_shots(rec):
    return rec.get('max_ip', 0) > 5.0e5

pipeline.keep(['shot', 'max_ip', 'mean_q95'])

records = pipeline.compute_serial()
for rec in records:
    print(f"Shot {rec['shot']}: Ip_max={rec['max_ip']:.2e}  q95={rec['mean_q95']:.2f}")
```

## Error Handling

Failed fetches are recorded in `rec.errors` and the field is absent from the record:

```python
@pipeline.map
def safe_process(rec):
    if 'ip' in rec.errors:
        # signal fetch failed — rec['ip'] does not exist
        rec['max_ip'] = None
        return
    rec['max_ip'] = float(np.max(np.abs(rec['ip']['data'])))
```

## Single-Shot Debugging

```python
# Run pipeline for one shot without building the full list
record = pipeline.compute_shot(202161)
```

## Best Practices

- Use `compute_serial()` during development; switch to parallel backends for production
- Call `keep()` before compute to avoid carrying large arrays through parallel serialization
- Use `rec.get('field', default)` in `where` filters — the field may be absent if fetch failed
- DIII-D shot numbers are 6 digits; avoid hard-coding short shot numbers in examples
