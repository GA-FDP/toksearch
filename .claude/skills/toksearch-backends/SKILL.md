---
name: toksearch-backends
description: TokSearch compute backends — serial, multiprocessing, Ray, and Spark; when to use each and configuration options
user-invocable: false
license: Apache-2.0
compatibility: Claude Code
metadata:
  author: GA-FDP
  version: "1.0"
  url: https://ga-fdp.github.io/toksearch/
---

# TokSearch Backends Skill

## Description

TokSearch pipelines can execute across four backends. Each backend processes shots independently (pipelines are stateless across shots), so the choice of backend is primarily a question of scale and available infrastructure.

## Backend Summary

| Backend | Method | Best for |
|---------|--------|----------|
| Serial | `compute_serial()` | Development, debugging, <~10 shots |
| Multiprocessing | `compute_multiprocessing()` | Local parallel, 10–1000 shots |
| Ray | `compute_ray()` | Distributed cluster or large local runs |
| Spark | `compute_spark()` | Existing Spark infrastructure |

## Serial

Single-process execution. Simple, no setup required, full Python tracebacks on errors.

```python
records = pipeline.compute_serial()
```

Use serial when:
- Writing and testing new pipelines
- Debugging — error tracebacks are clearest in serial mode
- Fetching a small number of shots

## Multiprocessing

Uses `joblib.Parallel` across local CPU cores. No distributed infrastructure needed.

```python
# Use half the available CPUs (default)
records = pipeline.compute_multiprocessing()

# Specify worker count explicitly
records = pipeline.compute_multiprocessing(num_workers=8)

# Control job batching (passed to joblib.Parallel)
records = pipeline.compute_multiprocessing(num_workers=4, batch_size=10)
```

Arguments:
- `num_workers`: number of parallel workers; defaults to `cpu_count() // 2`
- `batch_size`: joblib batch size; defaults to `"auto"`

Use multiprocessing when:
- Running on a single machine with multiple cores
- Shot list is moderate (tens to hundreds)
- No Ray or Spark cluster available

**Note**: Signal objects must be picklable for multiprocessing. Most TokSearch signals are picklable. If you encounter pickling errors, use serial mode.

## Ray

Distributed execution via [Ray](https://ray.io/). Can run on a single machine (Ray manages workers) or on a Ray cluster.

```python
# Let Ray auto-configure (uses all available CPUs)
records = pipeline.compute_ray()

# Set number of parallel partitions
records = pipeline.compute_ray(numparts=16)

# Limit memory per shot (bytes) to avoid OOM
records = pipeline.compute_ray(memory_per_shot=500_000_000)  # 500 MB

# Pass kwargs to ray.init
records = pipeline.compute_ray(address='ray://cluster-head:10001')
```

Arguments:
- `numparts`: number of partitions (defaults to number of records)
- `batch_size`: records per batch
- `verbose`: print progress (default `True`)
- `memory_per_shot`: memory limit per task in bytes
- `**ray_init_kwargs`: passed directly to `ray.init`

Use Ray when:
- Shot list is large (hundreds to thousands)
- Ray is available in the environment
- Cluster resources are needed

## Spark

Runs via Apache Spark. Requires an existing Spark installation or cluster.

```python
from pyspark.context import SparkContext

# Create or reuse a SparkContext
sc = SparkContext.getOrCreate()

records = pipeline.compute_spark(sc=sc)

# Control number of partitions
records = pipeline.compute_spark(sc=sc, numparts=50)

# Cache the RDD for repeated operations
records = pipeline.compute_spark(sc=sc, cache=True)
```

Arguments:
- `sc`: `SparkContext` to use; a default is created if not provided
- `numparts`: number of RDD partitions (defaults to number of records)
- `cache`: whether to cache the RDD (default `False`)

Use Spark when:
- Working within an existing Spark/HPC environment
- Very large shot lists (thousands+)

## Decision Guide

```
Small shot list or debugging?
  └─ compute_serial()

Single machine, moderate shots (10–1000)?
  └─ compute_multiprocessing()

Large shot list or need distributed execution?
  ├─ Ray available? → compute_ray()
  └─ Spark cluster? → compute_spark()
```

## Working with Results

All backends return an iterable `RecordSet`. Iterate directly or convert to a list:

```python
records = pipeline.compute_multiprocessing()

# Iterate
for rec in records:
    print(rec['shot'], rec.get('max_ip', None))

# Convert to list (loads all into memory)
result_list = list(records)
```

## Performance Tips

- Call `pipeline.keep([...])` before compute to drop large arrays and reduce serialization overhead in parallel backends
- For `compute_ray`, `memory_per_shot` prevents worker OOM when signals are large (e.g. 2-D psirz arrays)
- `compute_multiprocessing` is usually the easiest parallel option on a single workstation
