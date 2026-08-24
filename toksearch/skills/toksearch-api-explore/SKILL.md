---
name: toksearch-api-explore
description: Introspect the installed toksearch and device packages (toksearch_d3d, toksearch_mast, ...) via pydoc/inspect — list exports, walk submodules, and check signatures and docstrings against ground truth
user-invocable: true
license: Apache-2.0
compatibility: Claude Code
metadata:
  author: GA-FDP
  version: "2.0"
  url: https://ga-fdp.github.io/toksearch/
---

# TokSearch API Explorer

## When to Use

Curated docs (the other `toksearch-*` skills, the docs site) describe the API
as of when they were written; the *installed* packages are ground truth and
may be newer or older. Use this skill to introspect them directly:

- When unsure whether a class, function, method, or argument exists in the
  installed version
- When a documented call fails unexpectedly — check the live signature before
  assuming a usage bug
- When looking for Signal subclasses or helpers the curated skills don't
  mention

For *what the API is for and how to use it*, prefer the curated skills
(`toksearch-quickstart`, `toksearch-signal-routing`, `toksearch-pipeline`,
and the per-source skills) — this skill is for verifying and discovering, not
learning.

## Environment

Run snippets with whatever `python` has the FDP stack: plain `python` inside
an activated conda/pixi/`fdp-install` environment, or `pixi run python` /
`fdp run python` otherwise. Examples below use `toksearch`; substitute any
installed device package (`toksearch_d3d`, `toksearch_mast`, ...).

## Recipes

### 1. List top-level exports of a package

```bash
python -c "
import inspect, importlib
pkg = importlib.import_module('toksearch')   # or toksearch_d3d, toksearch_mast
for name in sorted(n for n in dir(pkg) if not n.startswith('_')):
    obj = getattr(pkg, name)
    kind = ('class' if inspect.isclass(obj) else
            'func' if inspect.isfunction(obj) else
            'module' if inspect.ismodule(obj) else type(obj).__name__)
    print(f'  {name:35s} {kind}')
"
```

### 2. Walk all submodules, listing public classes/functions with signatures

```bash
python -c "
import pkgutil, importlib, inspect

def explore(pkg_name):
    try:
        pkg = importlib.import_module(pkg_name)
    except ImportError:
        print(f'=== {pkg_name}: not installed ===')
        return
    print(f'=== {pkg_name} ===')
    for finder, mod_name, ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix=pkg.__name__ + '.'):
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:
            print(f'  [{mod_name}] (import error: {e})')
            continue
        names = [n for n in dir(mod) if not n.startswith('_')
                 and (inspect.isclass(getattr(mod, n))
                      or inspect.isfunction(getattr(mod, n)))]
        if not names:
            continue
        print(f'  [{mod_name}]')
        for n in sorted(names):
            obj = getattr(mod, n)
            try:
                sig = str(inspect.signature(obj))
                if len(sig) > 80:
                    sig = sig[:77] + '...'
            except (ValueError, TypeError):
                sig = ''
            print(f'    {n:30s} {sig}')

for p in ('toksearch', 'toksearch_d3d', 'toksearch_mast'):
    explore(p)
"
```

### 3. Full pydoc for a specific class or module

```bash
python -m pydoc toksearch.Pipeline
python -m pydoc toksearch.signal.mds.MdsSignal
python -m pydoc toksearch_d3d.signal.ptdata.PtDataSignal   # if installed
```

### 4. Inspect a class — MRO, public methods, signatures, first doc lines

```bash
python -c "
import inspect
from toksearch import MdsSignal as C   # swap in any class

print('MRO:', [k.__name__ for k in C.__mro__])
print()
for name, obj in inspect.getmembers(
        C, predicate=lambda o: inspect.isfunction(o) or inspect.ismethod(o)):
    if name.startswith('_'):
        continue
    try:
        sig = inspect.signature(obj)
    except (ValueError, TypeError):
        sig = '(?)'
    print(f'  {name}{sig}')
    doc = inspect.getdoc(obj)
    if doc:
        print(f'    {doc.splitlines()[0]}')
"
```

### 5. Constructor signature and class docstring

```bash
python -c "
import inspect
from toksearch import Pipeline as C   # swap in any class
print(inspect.signature(C.__init__))
print()
print(inspect.getdoc(C))
"
```

### 6. Find where a name actually lives (and the installed version)

```bash
python -c "
import toksearch
from toksearch import MdsSignal
print(MdsSignal.__module__)                  # defining module
print(getattr(toksearch, '__version__', '?'))  # installed version
"
```

## Invoking This Skill

On `/toksearch-api-explore` or a request to check available APIs, run recipe
2 for the installed packages and present the results as a structured table;
use recipes 3–5 for specific class or method questions. Prefer the live API
over memory or curated docs whenever they disagree.
