# MDSplus Data Access

## Reading a pinned version

A record can name which version of a shot to read, and `MdsSignal` honours it
over both the `fdp://` and `pelican://` transports:

```python
Pipeline([
    {"shot": 165920, "version": 2},
    {"shot": 165921, "snapshot": "catalog_20260907T232802Z"},
])
```

`version` pins one shot. `snapshot` resolves through that catalog rather than
the newest, so a single recorded value reproduces a whole campaign. With
neither, the newest version is read and anything the store has not absorbed
still resolves from the unversioned archive.

A pin is a **guarantee**. An unsatisfiable one raises `StoreVersionError`
rather than answering from a different version or from the archive — a pin
exists so a rerun can prove it read the same bytes, and a silent substitution
would defeat it.

Reading from the store requires `ptdata >= 2.7.0` and a deployment that
declares where its store lives: over `fdp://` the origin declares it, and a
local read takes it from `FDP_VIEWS_ROOT`. Without one, unpinned reads are
unchanged and pinned reads raise.

## API

::: toksearch.MdsSignal
    handler: python
    options:
        show_root_heading: True

::: toksearch.signal.mds.MdsTreePath
    handler: python
    options:
        show_root_heading: True
