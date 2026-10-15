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

"""Save and load a run_python namespace, one cloudpickle file per variable.

A checkpoint is written into ``<dir>.new`` and swapped into place, so an
interrupted write leaves the previous checkpoint intact. Files are named by
position (``v0001.pkl``, ...); the index keeps each variable's name, so no
name (``globals()['../x']``, ``"A"`` beside ``"a"`` on a case-insensitive
disk) can escape the folder or collide. Every file is fsynced, and the swap
is ordered so that after a crash at any point one of ``<dir>.new``,
``<dir>`` and ``<dir>.old`` holds a complete checkpoint, found by
``find_checkpoint`` (the index is written last, so a parseable index marks a
complete checkpoint, and a complete ``.new`` is always the newest).

Variables that could not be loaded at the last restore can be carried
forward (``carried=``): their old pickle is copied into the new checkpoint,
so a later restore (after the environment is fixed) tries them again.

Loading a pickle can run arbitrary code: only load folders the user's own
app wrote. Every checkpoint rewrites every variable, even unchanged ones.
"""

import json
import os
import shutil
import types
from pathlib import Path

INDEX = "index.json"
PER_VAR_LIMIT = 1 << 30   # 1 GiB
TOTAL_LIMIT = 4 << 30     # 4 GiB

_REASONS = {
    "too_large": "too large to save",
    "over_total": "the checkpoint size limit was reached",
    "unpicklable": "could not be saved",
    "unloadable": "could not be loaded",
}


class _TooLarge(Exception):
    pass


class _LimitedWriter:
    def __init__(self, f, limit: int):
        self._f = f
        self._limit = limit
        self.written = 0

    def write(self, b):
        self.written += memoryview(b).nbytes
        if self.written > self._limit:
            raise _TooLarge
        return self._f.write(b)


def user_names(ns: dict, baseline) -> list[str]:
    """Names the user created: not baseline, not dunder, not modules."""
    return sorted(k for k, v in ns.items()
                  if k not in baseline and not k.startswith("__")
                  and not isinstance(v, types.ModuleType))


def _sibling(d: Path, suffix: str) -> Path:
    return d.with_name(d.name + suffix)


def _fsync_dir(d: Path) -> None:
    try:
        fd = os.open(d, os.O_RDONLY)
    except OSError:   # e.g. platforms that cannot open a directory
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _sync_file(f) -> None:
    f.flush()
    os.fsync(f.fileno())


def _write_index(directory: Path, index: dict) -> None:
    with open(directory / INDEX, "w") as f:
        f.write(json.dumps(index, indent=1))
        _sync_file(f)


def _copy_synced(src, dst: Path) -> int:
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        shutil.copyfileobj(fin, fout)
        _sync_file(fout)
    return dst.stat().st_size


def _swap(new: Path, out_dir: Path) -> None:
    """Move the complete checkpoint in ``new`` to ``out_dir``, crash-safely."""
    old = _sibling(out_dir, ".old")
    parent = out_dir.parent
    if out_dir.exists():
        if old.exists():
            shutil.rmtree(old)
        os.rename(out_dir, old)
        _fsync_dir(parent)
    # When out_dir is missing, .old may be the only other checkpoint: keep it.
    os.rename(new, out_dir)
    _fsync_dir(parent)
    if old.exists():
        shutil.rmtree(old)
        _fsync_dir(parent)


def _index_ok(d: Path) -> bool:
    try:
        json.loads((d / INDEX).read_text())
    except (OSError, ValueError):
        return False
    return True


def save_namespace(ns: dict, baseline, origins: dict, out_dir, *,
                   per_var_limit: int = PER_VAR_LIMIT,
                   total_limit: int = TOTAL_LIMIT,
                   carried: dict | None = None) -> dict:
    """Checkpoint the user variables of ``ns`` into ``out_dir``; return the index.

    ``carried`` maps a name to ``(path, entry)``: a variable that could not
    be loaded last time. When the name is not in ``ns``, its pickle at
    ``path`` is copied into the checkpoint (entry ``carried: true``).
    """
    import cloudpickle
    out_dir = Path(out_dir)
    new = _sibling(out_dir, ".new")
    carried = dict(carried or {})
    if new.exists():
        if _index_ok(new):
            # A complete .new is the newest checkpoint (an earlier swap was
            # cut short): promote it rather than delete it, and follow any
            # carried pickles that lived in it.
            _swap(new, out_dir)
            carried = {k: (str(out_dir / Path(p).name)
                           if Path(p).resolve().parent == new.resolve()
                           else p, e)
                       for k, (p, e) in carried.items()}
        else:
            shutil.rmtree(new)
    new.mkdir(parents=True)
    entries, total, seq = [], 0, 0
    names = user_names(ns, baseline)

    def room_for(entry):
        room = total_limit - total
        if room <= 0:
            entry["status"] = "over_total"
            return None
        return room

    for name in names:
        value = ns[name]
        entry = {"name": name, "type": type(value).__name__,
                 "cell": origins.get(name), "size": None, "file": None}
        entries.append(entry)
        room = room_for(entry)
        if room is None:
            continue
        limit = min(per_var_limit, room)
        seq += 1
        path = new / f"v{seq:04d}.pkl"
        writer = None
        try:
            with open(path, "wb") as f:
                writer = _LimitedWriter(f, limit)
                cloudpickle.dump(value, writer, protocol=5)
                _sync_file(f)
        except _TooLarge:
            path.unlink(missing_ok=True)
            entry["status"] = ("too_large" if limit == per_var_limit
                               else "over_total")
        except Exception as e:
            path.unlink(missing_ok=True)
            entry["status"] = "unpicklable"
            entry["error"] = f"{type(e).__name__}: {e}"
        else:
            entry.update(status="saved", size=writer.written, file=path.name)
            total += writer.written

    present = set(names)
    for name in sorted(carried):
        if name in present:
            continue
        src, old_entry = carried[name]
        entry = {"name": name, "type": old_entry.get("type"),
                 "cell": old_entry.get("cell"), "size": None, "file": None}
        entries.append(entry)
        if room_for(entry) is None:
            continue
        seq += 1
        path = new / f"v{seq:04d}.pkl"
        try:
            size = _copy_synced(src, path)
        except OSError as e:
            path.unlink(missing_ok=True)
            entry["status"] = "unloadable"
            entry["error"] = f"{type(e).__name__}: {e}"
            continue
        entry.update(status="saved", carried=True, size=size, file=path.name)
        total += size

    index = {"variables": entries, "total_bytes": total,
             "fdp_core_version": os.environ.get("FDP_CORE_VERSION")}
    _write_index(new, index)
    _fsync_dir(new)
    _fsync_dir(new.parent)
    _swap(new, out_dir)
    return index


def find_checkpoint(out_dir) -> Path | None:
    """The directory holding the newest complete checkpoint, if any.

    Tries ``<dir>.new``, ``<dir>`` and ``<dir>.old`` in that order and
    returns the first whose index exists and parses.
    """
    out_dir = Path(out_dir)
    for d in (_sibling(out_dir, ".new"), out_dir, _sibling(out_dir, ".old")):
        if _index_ok(d):
            return d
    return None


def read_index(out_dir) -> dict | None:
    d = find_checkpoint(out_dir)
    return None if d is None else json.loads((d / INDEX).read_text())


def not_saved(index: dict) -> list[dict]:
    """Index entries that were not saved, in the shape load_namespace reports."""
    out = []
    for e in index["variables"]:
        if e["status"] != "saved":
            item = {"name": e["name"], "reason": e["status"],
                    "cell": e.get("cell")}
            if e.get("error"):
                item["error"] = e["error"]
            out.append(item)
    return out


def load_namespace(ns: dict, out_dir) -> dict:
    """Load a checkpoint into ``ns``; report what was and was not restored.

    Each ``unloadable`` entry of ``not_restored`` carries ``path``, the
    absolute path of its pickle, and ``type``, so it can be carried forward.
    """
    import cloudpickle
    d = find_checkpoint(out_dir)
    if d is None:
        return {"found": False, "restored": [], "not_restored": [],
                "origins": {}, "fdp_core_version": None}
    index = json.loads((d / INDEX).read_text())
    restored, origins = [], {}
    not_restored = not_saved(index)
    for e in index["variables"]:
        if e["status"] != "saved":
            continue
        path = (d / e["file"]).resolve()
        try:
            with open(path, "rb") as f:
                ns[e["name"]] = cloudpickle.load(f)
        except Exception as ex:
            not_restored.append({"name": e["name"], "reason": "unloadable",
                                 "cell": e.get("cell"),
                                 "error": f"{type(ex).__name__}: {ex}",
                                 "path": str(path), "type": e.get("type")})
            continue
        restored.append(e["name"])
        if e.get("cell") is not None:
            origins[e["name"]] = e["cell"]
    return {"found": True, "restored": restored, "not_restored": not_restored,
            "origins": origins,
            "fdp_core_version": index.get("fdp_core_version")}


def describe_not_restored(entries: list[dict]) -> str:
    parts = []
    for e in entries:
        reason = _REASONS.get(e["reason"], e["reason"])
        if e.get("cell") is not None:
            reason += f"; bound by cell {e['cell']}"
        parts.append(f"`{e['name']}` ({reason})")
    return ", ".join(parts)
