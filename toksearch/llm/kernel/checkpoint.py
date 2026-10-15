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
interrupted write leaves the previous checkpoint intact. Loading a pickle can
run arbitrary code: only load folders the user's own app wrote.
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


def _write_index(directory: Path, index: dict) -> None:
    (directory / INDEX).write_text(json.dumps(index, indent=1))


def save_namespace(ns: dict, baseline, origins: dict, out_dir, *,
                   per_var_limit: int = PER_VAR_LIMIT,
                   total_limit: int = TOTAL_LIMIT) -> dict:
    """Checkpoint the user variables of ``ns`` into ``out_dir``; return the index."""
    import cloudpickle
    out_dir = Path(out_dir)
    new = _sibling(out_dir, ".new")
    if new.exists():
        shutil.rmtree(new)
    new.mkdir(parents=True)
    entries, total = [], 0
    for name in user_names(ns, baseline):
        value = ns[name]
        entry = {"name": name, "type": type(value).__name__,
                 "cell": origins.get(name), "size": None, "file": None}
        room = total_limit - total
        if room <= 0:
            entry["status"] = "over_total"
            entries.append(entry)
            continue
        limit = min(per_var_limit, room)
        path = new / f"{name}.pkl"
        writer = None
        try:
            with open(path, "wb") as f:
                writer = _LimitedWriter(f, limit)
                cloudpickle.dump(value, writer, protocol=5)
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
        entries.append(entry)
    index = {"variables": entries, "total_bytes": total,
             "fdp_core_version": os.environ.get("FDP_CORE_VERSION")}
    _write_index(new, index)
    old = _sibling(out_dir, ".old")
    if old.exists():
        shutil.rmtree(old)
    if out_dir.exists():
        out_dir.rename(old)
    new.rename(out_dir)
    if old.exists():
        shutil.rmtree(old)
    return index


def find_checkpoint(out_dir) -> Path | None:
    """The directory holding the newest complete checkpoint, if any."""
    out_dir = Path(out_dir)
    for d in (out_dir, _sibling(out_dir, ".old")):
        if (d / INDEX).exists():
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
    """Load a checkpoint into ``ns``; report what was and was not restored."""
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
        try:
            with open(d / e["file"], "rb") as f:
                ns[e["name"]] = cloudpickle.load(f)
        except Exception as ex:
            not_restored.append({"name": e["name"], "reason": "unloadable",
                                 "cell": e.get("cell"),
                                 "error": f"{type(ex).__name__}: {ex}"})
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
