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

import inspect
import os

import xarray as xr
from ..signal.signal import SignalRegistry
from ..provenance.context import OpSpec
from ..provenance.hashing import callable_spec
from .writers import extension_for, write_object, writer_for


class _SafeMap(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, record):
        try:
            self.func(record)
        except Exception as e:
            name = getattr(self.func, "__name__", repr(self.func))
            record.set_error(name, e)
        return record

    def spec(self):
        # keep() and align() are implemented as map(), so delegate to the
        # wrapped callable when it can describe itself. Otherwise this really
        # is a user map function.
        inner = getattr(self.func, "spec", None)
        if callable(inner):
            return inner()
        return OpSpec("map", {"func": callable_spec(self.func)})


class _SafeFetch(object):

    def __init__(self, name, signal):
        self.signal = signal
        self.name = name

    def __call__(self, record):
        try:
            record[self.name] = self.signal.fetch(record.shot)
        except Exception as e:
            record.set_error(self.name, e)
            record[self.name] = None
        return record

    def spec(self):
        return OpSpec("fetch", {"name": self.name, "signal": self.signal.spec()})


class _SafeFetchAsXarray(object):
    def __init__(self, ds_name, signame, signal, append):
        self.ds_name = ds_name
        self.signame = signame
        self.signal = signal
        self.append = append

    def new_ds(self, shot):
        return xr.Dataset(coords={"shot": ("shot", [shot])})

    def __call__(self, record):
        try:
            if (not self.append) or (self.ds_name not in record):
                record[self.ds_name] = self.new_ds(record.shot)
            # Make sure that val is a DataArray
            # doing xr.DataArray(data_array) is
            # basically idempotent
            val = self.signal.fetch_as_xarray(record.shot)
            record[self.ds_name] = xr.merge(
                [record[self.ds_name], val.to_dataset(name=self.signame)],
                join="outer",
            )
        except Exception as e:
            record.set_error(self.ds_name, e)
        return record

    def spec(self):
        return OpSpec(
            "fetch_dataset",
            {
                "ds_name": self.ds_name,
                "signame": self.signame,
                "append": self.append,
                "signal": self.signal.spec(),
            },
        )


class _PipelineKeep(object):
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, rec):
        rec.keep(self.fields)

    def spec(self):
        return OpSpec("keep", {"fields": list(self.fields)})


class _PipelineAlign(object):
    def __init__(self, ds_name, aligner):
        self.ds_name = ds_name
        self.aligner = aligner

    def __call__(self, record):
        record[self.ds_name] = self.aligner(record[self.ds_name])

    def spec(self):
        # NOT repr(self.aligner): XarrayAligner defines no __repr__, so its
        # repr embeds a memory address and two identical aligners compare
        # unequal. canonical_json cannot rescue this -- it only rewrites
        # non-serializable *values*, and a repr string is already a str, so
        # the address would pass straight into the hash. Describe the
        # aligner's actual configuration instead.
        aligner = self.aligner
        align_with = getattr(aligner, "align_with", None)
        if callable(align_with):
            align_with = callable_spec(align_with)
        elif hasattr(align_with, "tolist"):
            align_with = align_with.tolist()
        return OpSpec(
            "align",
            {
                "ds_name": self.ds_name,
                "align_with": align_with,
                "dim": getattr(aligner, "dim", None),
                "method": getattr(aligner, "method", None),
                "extrapolate": getattr(aligner, "extrapolate", None),
                "interp_kwargs": getattr(aligner, "interp_kwargs", None),
            },
        )


class _PipelineWhere(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, record):
        func = self.func
        try:
            if func(record):
                return record
            else:
                return None
        except Exception as e:
            record.set_error("where", e)
            return None

    def spec(self):
        return OpSpec("where", {"func": callable_spec(self.func)})


class _SafeWrite(object):
    """Write one file per record, in the worker, and record where it went.

    This operation deliberately does *not* talk to a provenance backend. It
    runs inside worker processes, which may be forked; cmflib and DVC must
    never be touched there. It records the path on the record instead, and the
    driver collects those paths after compute returns.
    """

    def __init__(self, directory, field=None, fields=None, fmt=None,
                 func=None, name=None, track="directory",
                 path_field="output_path", on_error="skip"):
        self.on_error = on_error
        self.directory = os.path.abspath(directory)
        self.fields = [field] if field is not None else (list(fields) if fields else [])
        self.fmt = fmt
        self.func = func
        self.name = name
        self.track = track
        self.path_field = path_field

    def _basename(self, record):
        if self.name is not None:
            return str(self.name(record))
        return str(record.shot)

    def _payload(self, record):
        """Return (object_to_write, already_written_path)."""
        if self.func is not None:
            if len(inspect.signature(self.func).parameters) >= 2:
                path = os.path.join(self.directory, self._basename(record))
                return None, self.func(record, path)
            return self.func(record), None

        missing = [f for f in self.fields if f not in record]
        if missing:
            raise KeyError(f"record has no field(s) {missing}")

        values = [record[f] for f in self.fields]
        if len(values) == 1:
            return values[0], None

        return xr.merge(values, join="outer"), None

    def __call__(self, record):
        # A record that already failed an earlier operation must not be
        # written by default. Its fields are whatever survived the failure --
        # a fetch_dataset result that never went through the map that was
        # supposed to reduce it, say -- so the file would look valid while
        # silently missing a pipeline stage, and the output directory's DVC
        # hash would then vouch for it. Skipping keeps the recorded artifact
        # honest: it covers exactly the shots that completed.
        if self.on_error == "skip" and record.get("errors", None):
            return record

        try:
            os.makedirs(self.directory, exist_ok=True)
            obj, written_path = self._payload(record)

            if written_path is None:
                fmt = self.fmt or writer_for(obj=obj).fmt
                path = os.path.join(
                    self.directory, self._basename(record) + extension_for(fmt)
                )
                write_object(obj, path, fmt=fmt)
            else:
                path = str(written_path)

            record[self.path_field] = path
            record["_toksearch_write_dir"] = self.directory
        except Exception as e:
            record.set_error("write", e)
        return record

    def spec(self):
        return OpSpec(
            "write",
            {
                "directory": self.directory,
                "fields": list(self.fields),
                "fmt": self.fmt,
                "func": callable_spec(self.func),
                "name": callable_spec(self.name),
                "track": self.track,
                "path_field": self.path_field,
                "on_error": self.on_error,
            },
        )


def _map_multiple(record_list, operations):
    res = []
    for record in record_list:
        if record is None:
            continue

        updated_record = _map_single(record, operations)
        if updated_record:
            res.append(updated_record)

    SignalRegistry().cleanup()

    return res


def _map_single(record, operations):
    updated_record = _apply_operations(record, operations)
    shot = record.shot
    SignalRegistry().cleanup_shot(shot)

    return updated_record


def _apply_operations(record, operations):
    for func in operations:
        record = func(record)
        if not record:
            break

    return record
