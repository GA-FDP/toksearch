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

import xarray as xr
from ..signal.signal import SignalRegistry


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
                [record[self.ds_name], val.to_dataset(name=self.signame)]
            )
        except Exception as e:
            record.set_error(self.ds_name, e)
        return record


class _PipelineKeep(object):
    def __init__(self, fields):
        self.fields = fields

    def __call__(self, rec):
        rec.keep(self.fields)


class _PipelineAlign(object):
    def __init__(self, ds_name, aligner):
        self.ds_name = ds_name
        self.aligner = aligner

    def __call__(self, record):
        record[self.ds_name] = self.aligner(record[self.ds_name])


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
