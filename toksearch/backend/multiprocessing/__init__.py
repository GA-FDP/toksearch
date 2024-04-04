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

import os

from joblib import Parallel, delayed

from ...record.record_set import RecordSet
from ...pipeline.pipeline_funcs import _map_single, _map_multiple


DEFAULT_NUM_WORKERS = max(os.cpu_count() // 2, 1)


class _Mapper:
    def __init__(self, operations):
        self.operations = operations

    def __call__(self, record):
        return _map_single(record, self.operations)


class MultiprocessingBackend:

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def initialize(self, records):
        return MultiprocessingRecordSet.from_records(records)


class MultiprocessingRecordSet(RecordSet):

    @classmethod
    def from_records(cls, records, **kwargs):
        return cls(records, **kwargs)

    def __init__(self, records, **kwargs):
        self.kwargs = kwargs
        self.records = records

    def map(self, *operations):
        num_workers = self.kwargs.get("num_workers", DEFAULT_NUM_WORKERS)
        batch_size = self.kwargs.get("batch_size", "auto")

        # Use joblib to parallelize the mapping
        # Just shutdown the pool after we're done
        with Parallel(n_jobs=num_workers, batch_size=batch_size) as parallel:
            updated_records = parallel(
                delayed(_Mapper(operations))(record) for record in self.records
            )

        # Consolidate the results by just calling empty operations on the updated records
        updated_records = _map_multiple(updated_records, [])

        return self.__class__(updated_records, **self.kwargs)

    def __getitem__(self, index):
        return self.records[index]

    def __iter__(self):
        for x in self.records:
            yield x

    def __len__(self):
        return len(self.records)

    def cleanup(self):
        pass
