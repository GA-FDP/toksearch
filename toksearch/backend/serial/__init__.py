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

from ...record.record_set import RecordSet
from ...pipeline.pipeline_funcs import _map_multiple


class SerialBackend:
    def initialize(self, records):
        return SerialRecordSet.from_records(records)


class SerialRecordSet(RecordSet):

    @classmethod
    def from_records(cls, records):
        return cls(records)

    def __init__(self, records):
        self.records = records

    def map(self, *operations):
        updated_records = _map_multiple(self.records, operations)
        return self.__class__(updated_records)

    def __getitem__(self, index):
        return self.records[index]

    def __iter__(self):
        for x in self.records:
            yield x

    def __len__(self):
        return len(self.records)

    def cleanup(self):
        pass
