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

import copy
from ..record import Record, InvalidShotNumber


class PipelineSource:
    def __init__(self, input_items, batch_size=None):
        self._records = []
        for item in input_items:
            record = None
            if isinstance(item, Record):
                record = item
            if record is None:
                try:
                    record = Record(item)
                except InvalidShotNumber:
                    record = Record.from_dict(item)
                except Exception as e:
                    print(f"ERROR: Unable to create record from {item}")
                    raise (e)

            self.add_record(record)

        self._batch_size = batch_size

    @property
    def records(self):
        return copy.deepcopy(self._records)

    def add_record(self, record):
        self._records.append(record)

    @property
    def batch_size(self):
        return self._batch_size or len(self.records)

    @property
    def batches(self):
        return partition_it(self.records, self.batch_size)

    def initialize_result(self, backend):
        return backend.initialize(self.records)
