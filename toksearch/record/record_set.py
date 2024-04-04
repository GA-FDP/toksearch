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

from abc import ABC, abstractmethod


class RecordSet(ABC):
    """Abstract base class for a set of Records

    This class provides a common interface for a set of Records
    that are produced by the various execution backends.

    Methods:
        __len__: Returns the number of records in the set
        __getitem__: Returns a record by index
        __iter__: Iterates over the records in the set as a generator
        map: Applies one or more functions to the records
        cleanup: Cleans up any resources used by the RecordSet, such as
            shutting down a SparkContext or Ray cluster
    """

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def map(self, *operations):
        pass

    @abstractmethod
    def cleanup(self, **kwargs):
        pass
