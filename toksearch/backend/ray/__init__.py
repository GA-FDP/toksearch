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

"""
This module contains the RayBackend and RayRecordSet classes, which are used to
interface with Ray for distributed computing.

The RayBackend class is used to initialize a RayRecordSet from a set of records.

The RayRecordSet class is used to represent a set of records that are distributed
across a Ray cluster, and provides methods for retrieving and ray object ids, 
which is useful for doing distributed operations on the records outside of
toksearch.
"""
from typing import List
import ray

from .raydd import RayDD
from .ray_funcs import RunPartRay
from ...record.record_set import RecordSet
from ...record import Record


# TODO: Put in defaults for numparts, batch_size, verbose, placement_group_func, memory_per_task, ray_init_kwargs
class RayBackend:
    def __init__(self, **kwargs):
        """Create a RayBackend.

        The kwargs are passed to the RayRecordSet.from_records method when initializing.

        Keyword Arguments:
            numparts (int): Number of partitions to use when mapping
            batch_size (int): Number of elements to process in each batch
            verbose (bool): Whether to print verbose output
            placement_group_func (callable): A function that returns a placement group
            memory_per_task (int): Memory to allocate to each task in bytes
            ray_init_kwargs (dict): Keyword arguments to pass to ray.init

        """
        self.kwargs = kwargs
        self.kwargs["memory_per_task"] = self.kwargs.pop("memory_per_shot", None)

    def initialize(self, records: List[Record]) -> "RayRecordSet":
        """Initialize the backend with a list of records.

        Arguments:
            records: List of records to initialize the backend with.

        Returns:
            RayRecordSet initialized with the records.
        """
        return RayRecordSet.from_records(records, **self.kwargs)


class RayRecordSet(RecordSet):

    def __init__(self, raydd: RayDD):
        """Create a RayRecordSet from a RayDD."""
        self.raydd = raydd

    @classmethod
    def from_records(cls, records: List[Record], **kwargs):
        """Create a RayRecordSet from a list of records.

        Arguments:
            records: List of records to create the RecordSet from.

        Keyword Arguments:
            numparts (int): Number of partitions to use when mapping
            batch_size (int): Number of elements to process in each batch
            verbose (bool): Whether to print verbose output
            placement_group_func (callable): A function that returns a placement group
            memory_per_task (int): Memory to allocate to each task in bytes
            ray_init_kwargs (dict): Keyword arguments to pass to ray.init
        """
        raydd = RayDD.from_iterator(records, **kwargs)
        return cls(raydd)

    def map(self, *operations):
        run_part_ray = RunPartRay(operations)
        new_raydd = self.raydd.map(run_part_ray)
        return self.__class__(new_raydd)

    # def map(self, func):
    #    new_raydd = self.raydd.map(func)
    #    return self.__class__(new_raydd)

    def object_ids(self):
        return self.raydd.object_ids()

    def to_raydd(self, **kwargs):
        return self.raydd

    def __getitem__(self, index):
        obj_id = self.raydd.object_ids()[index]
        return ray.get(obj_id)

    def __iter__(self):
        for el in self.raydd.get():
            yield el

    def __len__(self):
        return len(self.raydd.object_ids())

    def _to_list(self):
        return self.raydd.object_ids()

    def cleanup(self):
        """Shutdown the Ray cluster.

        All Ray objects will be lost after this method is called, so only call it
        after all records have been copied back to the local machine, either by
        grabbing an index, or slice of the RayRecordSet or converting it to a list.
        """
        ray.shutdown()
