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

import sys
import gc
import time
import unittest
import ray

import itertools
import numpy as np
import socket
import tempfile

# Apparently, python 3.9 changed how it deals with Iterable
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from ...utilities.utilities import partition_it, chunk_it, capture_exception

from ...slurm import inside_toksearch_submit_job

import logging

logging.basicConfig(level=logging.ERROR)


@ray.remote
def _apply_func(func, *x, force_gc=False):
    intermediate_results = func(x)

    if not isinstance(intermediate_results, Iterable):
        raise BadFunctionError("func must return an iterable")

    results = [ray.put(el) for el in intermediate_results]

    if force_gc:
        gc.collect()

    return results


@ray.remote
def _passthrough(x):
    return x


@ray.remote
def _apply_to_element(func, element):
    print("result in apply", result)
    time.sleep(2)
    result = func(element)
    return result


@ray.remote
def _put_list(*l):
    ids = [ray.put(el) for el in l]
    return ids


class BadFunctionError(Exception):
    pass


class RayDD:
    @classmethod
    def from_iterator(cls, elements, **kwargs):
        """Instantiate from a list of regular objects (non-ray ids)

        Parameters:
            elements (list): List of regular objects

        Keyword Arguments:
            numparts (int): Number of partitions to use when mapping
            batch_size (int): Number of elements to process in each batch
            verbose (bool): Whether to print verbose output
            placement_group_func (callable): A function that returns a placement group
            memory_per_task (int): Memory to allocate to each task in bytes
            ray_init_kwargs (dict): Keyword arguments to pass to ray.init
        """

        ray_init_kwargs = kwargs.pop("ray_init_kwargs", {})
        ray_init_kwargs = ray_init_kwargs if ray_init_kwargs else {}

        if not ray.is_initialized():

            # Create a temp dir to avoid permissions issues when using
            # ray.init with the default arguments
            if "_temp_dir" not in ray_init_kwargs:
                dirname = tempfile.mkdtemp()
                print("Initializing ray with _temp_dir = {}".format(dirname))
                ray_init_kwargs["_temp_dir"] = dirname

            if "logging_level" not in ray_init_kwargs:
                ray_init_kwargs["logging_level"] = logging.ERROR

            if inside_toksearch_submit_job():
                from ...slurm.ray_cluster import SlurmRayCluster

                cluster = SlurmRayCluster.from_config(
                    temp_dir=ray_init_kwargs["_temp_dir"]
                )
                cluster.start()
                cluster.ray_init(**ray_init_kwargs)
                time.sleep(2)
            else:
                ray.init(**ray_init_kwargs)

        ids = [ray.put(el) for el in elements]
        return cls(ids, **kwargs)

    def __init__(self, ids, **kwargs):
        """Instantiate from a list of ray object ids

        Parameters:
            ids (list): List of ray object ids


        Keyword Arguments:
            numparts (int): Number of partitions to use when mapping
            batch_size (int): Number of elements to process in each batch
            verbose (bool): Whether to print verbose output
            placement_group_func (callable): A function that returns a placement group
            memory_per_task (int): Memory to allocate to each task in bytes
        """
        self._object_ids = ids

        self.numparts = kwargs.pop("numparts", None)
        self.batch_size = kwargs.pop("batch_size", None)
        self.verbose = kwargs.pop("verbose", False)
        self.placement_group_func = kwargs.pop("placement_group_func", None)
        self.memory_per_task = kwargs.pop("memory_per_task", None)

        if kwargs:
            raise ValueError(f"Unrecognized keyword arguments: {kwargs.keys()}")

    def map(self, func):
        """
        Apply func, which is assumed to take in an iterable and return an iterable

        Parameters:
            func (callable): Must accept a single argument that is an iterable
                and return an iterable
        """
        if self.placement_group_func:
            placement_group = self.placement_group_func()
            ray.get(placement_group.ready())
        else:
            placement_group = None
            while "CPU" not in ray.available_resources():
                pass

        batch_size = self.batch_size if self.batch_size else len(self._object_ids)
        batch_size = min(len(self._object_ids), batch_size)

        batches = partition_it(self._object_ids, batch_size)

        nested_results = []
        for i, batch in enumerate(batches):
            if self.verbose:
                print("*" * 80)
                print(f"BATCH {i+1}/{len(batches)}")
            batch_result = self._map_batch(func, batch, placement_group=placement_group)
            nested_results.append(batch_result)

        results = list(itertools.chain.from_iterable(nested_results))

        kwargs_copy = {k: v for k, v in self.__dict__.items() if k != "_object_ids"}

        return self.__class__(results, **kwargs_copy)

    def _map_batch(self, func, ids, placement_group=None):

        numparts = self.numparts
        verbose = self.verbose

        num_cpus = int(ray.available_resources()["CPU"])

        numparts = numparts if numparts else len(ids)

        # list of lists of form [[id1, id2], [id3, id4], ...]
        chunks = chunk_it(ids, numparts)
        chunk_lengths = [len(chunk) for chunk in chunks]
        median_chunk = int(np.median(chunk_lengths))
        if self.verbose:
            print(f"NUM CPUS: {num_cpus}")
            print(f"NUM PARTITIONS: {len(chunks)}")
            print(f"MEDIAN PARTITION SIZE: {median_chunk}")

        opts = {}
        if placement_group:
            opts["placement_group"] = placement_group

        tasks = []

        for chunk in chunks:
            if self.memory_per_task:
                opts["memory"] = len(chunk) * self.memory_per_task
            f = _apply_func.options(**opts)
            tasks.append(f.remote(func, *chunk))

        nested_results = ray.get(tasks)

        results = list(itertools.chain.from_iterable(nested_results))

        return results

    def object_ids(self):
        return self._object_ids

    def get(self):
        return ray.get(self._object_ids)

    def __str__(self):
        return f"{type(self)} object with {len(self._object_ids)} elements"
