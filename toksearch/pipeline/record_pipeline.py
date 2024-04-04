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
Module for processing sets of Record data in a series of steps

The principle abstraction is the Pipeline class, which provides a API for processing
data in a series of steps, including both predefined methods and arbitrary
user-defined functions.
"""

import os
import copy
import importlib

# Apparently, python 3.9 changed how it deals with Iterable
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import numpy as np
import itertools
import xarray as xr
import multiprocessing
from typing import List


from ..utilities.utilities import (
    chunk_it,
    partition_it,
    capture_exception,
)

from .pipeline_funcs import (
    _map_multiple,
    _map_single,
    _apply_operations,
    _SafeMap,
    _SafeFetch,
    _SafeFetchAsXarray,
    _PipelineKeep,
    _PipelineAlign,
    _PipelineWhere,
)

from .align import XarrayAligner

from ..record import Record, InvalidShotNumber
from ..record.record_set import RecordSet

from ..backend.ray import RayBackend
from ..backend.spark import SparkBackend
from ..backend.serial import SerialBackend
from ..backend.multiprocessing import MultiprocessingBackend

from .pipeline_source import PipelineSource


class MissingColumnName(Exception):
    pass


class Pipeline:
    """Pipeline class for processing data

    The Pipeline class is used to process data in a series of steps. The
    Pipeline class is designed to be used in a functional style, where
    operations are added to the pipeline in a linear fashion. The pipeline
    can then be applied to a set of Records objects. The pipeline can be applied
    in serial, using Ray, using Spark, or using multiprocessing, or using
    a custom backend. The results of the pipeline are stored in an object that
    is derived from a RecordSet, which is a list-like object that can be used
    to access the results of the pipeline.

    Methods:
        from_sql: Initialize a Pipeline using the results of an sql query
        __init__: Initialize a Pipeline object
        fetch: Add a signal to be fetched by the pipeline
        fetch_dataset: Create an xarray dataset field in the record
        map: Apply a function to the records from of the previous step in the pipeline,
            modifying the record in place
        keep: Keep only the fields specified in the list
        align: Align an xarray dataset with a specified set of coordinates
            (typically times)
        where: Apply a function to the records of the previous step in the pipeline,
            and keep the record if the result is truthy, remove it otherwise
        compute_shot: Run the pipeline for a single shot, returning a record object
        compute_record: Apply the pipeline to a record object
        compute: Apply the pipeline using a backend
        compute_serial: Apply the pipeline serially on the local host
        compute_ray: Apply the pipeline using Ray
        compute_spark: Apply the pipeline using Spark
        compute_multiprocessing: Apply the pipeline using multiprocessing

    """

    @classmethod
    def from_sql(cls, conn, query, *query_params, batch_size=None):
        '''
        Initialize a Pipeline using the results of an sql query

        Arguments:
            conn (Python DB API-compliant Connection object): A Connection object,
                for example, from pyodbc, pymssql, sqlite, etc...
            query (str): A query string. At a minimum, the query must produce
                rows with the column "shot". The query cannot have columns
                "key" or "errors" as those are reserved words in a Pipeline.
                Additionally, if the query has any unnamed column a
                MissingColumnName exception will be raised.
            query_params (arbitrary type): Optional. Used to pass parameters
                into a query. The exact query syntax is db-dependent. For SQL
                server (used for the d3drdb), use either %d or %s as
                placeholders (it doesn't matter which). For sqlite, ? is
                used as the placeholder.

        Keyword Arguments:
            batch_size (int): If set to an integer, limits the number of
                shots being processed at once. If not set, all shots are
                done in a single batch. This is useful for very large
                jobs that need more memory than available on the host or
                cluster being used.


        Examples:
            ```python
            from toksearch import Pipeline
            from toksearch.sql.mssql import connect_d3drdb

            # See documentation for connect_d3drdb for more details
            conn = connect_d3drdb()

            # Query without parameters
            query = "select shot from shots_type where shot_type = 'plasma'"
            pipe = Pipeline.from_sql(conn, query)

            # Query with parameters, limiting to shot numbers greater than a
            # threshold
            threshold = 180000
            query = """
                select shot
                from shots_type
                where shot_type = 'plasma' and shot > %d
                """
            pipe = Pipeline.from_sql(conn, query, threshold)
            ```
        '''

        cursor = conn.cursor()
        cursor.execute(query, query_params)

        desc = cursor.description
        column_names = [col[0] for col in desc]

        for col in column_names:
            if col == "":
                raise MissingColumnName("Cannot use anonymous field names")

        results = [dict(zip(column_names, row)) for row in cursor.fetchall()]

        return cls(results, batch_size=batch_size)

    def __init__(self, parent, batch_size=None):
        """
        Instantiate a Pipeline object

        Arguments:
            parent (Iterable or Pipeline or PipelineSource):
                If parent is an Iterable, then the elements of the Iterable
                must be one of three types:
                    1) A integer shot number
                    2) A dictionary containing at least the field "shot"
                       (and not the fields "key" or "errors")
                    3) A Record object.

                If the parent is another Pipeline, then the newly constructed
                Pipeline will act as a continuation of the parent.

                The parent can also be a PipelineSource, although typically this
                is handled internally.


        Keyword Arguments:
            batch_size (int): If set to an integer, limits the number of
                shots being processed at once. If not set, all shots are
                done in a single batch. This is useful for very large
                jobs that need more memory than available on the host or
                cluster being used.

        """

        self.batch_size = batch_size

        if isinstance(parent, Pipeline):
            self.parent = parent.parent
            self.do_shot_cleanups = parent.do_shot_cleanups
            self.do_cleanups = parent.do_cleanups
            self._operations = parent._operations.copy()

        else:
            if isinstance(parent, RecordSet):
                pass
            elif isinstance(parent, Iterable):
                parent = PipelineSource(parent, batch_size=batch_size)

            self.parent = parent
            self.do_shot_cleanups = False
            self.do_cleanups = False
            self._operations = []

    # @property
    # def batch_size(self):
    #    return self.parent.batch_size

    def fetch(self, name, signal):
        """Add a signal to be fetched by the pipeline

        Appends a field (name) to the record being processed by the pipeline

        Arguments:
            name: String
            signal: An object derived from AbstractSignal
        """
        self._append_operation(_SafeFetch(name, signal))

    def fetch_dataset(self, name, signals, append=True):
        """
        Create an xarray dataset field called name in the record.

        signal_dict is a dict of the form name: signal. Each key in
        signal_dict will become the name of a data var in the resulting
        dataset.


        If the append keyword is set to True, then if name exists, it
        will be appended to. Otherwise, a new field is created (and
        any existing data in that field will be lost).
        """

        for signame, signal in signals.items():
            f = _SafeFetchAsXarray(name, signame, signal, append)
            self._append_operation(f)

    def map(self, func):
        """
        Apply func to result of the previous step in the pipeline

        Func is expected to be of the form func(record) -> record
        """
        f = _SafeMap(func)
        self._append_operation(f)

    def keep(self, fields: List[str]):
        """Keep only the specified fields in the record

        Arguments:
            fields: List of fields to keep
        """
        self.map(_PipelineKeep(fields))

    def align(
        self,
        ds_name: str,
        align_with,
        dim="times",
        method="pad",
        extrapolate=True,
        interp_kwargs=None,
    ):
        """Align an xarray dataset with a specified set of coordinates

        Arguments:
            ds_name: Name of the dataset in the record
            align_with: The coordinates to align with. This can be a string
                (which will be interpreted as a field in the dataset), a list
                (which will be interpreted as a list of values), a numpy array,
                a callable (which will be called with the dataset and the dim
                as arguments), or a numeric value (which will be interpreted as
                a sample period).

        Keyword Arguments:
            dim: The dimension to align along. Default is 'times'
            method: The method to use for alignment. Default is 'pad', which
                zero-order holds the data. Other options include 'linear' and
                'cubic'.
            extrapolate: Whether to extrapolate data. Default is True.
            interp_kwargs: Keyword arguments to pass to the interpolation
                function provided by xarray. Default is None.

        """
        aligner = XarrayAligner(
            align_with,
            dim=dim,
            method=method,
            extrapolate=True,
            interp_kwargs=interp_kwargs,
        )

        self.map(_PipelineAlign(ds_name, aligner))

    def where(self, func):
        """
        Apply a func to result in the previous step in the pipeline. If
        the result of the func is truthy, then keep the record in the pipeline.
        Otherwise, purge the record from the pipeline.

        func must be of the form func(record) -> Truth-like value
        """
        self._append_operation(_PipelineWhere(func))

    def compute_shot(self, shot):
        """Run the pipeline for a single shot, returning a record object

        Note that an empty record object is first created, and the acted on
        by the pipeline. If there are prerequiste fields in the record, then the
        method compute_record should be used to pass a record object directly to
        the pipeline.
        """
        record = Record(shot)
        return self.compute_record(record)

    def compute_record(self, record):
        """Apply the pipeline to a record object"""
        return _map_single(record, self._operations)
        # return self._map_single_shot(record)

    def compute(self, backend):
        if isinstance(self.parent, RecordSet):
            initial_result = self.parent
        else:
            initial_result = self.parent.initialize_result(backend)

        return initial_result.map(*self._operations)

    ####################### SERIAL ######################

    def compute_serial(self):
        """Apply the pipeline serially on the local host

        Returns a SerialRecordSet object
        """
        backend = SerialBackend()
        return self.compute(backend)

    ####################### RAY  ######################

    def compute_ray(
        self,
        numparts=None,
        placement_group_func=None,
        memory_per_shot=None,
        **ray_init_kwargs,
    ):
        backend = RayBackend(
            numparts=numparts,
            placement_group_func=placement_group_func,
            memory_per_shot=memory_per_shot,
            **ray_init_kwargs,
        )

        return self.compute(backend)

        return raydd

    ####################### SPARK  ######################
    def compute_spark(self, **kwargs):
        """Apply the pipeline using Spark

        Keyword Arguments:
            sc: SparkContext to use. If not provided, a default SparkContext will be created.
            numparts: Number of partitions to use. If not provided, the default number of partitions
                will be used.
            cache: Whether to cache the RDD. Default is False.

        Returns:
            SparkRecordSet object
        """
        backend = SparkBackend(**kwargs)
        return self.compute(backend)

    ####################### MULTIPROCESSING  ######################
    def compute_multiprocessing(self, **kwargs):
        backend = MultiprocessingBackend(**kwargs)
        return self.compute(backend)

    ####################### Private methods ######################

    def _map_record_list(self, record_list):
        """Give a list of record objects, apply operations"""

        res = _map_multiple(record_list, self._operations)
        return res

    def _append_operation(self, func):
        self._operations.append(func)
