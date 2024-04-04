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
from typing import List

import pyspark
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession

from .spark_funcs import RunPartSpark, passthrough
from ...slurm import inside_toksearch_submit_job
from ...record.record_set import RecordSet
from ...record import Record


def default_spark_conf(master=None):
    master = master or os.environ.get("SPARK_MASTER", None)
    print("MASTER: {}".format(master))
    conf = SparkConf()
    conf.setMaster(master)
    print(conf.getAll())
    for key, val in list(os.environ.items()):
        conf.setExecutorEnv(key, val)
    return conf


def default_spark_context(master=None):
    master = master or os.environ.get("SPARK_MASTER", None)
    conf = default_spark_conf(master=master)
    sc = SparkContext(conf=conf)
    return sc


def get_or_create_default_spark_context():

    if not is_spark_initialized() and inside_toksearch_submit_job():
        from ..slurm.spark_cluster import SlurmSparkCluster

        cluster = SlurmSparkCluster.from_config()
        cluster.start()
        sc = cluster.spark_context()
    else:
        conf = default_spark_conf()
        sc = SparkContext.getOrCreate()

    sc.setLogLevel("ERROR")
    return sc


def is_spark_initialized():
    with SparkContext._lock:
        return SparkContext._active_spark_context is not None


class SparkBackend:
    def __init__(self, **kwargs):
        """Create a SparkBackend.

        The kwargs are passed to the SparkRecordSet.from_records method when initializing.

        Keyword Arguments:
            sc: SparkContext to use. If not provided, a default SparkContext will be created.
            numparts: Number of partitions to use. If not provided, the default number of partitions
                will be used.
            cache: Whether to cache the RDD. Default is False.
        """
        self.kwargs = kwargs

    def initialize(self, records: List[Record]) -> "SparkRecordSet":
        """Initialize the backend with a list of records.

        Arguments:
            records: List of records to initialize the backend with.

        Returns:
            SparkRecordSet initialized with the records.
        """
        return SparkRecordSet.from_records(records, **self.kwargs)


class SparkRecordSet(RecordSet):

    @classmethod
    def from_records(cls, records: List[Record], **kwargs):
        """Create a SparkRecordSet from a list of records.

        Arguments:
            records: List of records to create the RecordSet from.

        Keyword Arguments:
            sc: SparkContext to use. If not provided, a default SparkContext will be created.
            numparts: Number of partitions to use. If not provided, the default number of partitions
                will be used.
            cache: Whether to cache the RDD. Default is False.
        """

        sc = kwargs.pop("sc", None)
        numparts = kwargs.pop("numparts", None)
        cache = kwargs.pop("cache", False)

        sc = sc or get_or_create_default_spark_context()
        numparts = numparts or sc.defaultParallelism

        numparts = min(len(records), numparts)
        rdd = sc.parallelize(records, numparts)
        return cls(rdd, cache=cache)

    def __init__(self, rdd, cache=False):
        self.rdd = rdd
        self.do_cache = cache
        if self.cache:
            self.cache()

        self._data = None

    def map(self, *operations):
        run_part = RunPartSpark(operations)
        updated_rdd = self.rdd.glom().map(run_part).flatMap(passthrough)

        return self.__class__(updated_rdd, cache=self.do_cache)

    def _to_list(self):
        if not self._data:
            self._data = self.rdd.collect()
        return self._data

    def cache(self):
        self.rdd.cache()

    def to_rdd(self, **kwargs):
        return self.rdd

    def __getitem__(self, index):
        return self._to_list()[index]

    def __iter__(self):
        for el in self._to_list():
            yield el

    def __len__(self):
        return len(self._to_list())

    def cleanup(self, immediate=False):
        """Shut down the SparkContext.

        Arguments:
            immediate: Whether to shut down the SparkContext immediately. If false (default),
                the data will be collected before shutting down the SparkContext and remain
                accessible after the SparkContext is stopped.
        """
        if not immediate:
            # force the rdd to be computed, populating the _data attribute
            # This allows the data to be accessed after the SparkContext is stopped
            self._to_list()

        sc = SparkContext.getOrCreate()
        sc.stop()
