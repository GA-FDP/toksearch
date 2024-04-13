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
import shlex
import subprocess
import unittest
import time
import importlib

from sh import scontrol, srun, grep, awk

#import findspark

#findspark.init()
from pyspark import SparkContext

from .common import ConfigLoader


class SlurmSparkCluster:
    @classmethod
    def from_config(cls, filename=None):

        config_loader = ConfigLoader("spark", filename)
        cluster = cls(host_resolution_func=config_loader.host_resolution_func)

        cluster.master_node.add_srun_options(*config_loader.master_srun_options)
        cluster.master_node.add_start_options(*config_loader.master_start_options)

        for node in cluster.worker_nodes.values():
            node.add_srun_options(*config_loader.worker_srun_options)
            node.add_start_options(*config_loader.worker_start_options)

        return cluster

    def __init__(self, host_resolution_func=None):
        try:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
        except KeyError:
            msg = "Cannot start SlurmSparkCluster outside sbatch or salloc"
            e = Exception(msg)

        self._nodelist = nodelist
        passthrough = lambda x: x
        self.host_resolution_func = host_resolution_func or passthrough

        self.node_names = self._node_names()
        self.ips = [self.host_resolution_func(n) for n in self.node_names]
        print(self.ips)

        self.master_node_name = self.node_names[0]
        self.worker_node_names = self.node_names[:]

        self.master_ip = self.ips[0]
        self.worker_ips = self.ips[:]

        assert len(self.worker_node_names) == len(self.worker_ips)

        self.master_node = SlurmSparkMaster(self.master_node_name, self.master_ip)

        self.worker_nodes = {}
        for node_name, ip in zip(self.worker_node_names, self.worker_ips):
            self.worker_nodes[node_name] = SlurmSparkWorkerNode(
                node_name, ip, self.master_node.master_address
            )

        self.all_nodes = [self.master_node] + list(self.worker_nodes.values())

    def spark_context(self):
        return SparkContext(master=self.master_node.master_address)

    def start(self):
        print("STARTING CLUSTER")
        self.master_node.start()
        time.sleep(2)
        print("Ok, started head node")
        print(self.worker_nodes.keys())
        for node_name, node in self.worker_nodes.items():

            print(f"Starting {node_name}...")
            node.start()
        time.sleep(2)

    def _node_names(self):
        _nodes = scontrol("show", "hostnames", self._nodelist)
        nodes = [n for n in _nodes.split("\n") if n]
        return nodes


class SlurmSparkNode:

    def __init__(self, node_name, node_ip, spark_home=None):
        self.name = node_name
        self.node_ip = node_ip

        self.spark_home = spark_home or os.getenv("SPARK_HOME", None)
        #if self.spark_home is None:
        #    raise (Exception("Could not resolve SPARK_HOME"))

        self._additional_start_options = []
        self._additional_srun_options = []

    def start(self):
        os.environ["SPARK_NO_DAEMONIZE"] = "1"

        srun_options = ["--nodes=1", "--ntasks=1", "-w", self.name]

        args = (
            srun_options
            + self._additional_srun_options
            + ["spark-class"]
            + [self.spark_class()]
            + self.start_options()
            + self._additional_start_options
            + self.start_args()
        )

        print(args)
        srun(*args, _bg=True)

    def add_srun_options(self, *options):
        self._additional_srun_options += options

    def add_start_options(self, *options):
        self._additional_start_options += options


class SlurmSparkMaster(SlurmSparkNode):
    def __init__(self, node_name, node_ip, spark_home=None, port=7077):
        super().__init__(node_name, node_ip, spark_home=spark_home)
        self.port = port
        self.master_address = f"spark://{self.node_ip}:{self.port}"

    def spark_class(self):
        return "org.apache.spark.deploy.master.Master"

    def start_args(self) -> list:
        return []

    def start_options(self) -> list:
        return [
            "--host", self.node_ip,
            "--port", self.port,
        ]
        
    def start(self):
        os.environ["SPARK_MASTER_HOST"] = self.node_ip
        print("MASTER IP", self.node_ip)
        super().start()


class SlurmSparkWorkerNode(SlurmSparkNode):
    def __init__(self, node_name, node_ip, master_address, spark_home=None):
        super().__init__(node_name, node_ip, spark_home=spark_home)
        self.master_address = master_address

    def spark_class(self):
        return "org.apache.spark.deploy.worker.Worker"

    def start_args(self) -> list:
        return [self.master_address]

    def start_options(self) -> list:
        return ["-i", self.node_ip]
