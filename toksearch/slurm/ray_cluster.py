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
import yaml
import shlex
import subprocess
import unittest
import time
import ray
import psutil

import uuid
from sh import scontrol, srun, grep, awk

from .common import ConfigLoader




class TestSagaIpAddress(unittest.TestCase):
    def test_saga_ip_address(self):
        ip = saga_ip_address("saga01")

        self.assertTrue(isinstance(int(ip.split(".")[0]), int))


class SlurmRayCluster:
    @classmethod
    def from_config(cls, filename=None, temp_dir=None):

        config_loader = ConfigLoader("ray", filename)
        cluster = cls(
            host_resolution_func=config_loader.host_resolution_func, temp_dir=temp_dir
        )

        cluster.head_node.add_srun_options(*config_loader.master_srun_options)
        cluster.head_node.add_start_options(*config_loader.master_start_options)

        for node in cluster.worker_nodes.values():
            node.add_srun_options(*config_loader.worker_srun_options)
            node.add_start_options(*config_loader.worker_start_options)

        return cluster

    def __init__(self, port=6543, host_resolution_func=None, temp_dir=None):
        try:
            nodelist = os.environ["SLURM_JOB_NODELIST"]
        except KeyError:
            msg = "Cannot start SlurmRayCluster outside " "sbatch or salloc"
            e = Exception(msg)

        self.temp_dir = temp_dir

        self._nodelist = nodelist
        passthrough = lambda x: x
        self.host_resolution_func = host_resolution_func or passthrough
        self.port = port
        self.password = str(uuid.uuid4())

        self.node_names = self.node_names()
        self.ips = [self.host_resolution_func(n) for n in self.node_names]
        print(self.ips)

        self.head_node_name = self.node_names[0]
        self.worker_node_names = self.node_names[1:]

        self.head_ip = self.ips[0]
        self.worker_ips = self.ips[1:]

        assert len(self.worker_node_names) == len(self.worker_ips)

        self.head_node = SlurmRayNode(
            self.head_node_name,
            self.head_ip,
            self.port,
            self.password,
            temp_dir=self.temp_dir,
            is_head=True,
            ip=self.head_ip,
        )

        self.worker_nodes = {}
        for node_name, ip in zip(self.worker_node_names, self.worker_ips):
            self.worker_nodes[node_name] = SlurmRayNode(
                node_name,
                self.head_ip,
                self.port,
                self.password,
                temp_dir=self.temp_dir,
                is_head=False,
                ip=ip,
            )

        self.all_nodes = [self.head_node] + list(self.worker_nodes.values())

    def stop(self):
        for node in self.all_nodes:
            node.stop()

    def start(self):
        print("STARTING CLUSTER")
        self.head_node.start()
        time.sleep(2)
        print("Ok, started head node")
        print(self.worker_nodes.keys())
        for node_name, node in self.worker_nodes.items():

            print(f"Starting {node_name}...")
            node.start()
        time.sleep(2)

    def head(self):
        return f"{self.head_ip}:{self.port}"

    def node_names(self):
        _nodes = scontrol("show", "hostnames", self._nodelist)
        nodes = [n for n in _nodes.split("\n") if n]
        return nodes

    def create_yaml_info(self):
        return {"address": self.head_ip, "port": self.port, "password": self.password}

    def ray_init(self, **kwargs):
        kwargs["address"] = self.head()
        kwargs["_node_ip_address"] = self.head_ip
        ray.init(**kwargs)


class SlurmRayNode:
    def __init__(
        self,
        node_name,
        head_address,
        head_port,
        password,
        is_head=False,
        ip=None,
        temp_dir=None,
    ):
        self.head_address = head_address
        self.head_port = head_port
        self.name = node_name
        self.ip = ip or self.name
        self.is_head = is_head
        self.password = password
        self.temp_dir = temp_dir
        self._additional_start_options = []
        self._additional_srun_options = []

    def add_srun_options(self, *options):
        self._additional_srun_options += options

    def add_start_options(self, *options):
        self._additional_start_options += options

    def start(self):
        srun_options = ["--nodes=1", "--ntasks=1", "-w", self.name]
        ray_start_args = [
            "ray",
            "start",
            "--block",
            "--node-ip-address",
            self.ip,
        ]
        #'--redis-password', self.password]
        if self.temp_dir:
            ray_start_args += ["--temp-dir", self.temp_dir]

        if self.is_head:
            ray_start_args += ["--port", self.head_port, "--head"]
        else:
            address = f"{self.head_address}:{self.head_port}"
            ray_start_args.append(f"--address={address}")

        args = (
            srun_options
            + self._additional_srun_options
            + ray_start_args
            + self._additional_start_options
        )

        print(" ".join([str(a) for a in args]))
        srun(*args, _bg=True)

    def stop(self):
        srun_args = [
            "--nodes=1",
            "--ntasks=1",
            "--mem=5G",
            "-w",
            self.name,
            "ray",
            "stop",
        ]

        srun(*srun_args, _bg=False)


class TestSlurmRayCluster(unittest.TestCase):
    def test_node_names(self):
        f = saga_ip_address
        cluster = SlurmRayCluster(host_resolution_func=f)
        nodes = cluster.node_names()
        expected_num_nodes = int(os.environ["SLURM_JOB_NUM_NODES"])
        self.assertEqual(expected_num_nodes, len(nodes))
        print(cluster.head())
        ray.init(address=cluster.head())

        one_id = ray.put(666)

        print(ray.get(one_id))
