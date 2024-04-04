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
import os
import argparse
import yaml

from sh import sbatch, salloc
import subprocess

from . import slurm_default_config


def check_greater_than_two(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("{} must be >= 2" % value)
    return str(ivalue)


DEFAULT_CONFIG = os.getenv("TOKSEARCH_SLURM_CONFIG", None) or slurm_default_config()


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


class SlurmRunner:

    @classmethod
    def update_parser(cls, parser):
        parser.add_argument(
            "-N",
            "--num-nodes",
            type=int,
            default=1,
            help="Number of cluster nodes to use",
        )
        parser.add_argument(
            "-b",
            "--batch",
            action="store_true",
            help="Run in batch mode (ie use sbatch)",
        )
        parser.add_argument(
            "--dump",
            action="store_true",
            help="Print a batch script to stdout. Only applicable when -b is set",
        )
        parser.add_argument(
            "--config-file",
            default=DEFAULT_CONFIG,
            help="Config file used to set up slurm",
        )

    def __init__(self, num_nodes, config_file=DEFAULT_CONFIG):
        self.config_file = config_file
        self.config = load_config(config_file)
        self.num_nodes = num_nodes

        self.job_config = [f"-N {num_nodes}", "--exclusive"]
        self.job_config += self.config.get("job", [])

    def get_envs(self):
        envs = {}
        envs["TOKSEARCH_SLURM_CONFIG"] = self.config_file
        envs["TOKSEARCH_SUBMIT"] = "active"
        return envs

    def set_envs(self):
        envs = self.get_envs()
        for key, val in envs.items():
            os.environ[key] = val

    def run_interactive(self, script, *script_args):
        self.set_envs()
        srun_args = ["srun", "-u", "--pty", "-N", "1", "--exclusive", "--propagate"]
        srun_args += self.config.get("interactive", [])
        args = self.job_config + srun_args + [script] + list(script_args)
        comm = ["salloc"] + args
        print(" ".join(comm))
        return subprocess.run(comm)

    def run_batch(self, script, *script_args):
        self.set_envs()
        args = self.job_config + [script] + list(script_args)
        comm = ["sbatch"] + args
        return subprocess.run(comm)

    def create_batch_script(self, script, *script_args):

        directives = [f"#SBATCH {directive}" for directive in self.job_config]
        directive_string = "\n".join(directives)
        script_args_string = " ".join(script_args)

        sbatch_script = f"""#!/bin/bash
{directive_string} 
{script} {script_args_string}
        """
        return sbatch_script
