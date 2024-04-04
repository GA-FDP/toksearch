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

import yaml
import importlib
import os


def get_host_resolution_func(config):
    module_name = config.get("host_resolution", {}).get("module", "")
    if not module_name:
        return None

    module = importlib.import_module(module_name)

    func_name = config.get("host_resolution", {}).get("func", "")

    if not func_name:
        return None

    return getattr(module, func_name)


def load_config(config_file_path):
    with open(config_file_path, "r") as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


class ConfigLoader:
    def __init__(self, disttype, filename):

        filename = filename or os.getenv("TOKSEARCH_SLURM_CONFIG")

        self.config = load_config(filename)
        config = self.config

        self.host_resolution_func = get_host_resolution_func(config)

        self.master_srun_options = (
            config.get(disttype, {}).get("master", {}).get("srun", [])
        )

        self.master_start_options = (
            config.get(disttype, {}).get("master", {}).get("start", [])
        )

        self.worker_srun_options = (
            config.get(disttype, {}).get("worker", {}).get("srun", [])
        )

        self.worker_start_options = (
            config.get(disttype, {}).get("worker", {}).get("start", [])
        )
