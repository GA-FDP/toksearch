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

import unittest
import socket
import toksearch.slurm
from toksearch.slurm import slurm_default_config

import os


class TestConfigFileHandling(unittest.TestCase):
    def test_saga_yml_exists(self):
        f = toksearch.slurm.__file__

        yml_file = os.path.join(os.path.dirname(f), "saga", "saga_slurm.yaml")
        self.assertTrue(os.path.exists(yml_file))

    def test_slurm_default_config(self):
        host = socket.gethostname()

        default_config = slurm_default_config()

        if host.startswith("saga"):
            self.assertTrue(os.path.exists(default_config))
        else:
            self.assertIsNone(default_config)
