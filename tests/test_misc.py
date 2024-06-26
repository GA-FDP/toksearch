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
import os
from toksearch.utilities.utilities import set_env


class TestSetEnv(unittest.TestCase):
    def test_set_env(self):
        arbitrary_string = "aardvark"
        var = "some_variable"
        with set_env(var, arbitrary_string):
            self.assertEquals(os.environ[var], arbitrary_string)
