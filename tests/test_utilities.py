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


from toksearch.utilities.utilities import capture_exception, set_env, unset_env


class TestUtilities(unittest.TestCase):
    def test_capture_exception(self):
        try:
            msg = "BLAH"
            raise Exception
        except Exception as e:
            val = capture_exception("my label", e)

        self.assertEqual(val["label"], "my label")
        self.assertTrue("Exception" in val["type"])
        self.assertTrue(val["traceback"].startswith("Traceback"))


    def test_set_env(self):

        os.environ.pop("MY_VAR", None)

        with set_env("MY_VAR", "MY_VAL"):
            self.assertEqual(os.getenv("MY_VAR", None), "MY_VAL")

        self.assertIsNone(os.getenv("MY_VAR", None))

    def test_unset_env(self):

        os.environ["MY_VAR"] = "MY_VAL"

        with unset_env("MY_VAR"):
            self.assertIsNone(os.getenv("MY_VAR", None))

        self.assertEqual(os.getenv("MY_VAR", None), "MY_VAL")
