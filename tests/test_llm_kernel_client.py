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

"""Tests for the run_python worker process and KernelClient.

These start real spawned processes; one client is shared per class to
amortise the namespace import cost. packages=[] skips contributor imports.
"""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from toksearch.llm.kernel.client import KernelClient


class _SharedKernel(unittest.TestCase):
    grace = 10.0

    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.ckpt = Path(cls.tmp.name) / "namespace"
        cls.client = KernelClient(packages=[], grace=cls.grace,
                                  checkpoint_dir=cls.ckpt)
        cls.cell = 0

    @classmethod
    def tearDownClass(cls):
        cls.client.close()
        cls.tmp.cleanup()

    def run_code(self, code, timeout=60.0):
        type(self).cell += 1
        return self.client.run(code, cell=self.cell,
                               fig_dir=str(Path(self.tmp.name) / "figures"),
                               timeout=timeout)


class TestExecute(_SharedKernel):
    def test_namespace_persists_between_calls(self):
        self.run_code("x_persist = 41")
        r = self.run_code("print(x_persist + 1)")
        self.assertEqual(r.stdout, "42\n")
        self.assertFalse(r.is_error)
        self.assertFalse(r.restarted)

    def test_namespace_is_prepopulated(self):
        r = self.run_code("print(toksearch.__name__, np.__name__)")
        self.assertEqual(r.stdout, "toksearch numpy\n")

    def test_bound_names(self):
        self.run_code("keep_me = 1")
        r = self.run_code("keep_me = 2\nnew_one = 3\nimport json")
        self.assertEqual(r.bound, ["keep_me", "new_one"])

    def test_exception_is_reported_and_worker_survives(self):
        self.run_code("y_survive = 7")
        r = self.run_code("1/0")
        self.assertTrue(r.is_error)
        self.assertIn("ZeroDivisionError", r.stderr)
        self.assertEqual(self.run_code("print(y_survive)").stdout, "7\n")

    def test_exit_does_not_kill_worker(self):
        r = self.run_code("exit(3)")
        self.assertTrue(r.is_error)
        self.assertTrue(self.client.running)

    def test_starts_lazily_and_checkpoint_needs_a_worker(self):
        with TemporaryDirectory() as tmp:
            c = KernelClient(packages=[])
            try:
                self.assertFalse(c.running)
                self.assertIsNone(c.checkpoint(Path(tmp) / "namespace"))
            finally:
                c.close()


if __name__ == "__main__":
    unittest.main()
