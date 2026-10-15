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


class TestTimeoutsAndCrashes(_SharedKernel):
    grace = 2.0

    def test_interrupt_keeps_namespace(self):
        self.run_code("kept = 1")
        r = self.run_code("import time; time.sleep(60)", timeout=1.0)
        self.assertTrue(r.interrupted)
        self.assertFalse(r.restarted)
        self.assertEqual(self.run_code("print(kept)").stdout, "1\n")

    def test_interrupt_from_another_thread(self):
        import threading
        timer = threading.Timer(1.0, self.client.interrupt)
        timer.start()
        r = self.run_code("import time; time.sleep(60)", timeout=30.0)
        timer.join()
        self.assertTrue(r.interrupted)
        self.assertFalse(r.restarted)

    def test_unstoppable_code_is_killed(self):
        r = self.run_code(
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "time.sleep(60)", timeout=1.0)
        self.assertTrue(r.interrupted)
        self.assertTrue(r.restarted)
        self.assertIn("worker was killed", r.stderr)

    def test_segfault_restarts(self):
        r = self.run_code("import ctypes; ctypes.string_at(0)")
        self.assertTrue(r.restarted)
        self.assertIn("exit code -11", r.stderr)
        self.assertEqual(self.run_code("print(2)").stdout, "2\n")


class TestCheckpointRestore(_SharedKernel):
    def test_crash_restores_last_checkpoint(self):
        self.run_code("saved_a = 1")
        index = self.client.checkpoint()
        self.assertIn("saved_a", [e["name"] for e in index["variables"]])
        self.run_code("after_b = 2")
        r = self.run_code("import os; os._exit(7)")
        self.assertTrue(r.restarted)
        self.assertIn("exit code 7", r.stderr)
        self.assertTrue(r.restore["found"])
        self.assertIn("saved_a", r.restore["restored"])
        out = self.run_code("print(saved_a, 'after_b' in globals())").stdout
        self.assertEqual(out, "1 False\n")

    def test_new_client_loads_checkpoint_on_first_run(self):
        self.run_code("carried = 'yes'")
        self.client.checkpoint()
        c2 = KernelClient(packages=[], checkpoint_dir=self.ckpt)
        try:
            r = c2.run("print(carried)", cell=1,
                       fig_dir=str(Path(self.tmp.name) / "figures"))
            self.assertEqual(r.stdout, "yes\n")
            self.assertIn("carried", r.restore["restored"])
            r2 = c2.run("pass", cell=2,
                        fig_dir=str(Path(self.tmp.name) / "figures"))
            self.assertIsNone(r2.restore)
        finally:
            c2.close()

    def test_restart_does_not_restore_after_checkpoint_dir_changes(self):
        self.run_code("gone = 1")
        self.client.checkpoint()
        old = self.client.checkpoint_dir
        self.client.checkpoint_dir = Path(self.tmp.name) / "fresh"
        try:
            self.client.restart()
            r = self.run_code("print('gone' in globals())")
            self.assertEqual(r.stdout, "False\n")
            self.assertIsNone(r.restore)
        finally:
            self.client.checkpoint_dir = old


class TestFiguresThroughWorker(_SharedKernel):
    def test_matplotlib_figure_is_saved_and_closed(self):
        r = self.run_code("plt.plot([1, 2, 3])")
        self.assertEqual(len(r.figures), 1)
        self.assertTrue(r.figures[0]["png_path"].endswith(f"c{self.cell}-1.png"))
        self.assertTrue(Path(r.figures[0]["png_path"]).exists())
        self.assertEqual(self.run_code("print(len(plt.get_fignums()))").stdout,
                         "0\n")

    def test_plotly_show_is_captured(self):
        r = self.run_code("px.line(y=[1, 2, 3]).show()")
        self.assertEqual(len(r.figures), 1)
        self.assertEqual(r.figures[0]["kind"], "plotly")
        self.assertTrue(Path(r.figures[0]["html_path"]).exists())

    def test_no_figures_when_none_made(self):
        self.assertEqual(self.run_code("z = 3").figures, [])


if __name__ == "__main__":
    unittest.main()
