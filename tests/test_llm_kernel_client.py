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

import os
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from toksearch.llm.kernel.checkpoint import find_checkpoint
from toksearch.llm.kernel.client import KernelClient, KernelError

REPO = Path(__file__).resolve().parents[1]


class Fragile:
    """Unpickling fails while ``marker`` exists: raises, or exits the process."""

    def __init__(self, marker, mode, value):
        self.marker, self.mode, self.value = str(marker), mode, value

    def __setstate__(self, state):
        if os.path.exists(state["marker"]):
            if state["mode"] == "exit":
                os._exit(9)
            raise RuntimeError("environment is broken")
        self.__dict__.update(state)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def _kill_quietly(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


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
        self.run_code("warm = 1")   # the worker is up: Stop reaches the cell
        timer = threading.Timer(1.0, self.client.interrupt)
        t0 = time.monotonic()
        timer.start()
        r = self.run_code("import time; time.sleep(60)", timeout=30.0)
        elapsed = time.monotonic() - t0
        timer.join()
        self.assertTrue(r.interrupted)
        self.assertFalse(r.restarted)
        self.assertLess(elapsed, 10.0)

    def test_stop_kills_a_cell_that_ignores_sigint(self):
        old = int(self.run_code("import os; print(os.getpid())").stdout)
        stopped, died = [], []

        def stop():
            stopped.append(time.monotonic())
            self.client.interrupt()

        def watch():
            while _pid_alive(old) and time.monotonic() < t0 + 120:
                time.sleep(0.02)
            died.append(time.monotonic())

        t0 = time.monotonic()
        timer = threading.Timer(1.0, stop)
        watcher = threading.Thread(target=watch)
        timer.start()
        watcher.start()
        r = self.run_code(
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "time.sleep(60)", timeout=600.0)
        returned = time.monotonic()
        timer.join()
        watcher.join()
        self.assertTrue(r.interrupted)
        self.assertTrue(r.restarted)
        self.assertIn("did not stop when interrupted", r.stderr)
        self.assertIn("worker was killed", r.stderr)
        # The worker is killed within grace + 3 s of Stop; run() then also
        # starts its replacement (a few seconds of imports) before returning.
        self.assertLess(died[0] - stopped[0], self.grace + 3.0)
        self.assertLess(returned - stopped[0], self.grace + 3.0 + 30.0)

    def test_stop_while_idle_is_discarded(self):
        self.run_code("pass")
        self.client.interrupt()
        r = self.run_code("print('ran')")
        self.assertEqual(r.stdout, "ran\n")
        self.assertFalse(r.interrupted)

    def test_stop_during_startup_does_not_run_the_cell(self):
        with TemporaryDirectory() as tmp:
            c = KernelClient(packages=[])
            self.addCleanup(c.close)
            marker = Path(tmp) / "ran"
            timer = threading.Timer(0.1, c.interrupt)
            timer.start()
            r = c.run(f"open({str(marker)!r}, 'w').close()", cell=1,
                      fig_dir=str(Path(tmp) / "figures"))
            timer.join()
            self.assertEqual(r.stderr, "Interrupted before the code ran.\n")
            self.assertTrue(r.interrupted)
            self.assertTrue(r.is_error)
            self.assertFalse(marker.exists())
            self.assertEqual(c.run("print(1)", cell=2,
                                   fig_dir=str(Path(tmp) / "f")).stdout, "1\n")

    def test_slow_figure_save_is_interrupted_and_namespace_survives(self):
        self.run_code("kept_fig = 1")
        try:
            t0 = time.monotonic()
            r = self.run_code(
                "import matplotlib.figure as _mf, time as _t\n"
                "_orig_savefig = _mf.Figure.savefig\n"
                "_mf.Figure.savefig = lambda self, *a, **k: _t.sleep(30)\n"
                "plt.figure(); plt.plot([1, 2])\n"
                "plt.figure(); plt.plot([2, 1])\n", timeout=1.0)
            elapsed = time.monotonic() - t0
        finally:
            self.run_code("_mf.Figure.savefig = _orig_savefig")
        self.assertTrue(r.interrupted)
        self.assertFalse(r.restarted)
        self.assertLess(elapsed, 1.0 + self.grace)
        self.assertEqual([f.get("error") for f in r.figures],
                         ["skipped: interrupted"] * 2)
        self.assertEqual(self.run_code("print(kept_fig)").stdout, "1\n")

    def test_timeout_kills_processes_the_cell_started(self):
        with TemporaryDirectory() as tmp:
            pidfile = Path(tmp) / "pid"
            r = self.run_code(
                "import signal, subprocess, time\n"
                "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                "p = subprocess.Popen(['sleep', '300'])\n"
                f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
                "time.sleep(60)", timeout=1.0)
            pid = int(pidfile.read_text())
            self.addCleanup(_kill_quietly, pid)
            self.assertTrue(r.restarted)
            self.assertTrue(_wait_gone(pid, 5.0),
                            f"sleep (pid {pid}) outlived its worker")

    def test_worker_dies_with_its_parent(self):
        # The worker is busy in a cell (not waiting on the pipe, where EOF
        # would end it anyway) when its parent is SIGKILLed.
        with TemporaryDirectory() as tmp:
            helper = subprocess.Popen(
                [sys.executable, "-c",
                 "import sys, time\n"
                 "from toksearch.llm.kernel.client import KernelClient\n"
                 "c = KernelClient(packages=[])\n"
                 "r = c.run('import os; print(os.getpid())', cell=1,\n"
                 "          fig_dir=sys.argv[1])\n"
                 "print(r.stdout.strip(), flush=True)\n"
                 "c.run('import time; time.sleep(300)', cell=2,\n"
                 "      fig_dir=sys.argv[1], timeout=600)\n",
                 str(Path(tmp) / "figures")],
                cwd=REPO, stdout=subprocess.PIPE, text=True)
            self.addCleanup(helper.wait)
            self.addCleanup(_kill_quietly, helper.pid)
            line = helper.stdout.readline()
            helper.stdout.close()
            worker = int(line)
            self.addCleanup(_kill_quietly, worker)
            self.assertTrue(_pid_alive(worker))
            os.kill(helper.pid, signal.SIGKILL)
            helper.wait()
            self.assertTrue(_wait_gone(worker, 5.0),
                            "the worker outlived its parent")

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


class TestCheckpointCalls(_SharedKernel):
    def test_checkpoint_while_a_cell_runs_raises_busy(self):
        self.run_code("pass")
        t = threading.Thread(
            target=self.run_code, args=("import time; time.sleep(5)",))
        t.start()
        self.addCleanup(t.join)
        deadline = time.monotonic() + 5
        while not self.client._lock.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        t0 = time.monotonic()
        with self.assertRaises(KernelError) as cm:
            self.client.checkpoint(wait=0.5)
        self.assertLess(time.monotonic() - t0, 1.0)
        self.assertIn("busy", str(cm.exception))

    def test_checkpoint_without_a_directory_is_refused(self):
        with TemporaryDirectory() as tmp:
            c = KernelClient(packages=[])
            self.addCleanup(c.close)
            c.run("x = 1", cell=1, fig_dir=str(Path(tmp) / "figures"))
            with self.assertRaises(ValueError):
                c.checkpoint()
            self.assertFalse(Path("None").exists())

    def test_restart_can_change_the_checkpoint_dir(self):
        self.run_code("moved = 1")
        old = self.client.checkpoint_dir
        new = Path(self.tmp.name) / "elsewhere"
        self.client.restart(checkpoint_dir=new)
        try:
            self.assertEqual(self.client.checkpoint_dir, new)
            self.run_code("moved2 = 2")
            self.client.checkpoint()
            self.assertIsNotNone(find_checkpoint(new))
        finally:
            self.client.restart(checkpoint_dir=old)


class TestOutput(_SharedKernel):
    def test_huge_output_is_bounded(self):
        r = self.run_code("print('x' * 10_000_000)")
        self.assertLess(len(r.stdout), 5_100_000)
        self.assertIn("characters dropped", r.stdout)
        self.assertTrue(r.stdout.startswith("x" * 1000))
        self.assertTrue(r.stdout.endswith("x" * 1000 + "\n"))


class TestRestoreFailures(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.ckpt = self.root / "namespace"
        self.marker = self.root / "broken"
        self.figs = str(self.root / "figures")

    def _client(self):
        c = KernelClient(packages=[], checkpoint_dir=self.ckpt)
        self.addCleanup(c.close)
        return c

    def _save_fragile(self, mode):
        c = self._client()
        c.run("frag = __import__('tests.test_llm_kernel_client', "
              f"fromlist=['Fragile']).Fragile({str(self.marker)!r}, "
              f"{mode!r}, 5)\nok = 1", cell=1, fig_dir=self.figs)
        c.checkpoint()
        c.close()

    def test_unloadable_variable_is_carried_until_it_loads(self):
        self._save_fragile("raise")
        self.marker.touch()
        c2 = self._client()
        r = c2.run("print('frag' in globals(), ok)", cell=2, fig_dir=self.figs)
        self.assertEqual(r.stdout, "False 1\n")
        self.assertEqual([(e["name"], e["reason"])
                          for e in r.restore["not_restored"]],
                         [("frag", "unloadable")])
        index = c2.checkpoint()
        frag = next(e for e in index["variables"] if e["name"] == "frag")
        self.assertEqual(frag["status"], "saved")
        self.assertTrue(frag["carried"])
        index = c2.checkpoint()   # carried again from its new location
        frag = next(e for e in index["variables"] if e["name"] == "frag")
        self.assertEqual(frag["status"], "saved")
        c2.close()
        self.marker.unlink()      # the environment is fixed
        c3 = self._client()
        r = c3.run("print(frag.value)", cell=3, fig_dir=self.figs)
        self.assertEqual(r.stdout, "5\n")
        self.assertIn("frag", r.restore["restored"])

    def test_binding_a_carried_name_stops_carrying_it(self):
        self._save_fragile("raise")
        self.marker.touch()
        c2 = self._client()
        c2.run("frag = 'new'", cell=2, fig_dir=self.figs)
        index = c2.checkpoint()
        frag = next(e for e in index["variables"] if e["name"] == "frag")
        self.assertNotIn("carried", frag)
        self.assertEqual(frag["type"], "str")

    def test_restore_that_crashes_the_worker_is_quarantined(self):
        self._save_fragile("exit")
        self.marker.touch()
        c2 = self._client()
        r = c2.run("print('frag' in globals(), 'ok' in globals())", cell=2,
                   fig_dir=self.figs)
        self.assertFalse(r.is_error, r.stderr)
        self.assertEqual(r.stdout, "False False\n")
        rep = r.restore
        self.assertTrue(rep["found"])
        self.assertEqual(rep["restored"], [])
        self.assertIn("exit code 9", rep["error"])
        q = Path(rep["quarantined"])
        self.assertTrue(q.name.startswith("namespace.failed-"))
        self.assertTrue((q / "index.json").exists())
        self.assertTrue(any(q.glob("v*.pkl")))
        self.assertIsNone(find_checkpoint(self.ckpt))
        self.assertTrue(c2.running)


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
