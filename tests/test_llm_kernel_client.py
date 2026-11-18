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
import shutil
import signal
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from toksearch.llm.kernel.checkpoint import find_checkpoint
from toksearch.llm.kernel.client import KernelClient, KernelError

REPO = Path(__file__).resolve().parents[1]


# Defined inside the worker by the cell itself, so cloudpickle saves the class
# by value: the restore path is exercised however the tests were started (the
# worker cannot import this module under every runner's sys.path).
FRAGILE_SRC = """
class Fragile:
    \"\"\"Unpickling fails while ``marker`` exists: raises, or exits.\"\"\"

    def __init__(self, marker, mode, value):
        self.marker, self.mode, self.value = str(marker), mode, value

    def __setstate__(self, state):
        import os
        if os.path.exists(state["marker"]):
            if state["mode"] == "exit":
                os._exit(9)
            raise RuntimeError("environment is broken")
        self.__dict__.update(state)
"""


def _fragile_cell(marker, mode, extra=""):
    """Cell code binding ``frag`` (value 5) and nothing else of Fragile's."""
    return (FRAGILE_SRC + f"frag = Fragile({str(marker)!r}, {mode!r}, 5)\n"
            "del Fragile\n" + extra)


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

    def test_stop_while_idle_is_discarded(self):
        self.run_code("pass")
        self.client.interrupt()
        r = self.run_code("print('ran')")
        self.assertEqual(r.stdout, "ran\n")
        self.assertFalse(r.interrupted)

    def test_segfault_restarts(self):
        r = self.run_code("import ctypes; ctypes.string_at(0)")
        self.assertTrue(r.restarted)
        self.assertIn("exit code -11", r.stderr)
        self.assertEqual(self.run_code("print(2)").stdout, "2\n")


class _Dedicated(unittest.TestCase):
    """Each test gets its own client (grace=2): kills never leak across tests."""

    grace = 2.0

    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cell = 0

    def client(self, **kw):
        c = KernelClient(packages=[], grace=self.grace, **kw)
        self.addCleanup(c.close)
        return c

    def run_in(self, c, code, timeout=60.0):
        self.cell += 1
        return c.run(code, cell=self.cell, fig_dir=str(self.root / "figures"),
                     timeout=timeout)


class TestKillPath(_Dedicated):
    def test_stop_kills_a_cell_that_ignores_sigint(self):
        c = self.client()
        old = int(self.run_in(c, "import os; print(os.getpid())").stdout)
        stopped, died = [], []

        def stop():
            stopped.append(time.monotonic())
            c.interrupt()

        def watch():
            while _pid_alive(old) and time.monotonic() < t0 + 120:
                time.sleep(0.02)
            died.append(time.monotonic())

        t0 = time.monotonic()
        timer = threading.Timer(1.0, stop)
        watcher = threading.Thread(target=watch)
        timer.start()
        watcher.start()
        r = self.run_in(c, "import signal, time\n"
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

    def test_unstoppable_code_is_killed(self):
        c = self.client()
        r = self.run_in(c, "import signal, time\n"
                           "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                           "time.sleep(60)", timeout=1.0)
        self.assertTrue(r.interrupted)
        self.assertTrue(r.restarted)
        self.assertIn("worker was killed", r.stderr)

    def test_timeout_kills_processes_the_cell_started(self):
        c = self.client()
        pidfile = self.root / "pid"
        r = self.run_in(c, "import signal, subprocess, time\n"
                           "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                           "p = subprocess.Popen(['sleep', '300'])\n"
                           f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
                           "time.sleep(60)", timeout=1.0)
        pid = int(pidfile.read_text())
        self.addCleanup(_kill_quietly, pid)
        self.assertTrue(r.restarted)
        self.assertTrue(_wait_gone(pid, 5.0),
                        f"sleep (pid {pid}) outlived its worker")

    def test_slow_figure_save_is_interrupted_and_namespace_survives(self):
        # A dedicated client: the patched savefig dies with it, so no
        # clean-up cell can fail or hide the assertions below.
        c = self.client()
        self.run_in(c, "kept_fig = 1")
        t0 = time.monotonic()
        r = self.run_in(c, "import matplotlib.figure as _mf, time as _t\n"
                           "_mf.Figure.savefig = "
                           "lambda self, *a, **k: _t.sleep(30)\n"
                           "plt.figure(); plt.plot([1, 2])\n"
                           "plt.figure(); plt.plot([2, 1])\n", timeout=1.0)
        elapsed = time.monotonic() - t0
        self.assertTrue(r.interrupted)
        self.assertFalse(r.restarted)
        self.assertLess(elapsed, 1.0 + self.grace)
        self.assertEqual([f.get("error") for f in r.figures],
                         ["skipped: interrupted"] * 2)
        self.assertEqual(self.run_in(c, "print(kept_fig)").stdout, "1\n")

    def test_stop_during_startup_does_not_run_the_cell(self):
        c = self.client()
        real_start = c._start

        def start_and_stop():   # Stop pressed while the worker starts
            c.interrupt()
            real_start()

        c._start = start_and_stop
        marker = self.root / "ran"
        r = self.run_in(c, f"open({str(marker)!r}, 'w').close()")
        self.assertEqual(r.stderr, "Interrupted before the code ran.\n")
        self.assertTrue(r.interrupted)
        self.assertTrue(r.is_error)
        self.assertFalse(marker.exists())
        c._start = real_start
        self.assertEqual(self.run_in(c, "print(1)").stdout, "1\n")

    def test_closing_client_kills_without_a_cold_start(self):
        c = self.client()
        self.run_in(c, "pass")
        spawns = []
        real_spawn = c._spawn
        c._spawn = lambda: (spawns.append(1), real_spawn())
        c.begin_close()
        timer = threading.Timer(0.5, c.interrupt)
        timer.start()
        r = self.run_in(c, "import signal, time\n"
                           "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
                           "time.sleep(60)", timeout=600.0)
        timer.join()
        self.assertTrue(r.interrupted)
        self.assertFalse(r.restarted)
        self.assertIn("conversation is closing", r.stderr)
        self.assertEqual(spawns, [])
        self.assertFalse(c.running)

    def test_broken_pipe_on_send_is_a_crash_not_an_exception(self):
        c = self.client()
        self.run_in(c, "pass")

        def broken(*a, **k):
            raise BrokenPipeError(32, "Broken pipe")

        c._conn.send = broken
        r = self.run_in(c, "print(1)")
        self.assertTrue(r.restarted)
        self.assertIn("crashed", r.stderr)
        self.assertEqual(self.run_in(c, "print(2)").stdout, "2\n")
        c._conn.send = broken
        with self.assertRaises(KernelError):
            c.checkpoint(self.root / "namespace")


class TestProcessLifetime(unittest.TestCase):
    """Workers and what their cells start never outlive the client."""

    HELPER = ("import sys, time\n"
              "from toksearch.llm.kernel.client import KernelClient\n"
              "c = KernelClient(packages=[])\n"
              "r = c.run('import os, subprocess\\n"
              "p = subprocess.Popen([\"sleep\", \"300\"])\\n"
              "print(os.getpid(), p.pid)', cell=1, fig_dir=sys.argv[1])\n"
              "print(r.stdout.strip(), flush=True)\n")

    def _helper(self, tail):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        helper = subprocess.Popen(
            [sys.executable, "-c", self.HELPER + tail,
             str(Path(tmp.name) / "figures")],
            cwd=REPO, stdout=subprocess.PIPE, text=True)
        self.addCleanup(helper.wait)
        self.addCleanup(_kill_quietly, helper.pid)
        worker, sleeper = map(int, helper.stdout.readline().split())
        helper.stdout.close()
        self.addCleanup(_kill_quietly, worker)
        self.addCleanup(_kill_quietly, sleeper)
        self.assertTrue(_pid_alive(sleeper))
        return helper, worker, sleeper

    def test_busy_worker_dies_with_its_parent(self):
        helper, worker, sleeper = self._helper(
            "c.run('import time; time.sleep(300)', cell=2,\n"
            "      fig_dir=sys.argv[1], timeout=600)\n")
        os.kill(helper.pid, signal.SIGKILL)
        helper.wait()
        self.assertTrue(_wait_gone(worker, 5.0), "the worker outlived its parent")
        self.assertTrue(_wait_gone(sleeper, 5.0), "sleep outlived the parent")

    def test_idle_worker_and_its_children_die_with_the_parent(self):
        helper, worker, sleeper = self._helper("time.sleep(300)\n")
        os.kill(helper.pid, signal.SIGKILL)
        helper.wait()
        self.assertTrue(_wait_gone(sleeper, 5.0), "sleep outlived the parent")
        self.assertTrue(_wait_gone(worker, 5.0), "the worker outlived its parent")

    def test_interpreter_exit_without_close_takes_children(self):
        helper, worker, sleeper = self._helper("")   # just exits
        self.assertEqual(helper.wait(30), 0)
        self.assertTrue(_wait_gone(sleeper, 5.0), "sleep outlived the parent")
        self.assertTrue(_wait_gone(worker, 5.0), "the worker outlived its parent")

    def test_crashed_worker_leaves_no_children(self):
        with TemporaryDirectory() as tmp:
            c = KernelClient(packages=[])
            self.addCleanup(c.close)
            pidfile = Path(tmp) / "pid"
            r = c.run("import os, subprocess\n"
                      "p = subprocess.Popen(['sleep', '300'])\n"
                      f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
                      "os._exit(1)", cell=1, fig_dir=tmp)
            pid = int(pidfile.read_text())
            self.addCleanup(_kill_quietly, pid)
            self.assertTrue(r.restarted)
            self.assertTrue(_wait_gone(pid, 5.0),
                            "sleep outlived its crashed worker")

    def test_exit_while_a_cell_is_stuck_in_c_does_not_hang(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        started = Path(tmp.name) / "started"
        cell = Path(tmp.name) / "cell.py"
        cell.write_text(f"open({str(started)!r}, 'w').close()\n"
                        "sum(range(10**12))\n")
        helper, worker, sleeper = self._helper(
            "import os, threading\n"
            f"code = open({str(cell)!r}).read()\n"
            "threading.Thread(target=c.run, daemon=True, args=(code,),\n"
            "                 kwargs=dict(cell=2, fig_dir=sys.argv[1],\n"
            "                             timeout=3600)).start()\n"
            f"while not os.path.exists({str(started)!r}):\n"
            "    time.sleep(0.05)\n")
        try:
            helper.wait(15)
        except subprocess.TimeoutExpired:
            self.fail("the parent hung on exit while a cell ran C code")
        self.assertTrue(_wait_gone(worker, 5.0), "the worker outlived its parent")
        self.assertTrue(_wait_gone(sleeper, 5.0), "sleep outlived the parent")

    def test_exit_hook_is_removed_by_close_and_by_collection(self):
        import atexit
        import gc
        import weakref
        import toksearch.llm.kernel.client as client_mod
        registered, unregistered = [], []
        real_register, real_unregister = atexit.register, atexit.unregister

        def register(f, *a, **k):
            registered.append(f)
            return real_register(f, *a, **k)

        def unregister(f):
            unregistered.append(f)
            return real_unregister(f)

        with TemporaryDirectory() as tmp, \
                mock.patch.object(client_mod.atexit, "register",
                                  side_effect=register), \
                mock.patch.object(client_mod.atexit, "unregister",
                                  side_effect=unregister):
            c = KernelClient(packages=[])
            c.run("pass", cell=1, fig_dir=tmp)
            self.assertEqual(len(registered), 1)
            c.close()
            self.assertEqual(unregistered, registered)
            c2 = KernelClient(packages=[])
            pid = int(c2.run("import os; print(os.getpid())", cell=1,
                             fig_dir=tmp).stdout)
            self.addCleanup(_kill_quietly, pid)
            self.assertEqual(len(registered), 2)
            ref = weakref.ref(c2)
            del c2
            gc.collect()
            self.assertIsNone(ref(), "the exit hook kept the client alive")
            self.assertEqual(unregistered, registered)
            # The worker ends on pipe EOF; reap it (it is our child).
            import multiprocessing
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                multiprocessing.active_children()
                if not _pid_alive(pid):
                    break
                time.sleep(0.05)
            self.assertFalse(_pid_alive(pid))

    def test_close_kills_processes_a_cell_started(self):
        with TemporaryDirectory() as tmp:
            for i in range(3):   # the race was on close's pipe EOF
                c = KernelClient(packages=[])
                r = c.run("import subprocess\n"
                          "p = subprocess.Popen(['sleep', '300'])\n"
                          "print(p.pid)", cell=1, fig_dir=tmp)
                pid = int(r.stdout)
                self.addCleanup(_kill_quietly, pid)
                c.close()
                self.assertTrue(_wait_gone(pid, 5.0),
                                f"sleep outlived close() (round {i})")


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
        started = Path(self.tmp.name) / "busy-started"
        t = threading.Thread(target=self.run_code, args=(
            f"open({str(started)!r}, 'w').close()\n"
            "import time; time.sleep(5)",))
        t.start()
        self.addCleanup(t.join)
        deadline = time.monotonic() + 60   # includes a possible cold start
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(started.exists(), "the cell never started")
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
        r = c.run(_fragile_cell(self.marker, mode, "ok = 1\n"), cell=1,
                  fig_dir=self.figs)
        self.assertFalse(r.is_error, r.stderr)
        self.assertEqual(r.bound, ["frag", "ok"])
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
        self.assertEqual(sorted(p.name for p in q.iterdir()), ["current"])
        self.assertTrue((q / "current" / "index.json").exists())
        self.assertTrue(any((q / "current").glob("v*.pkl")))
        self.assertIsNone(find_checkpoint(self.ckpt))
        self.assertTrue(c2.running)

    def test_quarantine_takes_every_copy_so_none_is_restored_later(self):
        c1 = self._client()
        c1.run("ok = 1", cell=1, fig_dir=self.figs)
        c1.checkpoint()                       # older, loadable checkpoint
        older = self.root / "older"
        shutil.copytree(self.ckpt, older)
        r = c1.run(_fragile_cell(self.marker, "exit"), cell=2,
                   fig_dir=self.figs)
        self.assertFalse(r.is_error, r.stderr)
        c1.checkpoint()                       # current: crashes on load
        c1.close()
        shutil.copytree(older, self.ckpt.with_name("namespace.old"))
        self.marker.touch()

        c2 = self._client()
        r1 = c2.run("pass", cell=3, fig_dir=self.figs)
        self.assertIn("exit code 9", r1.restore["error"])
        q = Path(r1.restore["quarantined"])
        self.assertEqual(sorted(p.name for p in q.iterdir()),
                         ["current", "old"])
        self.assertTrue((q / "current" / "index.json").exists())
        self.assertTrue((q / "old" / "index.json").exists())
        self.assertIsNone(find_checkpoint(self.ckpt))

        r2 = c2.run("import os; os._exit(3)", cell=4, fig_dir=self.figs)
        self.assertTrue(r2.restarted)
        self.assertIsNone(r2.restore)
        r3 = c2.run("print('ok' in globals(), 'frag' in globals())", cell=5,
                    fig_dir=self.figs)
        self.assertEqual(r3.stdout, "False False\n")
        self.assertIsNone(r3.restore)


class TestBlockedCheckpoint(unittest.TestCase):
    def test_incomplete_quarantine_blocks_checkpoints_into_that_dir(self):
        import toksearch.llm.kernel.client as client_mod
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ckpt, marker = root / "namespace", root / "broken"
            figs = str(root / "figures")
            c1 = KernelClient(packages=[], checkpoint_dir=ckpt)
            self.addCleanup(c1.close)
            r = c1.run(_fragile_cell(marker, "exit"), cell=1, fig_dir=figs)
            self.assertFalse(r.is_error, r.stderr)
            c1.checkpoint()
            c1.close()
            old = ckpt.with_name("namespace.old")
            shutil.copytree(ckpt, old)
            before = {p.name: p.read_bytes() for p in old.iterdir()}
            marker.touch()

            real_rename = os.rename
            moved = []

            def rename_once(src, dst, *a, **k):
                if ".failed-" in str(dst):
                    if moved:
                        raise OSError(13, "Permission denied")
                    moved.append(src)
                return real_rename(src, dst, *a, **k)

            c2 = KernelClient(packages=[], checkpoint_dir=ckpt)
            self.addCleanup(c2.close)
            with mock.patch.object(client_mod.os, "rename",
                                   side_effect=rename_once):
                r = c2.run("x = 1", cell=2, fig_dir=figs)
            self.assertTrue(r.restore["checkpoint_blocked"])
            self.assertIn("could not move it aside", r.restore["error"])
            with self.assertRaises(KernelError) as cm:
                c2.checkpoint()
            self.assertIn("checkpoint blocked", str(cm.exception))
            (root / "sub").mkdir()
            with self.assertRaises(KernelError):   # same dir, other spelling
                c2.checkpoint(root / "sub" / ".." / "namespace")
            self.assertEqual({p.name: p.read_bytes() for p in old.iterdir()},
                             before)
            # a crash does not restore the copy left behind either
            r = c2.run("import os; os._exit(3)", cell=3, fig_dir=figs)
            self.assertTrue(r.restarted)
            self.assertIsNone(r.restore)
            # another directory clears the block
            elsewhere = root / "elsewhere"
            c2.restart(checkpoint_dir=elsewhere)
            c2.run("y = 2", cell=4, fig_dir=figs)
            self.assertEqual(
                [e["name"] for e in c2.checkpoint()["variables"]], ["y"])


class TestWorkerUnits(unittest.TestCase):
    """In-process checks of worker and client details (no spawn)."""

    def test_sigint_between_regions_interrupts_the_next_figure_save(self):
        from toksearch.llm.kernel import worker as wmod
        w = wmod.Worker.__new__(wmod.Worker)
        w._interruptible, w._in_execute, w._pending_interrupt = False, True, False
        w.captured = [("plotly", {}), ("plotly", {})]
        w._on_sigint(signal.SIGINT, None)      # after exec, before figures
        self.assertTrue(w._pending_interrupt)
        with mock.patch.object(wmod, "save_figure") as save, \
                mock.patch.object(wmod, "_matplotlib_figures",
                                  return_value=([], lambda: None)):
            figures, interrupted = w._save_figures(
                {"fig_dir": "/nonexistent", "cell": 1})
        save.assert_not_called()
        self.assertTrue(interrupted)
        self.assertEqual([f["error"] for f in figures],
                         ["skipped: interrupted"] * 2)
        self.assertFalse(w._pending_interrupt)

    def test_sigint_during_cell_setup_interrupts_before_exec(self):
        from toksearch.llm.kernel import worker as wmod
        w = wmod.Worker.__new__(wmod.Worker)
        w._interruptible = w._in_execute = w._pending_interrupt = False
        w.ns, w.baseline, w.origins, w.carried, w.captured = (
            {}, frozenset(), {}, {}, [])
        w._reinstall_sigint = lambda: None   # keep the runner's handler
        real_ids = w._ids
        calls = []

        def ids_then_sigint():
            calls.append(1)
            if len(calls) == 1:   # setup, before exec
                w._on_sigint(signal.SIGINT, None)
            return real_ids()

        w._ids = ids_then_sigint
        with mock.patch.object(wmod, "_matplotlib_figures",
                               return_value=([], lambda: None)):
            r = w.execute({"code": "ran = 1", "cell": 1,
                           "fig_dir": "/nonexistent"})
        self.assertTrue(r["interrupted"])
        self.assertNotIn("ran", w.ns)

    def test_sigint_outside_execute_is_ignored(self):
        from toksearch.llm.kernel import worker as wmod
        w = wmod.Worker.__new__(wmod.Worker)
        w._interruptible, w._in_execute, w._pending_interrupt = False, False, False
        w._on_sigint(signal.SIGINT, None)
        self.assertFalse(w._pending_interrupt)

    def test_signal_tolerates_eperm_from_a_zombie_group(self):
        import toksearch.llm.kernel.client as client_mod
        c = KernelClient(packages=[])
        c._proc = mock.Mock(pid=999999)
        with mock.patch.object(client_mod.os, "killpg",
                               side_effect=PermissionError), \
                mock.patch.object(client_mod.os, "kill",
                                  side_effect=PermissionError) as kill:
            c._signal(signal.SIGKILL)       # must not raise
        kill.assert_called_once_with(999999, signal.SIGKILL)


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
