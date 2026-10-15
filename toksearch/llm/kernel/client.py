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

"""KernelClient: runs code in the worker process with a timeout.

When the timeout passes or interrupt() is called, the worker gets SIGINT and
keeps its namespace; if it does not answer within ``grace`` seconds its whole
process group is killed and a new worker is started. A crashed worker is
restarted on the spot. Whenever a worker starts and ``checkpoint_dir`` holds
a checkpoint, the checkpoint is loaded and the report is attached to the
next ExecResult. A restore that fails or crashes the worker moves the
checkpoint and every older copy aside (``<dir>.failed-<stamp>``) instead of
deleting them, and the worker starts empty.
"""

import multiprocessing as mp
import os
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .checkpoint import PER_VAR_LIMIT, TOTAL_LIMIT, find_checkpoint
from .worker import worker_main

_POLL = 0.2       # seconds between checks while waiting for the worker
_KEEP = object()  # restart(): keep the current checkpoint_dir


class KernelError(RuntimeError):
    """The worker could not be started or failed an internal operation."""


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    is_error: bool = False
    interrupted: bool = False
    restarted: bool = False
    figures: list = field(default_factory=list)
    bound: list = field(default_factory=list)
    restore: dict | None = None


class KernelClient:
    """Runs run_python code in a spawned worker process.

    The worker is started with multiprocessing's ``spawn`` method, which
    re-imports the main module in the child: a script that creates a
    KernelClient (or a Session/Conversation using one) at import time needs
    an ``if __name__ == "__main__":`` guard.
    """

    def __init__(self, packages: list[str] | None = None, *,
                 checkpoint_dir=None, startup_timeout: float = 180.0,
                 grace: float = 10.0, op_timeout: float = 900.0):
        self.packages = packages
        self.checkpoint_dir = (Path(checkpoint_dir)
                               if checkpoint_dir is not None else None)
        self._startup_timeout = startup_timeout
        self._grace = grace
        self._op_timeout = op_timeout
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._interrupt = threading.Event()
        self._proc = None
        self._conn = None
        self._pending_restore = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # ---- process management (callers hold self._lock) ----

    def _signal(self, sig) -> None:
        """Send ``sig`` to the worker's process group (or just the worker)."""
        pid = self._proc.pid
        try:
            os.killpg(pid, sig)
        except (ProcessLookupError, PermissionError):
            # The child may not have called setsid() yet.
            try:
                os.kill(pid, sig)
            except ProcessLookupError:
                pass

    def _spawn(self) -> None:
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=worker_main,
                                 args=(child, self.packages, os.getpid()),
                                 daemon=True)
        proc.start()
        child.close()
        try:
            if not parent.poll(self._startup_timeout):
                raise KernelError("The Python worker did not start within "
                                  f"{self._startup_timeout:.0f} s.")
            parent.recv()
        except EOFError:
            proc.join(5)
            parent.close()
            raise KernelError("The Python worker exited during startup "
                              f"(exit code {proc.exitcode}).") from None
        except KernelError:
            self._proc = proc
            self._signal(signal.SIGKILL)
            self._proc = None
            proc.join(5)
            parent.close()
            raise
        self._proc, self._conn = proc, parent

    def _start(self) -> None:
        """Start a worker and load the checkpoint, if any.

        Raises KernelError only when a worker cannot be started at all.
        """
        self._spawn()
        if (self.checkpoint_dir is None
                or find_checkpoint(self.checkpoint_dir) is None):
            return
        try:
            reply = self._request({"op": "restore",
                                   "dir": str(self.checkpoint_dir)})
        except KernelError as e:
            self._close()
            quarantined, why = self._quarantine()
            self._spawn()
            error = str(e) if why is None else f"{e} ({why})"
            self._pending_restore = {
                "found": True, "restored": [], "not_restored": [],
                "origins": {}, "fdp_core_version": None, "error": error,
                "quarantined": quarantined}
            return
        self._pending_restore = reply["report"]

    def _quarantine(self) -> tuple[str | None, str | None]:
        """Move every copy of the checkpoint aside; (path, problem).

        ``<dir>.new``, ``<dir>`` and ``<dir>.old`` (whichever exist) go into
        one folder ``<dir>.failed-<stamp>/{new,current,old}``, so no older
        copy is silently restored by the next start.
        """
        d = self.checkpoint_dir
        copies = [(d.with_name(d.name + ".new"), "new"), (d, "current"),
                  (d.with_name(d.name + ".old"), "old")]
        copies = [(src, name) for src, name in copies if src.exists()]
        if not copies:
            return None, "the checkpoint had already gone"
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        base = d.with_name(f"{d.name}.failed-{stamp}")
        target, n = base, 1
        while target.exists():
            n += 1
            target = base.with_name(f"{base.name}-{n}")
        try:
            target.mkdir()
            for src, name in copies:
                os.rename(src, target / name)
        except OSError as e:
            return (str(target) if target.exists() else None,
                    f"could not move it aside: {e}")
        return str(target), None

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        if self._proc is not None:
            if self._proc.is_alive():
                self._signal(signal.SIGKILL)
            self._proc.join(5)
        self._proc = self._conn = None

    def _ensure_started(self) -> None:
        if not self.running:
            self._close()
            self._start()

    def _request(self, req: dict) -> dict:
        """Send a non-exec op and wait for its reply."""
        self._conn.send(req)
        if not self._conn.poll(self._op_timeout):
            self._close()
            raise KernelError(f"The Python worker did not finish {req['op']} "
                              f"within {self._op_timeout:.0f} s.")
        try:
            reply = self._conn.recv()
        except EOFError:
            self._proc.join(5)
            code = self._proc.exitcode
            self._close()
            raise KernelError(f"The Python worker crashed during {req['op']} "
                              f"(exit code {code}).") from None
        if "error" in reply:
            raise KernelError(f"{req['op']} failed: {reply['error']}")
        return reply

    def _wait_reply(self, stop) -> bool:
        """Wait in short slices until the worker replies (True) or stop()."""
        while not self._conn.poll(_POLL):
            if stop():
                return False
        return True

    # ---- public API ----

    def run(self, code: str, *, cell: int, fig_dir: str,
            timeout: float = 600.0) -> ExecResult:
        with self._lock:
            self._interrupt.clear()   # a Stop pressed while idle is discarded
            self._ensure_started()
            if self._interrupt.is_set():
                result = ExecResult(stderr="Interrupted before the code ran.\n",
                                    is_error=True, interrupted=True)
            else:
                result = self._execute(code, cell, fig_dir, timeout)
            result.restore, self._pending_restore = self._pending_restore, None
            return result

    def _execute(self, code, cell, fig_dir, timeout) -> ExecResult:
        self._conn.send({"op": "execute", "code": code, "cell": cell,
                         "fig_dir": fig_dir})
        deadline = time.monotonic() + timeout
        if self._wait_reply(lambda: self._interrupt.is_set()
                            or time.monotonic() >= deadline):
            return self._receive()
        stopped = self._interrupt.is_set()
        self._signal(signal.SIGINT)
        grace_end = time.monotonic() + self._grace
        if self._wait_reply(lambda: time.monotonic() >= grace_end):
            return self._receive()
        self._close()
        self._start()
        why = ("did not stop when interrupted" if stopped
               else f"did not stop within {timeout:.0f} s")
        return ExecResult(
            stderr=(f"The code {why}, nor {self._grace:.0f} s after an "
                    "interrupt, so the worker was killed.\n"),
            is_error=True, interrupted=True, restarted=True)

    def _receive(self) -> ExecResult:
        try:
            reply = self._conn.recv()
        except EOFError:
            self._proc.join(5)
            code = self._proc.exitcode
            self._close()
            self._start()
            return ExecResult(
                stderr=f"The Python worker crashed (exit code {code}).\n",
                is_error=True, restarted=True)
        if "error" in reply:  # the worker failed outside the user's code
            return ExecResult(stderr=reply["error"] + "\n", is_error=True)
        return ExecResult(**reply)

    def interrupt(self) -> None:
        """Stop the running cell. Safe to call from another thread.

        Lock-free: it only sets an event that run() polls. A Stop that
        arrives while no cell runs is discarded by the next run().
        """
        self._interrupt.set()

    def checkpoint(self, out_dir=None, *, wait: float | None = None,
                   per_var_limit: int = PER_VAR_LIMIT,
                   total_limit: int = TOTAL_LIMIT) -> dict | None:
        """Checkpoint the namespace; None when no worker is running.

        ``wait`` bounds how long to wait for a running cell (None: forever);
        if the cell is still running then, KernelError("busy: ...") is raised.
        """
        target = out_dir if out_dir is not None else self.checkpoint_dir
        if target is None:
            raise ValueError("checkpoint() needs out_dir when the client has "
                             "no checkpoint_dir")
        if not self._lock.acquire(timeout=-1 if wait is None else wait):
            raise KernelError("busy: a cell is running")
        try:
            if not self.running:
                return None
            return self._request({"op": "checkpoint", "dir": str(target),
                                  "per_var_limit": per_var_limit,
                                  "total_limit": total_limit})["index"]
        finally:
            self._lock.release()

    def restart(self, checkpoint_dir=_KEEP) -> None:
        """Stop the worker; the next run() starts a fresh one.

        ``checkpoint_dir`` (optional) replaces the directory the next worker
        restores from and checkpoints into.
        """
        with self._lock:
            self._close()
            self._pending_restore = None
            if checkpoint_dir is not _KEEP:
                self.checkpoint_dir = (Path(checkpoint_dir)
                                       if checkpoint_dir is not None else None)

    def close(self) -> None:
        self.restart()
