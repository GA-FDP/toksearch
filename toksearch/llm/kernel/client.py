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

On timeout (or interrupt()) the worker gets SIGINT and keeps its namespace;
if it does not answer within ``grace`` seconds it is killed and restarted. A
crashed worker is restarted on the spot. Whenever a worker starts and
``checkpoint_dir`` holds a checkpoint, the checkpoint is loaded and the
report is attached to the next ExecResult.
"""

import multiprocessing as mp
import os
import signal
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .checkpoint import PER_VAR_LIMIT, TOTAL_LIMIT, find_checkpoint
from .worker import worker_main


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
        self._proc = None
        self._conn = None
        self._pending_restore = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.is_alive()

    # ---- process management (callers hold self._lock) ----

    def _start(self) -> None:
        parent, child = self._ctx.Pipe()
        proc = self._ctx.Process(target=worker_main,
                                 args=(child, self.packages), daemon=True)
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
            proc.kill()
            proc.join(5)
            parent.close()
            raise
        self._proc, self._conn = proc, parent
        if (self.checkpoint_dir is not None
                and find_checkpoint(self.checkpoint_dir) is not None):
            reply = self._request({"op": "restore",
                                   "dir": str(self.checkpoint_dir)})
            self._pending_restore = reply["report"]

    def _close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        if self._proc is not None:
            self._proc.kill()
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
            self._close()
            raise KernelError(f"The Python worker crashed during {req['op']}.")
        if "error" in reply:
            raise KernelError(f"{req['op']} failed: {reply['error']}")
        return reply

    # ---- public API ----

    def run(self, code: str, *, cell: int, fig_dir: str,
            timeout: float = 600.0) -> ExecResult:
        with self._lock:
            self._ensure_started()
            self._conn.send({"op": "execute", "code": code, "cell": cell,
                             "fig_dir": fig_dir})
            if self._conn.poll(timeout):
                result = self._receive()
            else:
                os.kill(self._proc.pid, signal.SIGINT)
                if self._conn.poll(self._grace):
                    result = self._receive()
                else:
                    self._close()
                    self._start()
                    result = ExecResult(
                        stderr=(f"The code did not stop within {timeout:.0f} s"
                                f", nor {self._grace:.0f} s after an "
                                "interrupt, so the worker was killed.\n"),
                        is_error=True, interrupted=True, restarted=True)
            result.restore, self._pending_restore = self._pending_restore, None
            return result

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
        """Interrupt the running cell. Safe to call from another thread."""
        proc = self._proc
        if proc is not None and proc.is_alive():
            os.kill(proc.pid, signal.SIGINT)

    def checkpoint(self, out_dir=None, *, per_var_limit: int = PER_VAR_LIMIT,
                   total_limit: int = TOTAL_LIMIT) -> dict | None:
        """Checkpoint the namespace; None when no worker is running."""
        with self._lock:
            if not self.running:
                return None
            target = out_dir if out_dir is not None else self.checkpoint_dir
            return self._request({"op": "checkpoint", "dir": str(target),
                                  "per_var_limit": per_var_limit,
                                  "total_limit": total_limit})["index"]

    def restart(self) -> None:
        """Stop the worker; the next run() starts a fresh one."""
        with self._lock:
            self._close()
            self._pending_restore = None

    def close(self) -> None:
        self.restart()
