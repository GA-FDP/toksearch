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

"""Child process that owns the persistent run_python namespace.

Started by KernelClient with multiprocessing's spawn method; exchanges plain
dicts over a Pipe. One SIGINT handler is installed at start-up; it raises
KeyboardInterrupt only while user code or a figure save runs, so a cell can
be stopped without losing the namespace and a stray SIGINT can never end the
worker. The worker leads its own process group (the client signals the whole
group, so processes a cell started die with it) and kills that group if its
parent dies. Captured stdout/stderr are bounded.

Accepted limitations: bound names are tracked by ``id()``, so a rebind that
happens to reuse an address is missed; baseline names (``np``, ``toksearch``,
...) that the user rebinds are not checkpointed.
"""

import contextlib
import io
import os
import signal
import threading
import time
import traceback
from pathlib import Path

from .checkpoint import (follow_carried, load_namespace, save_namespace,
                         user_names)
from .figures import save_figure

HEAD_CHARS = 4_000_000
TAIL_CHARS = 1_000_000


class BoundedText(io.TextIOBase):
    """A text sink keeping the first ``head`` and last ``tail`` characters."""

    def __init__(self, head: int = HEAD_CHARS, tail: int = TAIL_CHARS):
        self._head_max, self._tail_max = head, tail
        self._head, self._head_len = [], 0
        self._tail, self._tail_len = [], 0
        self._dropped = 0

    def writable(self) -> bool:
        return True

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            raise TypeError(f"write() argument must be str, not "
                            f"{type(s).__name__}")
        n = len(s)
        room = self._head_max - self._head_len
        if room > 0:
            part = s[:room]
            self._head.append(part)
            self._head_len += len(part)
            s = s[room:]
        if s:
            self._tail.append(s)
            self._tail_len += len(s)
            if self._tail_len > 2 * self._tail_max:
                self._compact()
        return n

    def _compact(self) -> None:
        text = "".join(self._tail)
        keep = text[len(text) - self._tail_max:] if self._tail_max else ""
        self._dropped += len(text) - len(keep)
        self._tail, self._tail_len = [keep], len(keep)

    def getvalue(self) -> str:
        if self._tail_len > self._tail_max:
            self._compact()
        head, tail = "".join(self._head), "".join(self._tail)
        if self._dropped:
            return (f"{head}\n…[{self._dropped} characters dropped]…\n"
                    f"{tail}")
        return head + tail


def build_namespace(packages: list[str] | None) -> dict:
    """The namespace Session builds: defaults plus discovered contributors."""
    from ..discovery import discover_namespace_contributors
    from ..session import _default_namespace
    ns = _default_namespace()
    for name, value, _desc in discover_namespace_contributors():
        if packages is None or name in packages:
            ns[name] = value
    return ns


def _install_plotly_capture(sink: list) -> None:
    # Not toksearch.llm.gui.figure_capture: importing toksearch.llm.gui pulls in gradio.
    try:
        import plotly.io as pio
        from plotly.io._base_renderers import ExternalRenderer
    except ImportError:
        return

    class _Capture(ExternalRenderer):
        def render(self, fig_dict):
            sink.append(("plotly", fig_dict))

    pio.renderers["toksearch_kernel"] = _Capture()
    pio.renderers.default = "toksearch_kernel"


def _matplotlib_figures():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return [], (lambda: None)
    figs = [("matplotlib", plt.figure(n)) for n in plt.get_fignums()]
    return figs, (lambda: plt.close("all"))


class Worker:
    def __init__(self, packages: list[str] | None):
        self._interruptible = False
        signal.signal(signal.SIGINT, self._on_sigint)
        self.ns = build_namespace(packages)
        self.baseline = frozenset(self.ns)
        self.origins: dict[str, int] = {}
        # name -> (pickle path, entry) for variables the last restore could
        # not load; carried into checkpoints until a cell binds the name.
        self.carried: dict[str, tuple] = {}
        self.captured: list = []
        _install_plotly_capture(self.captured)

    def _on_sigint(self, signum, frame):
        if self._interruptible:
            raise KeyboardInterrupt

    def _reinstall_sigint(self) -> None:
        # A cell may have replaced the handler (signal.signal(SIGINT, ...)).
        if signal.getsignal(signal.SIGINT) != self._on_sigint:
            signal.signal(signal.SIGINT, self._on_sigint)

    def _ids(self) -> dict[str, int]:
        return {k: id(self.ns[k]) for k in user_names(self.ns, self.baseline)}

    def execute(self, req: dict) -> dict:
        out, err = BoundedText(), BoundedText()
        is_error = interrupted = False
        self.captured.clear()
        before = self._ids()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                self._interruptible = True
                exec(req["code"], self.ns)  # noqa: S102 -- intentional
                self._interruptible = False
            except KeyboardInterrupt:
                self._interruptible = False
                is_error = interrupted = True
                err.write("Interrupted.\n")
            except BaseException:  # includes SystemExit: exit() must not end the worker
                self._interruptible = False
                is_error = True
                err.write(traceback.format_exc())
            finally:
                self._interruptible = False
                self._reinstall_sigint()
        after = self._ids()
        bound = sorted(k for k, i in after.items() if before.get(k) != i)
        for k in bound:
            self.origins[k] = req["cell"]
            self.carried.pop(k, None)
        for k in [k for k in self.origins if k not in after]:
            del self.origins[k]

        figures, fig_interrupted = self._save_figures(req)
        if fig_interrupted:
            is_error = interrupted = True
            err.write("Interrupted while saving figures.\n")
        return {"stdout": out.getvalue(), "stderr": err.getvalue(),
                "is_error": is_error, "interrupted": interrupted,
                "figures": figures, "bound": bound}

    def _save_figures(self, req: dict) -> tuple[list, bool]:
        mpl, close_mpl = _matplotlib_figures()
        figures, skip = [], False
        for i, (kind, payload) in enumerate(mpl + list(self.captured), start=1):
            if skip:
                figures.append({"kind": kind, "error": "skipped: interrupted"})
                continue
            try:
                self._interruptible = True
                rec = save_figure(kind, payload, req["fig_dir"],
                                  f"c{req['cell']}-{i}")
                self._interruptible = False
                figures.append(rec)
            except KeyboardInterrupt:
                self._interruptible = False
                skip = True
                figures.append({"kind": kind, "error": "skipped: interrupted"})
            except Exception as e:
                self._interruptible = False
                figures.append({"kind": kind,
                                "error": f"{type(e).__name__}: {e}"})
            finally:
                self._interruptible = False
        close_mpl()
        self.captured.clear()
        return figures, skip

    def checkpoint(self, req: dict) -> dict:
        out_dir = Path(req["dir"])
        try:
            index = save_namespace(
                self.ns, self.baseline, self.origins, out_dir,
                per_var_limit=req["per_var_limit"],
                total_limit=req["total_limit"], carried=self.carried)
        finally:
            # Point carried pickles at where they are now, whether the swap
            # finished or failed part-way.
            follow_carried(self.carried, out_dir)
        return {"index": index}

    def restore(self, req: dict) -> dict:
        report = load_namespace(self.ns, req["dir"])
        self.origins.update(report["origins"])
        self.carried = {e["name"]: (e["path"], e)
                        for e in report["not_restored"]
                        if e["reason"] == "unloadable" and e.get("path")}
        return {"report": report}

    def handle(self, req: dict) -> dict:
        return getattr(self, req["op"])(req)


def _watch_parent(ppid: int) -> None:
    """Kill this worker's process group once its parent has gone."""
    while True:
        time.sleep(1.0)
        if os.getppid() != ppid:
            if os.getpgid(0) == os.getpid():   # we lead our own group
                os.killpg(os.getpid(), signal.SIGKILL)
            os.kill(os.getpid(), signal.SIGKILL)


def worker_main(conn, packages: list[str] | None, ppid: int | None = None) -> None:
    try:
        os.setsid()   # own process group: the client signals all of it
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # until Worker installs its handler
    if ppid is None:
        ppid = os.getppid()
    threading.Thread(target=_watch_parent, args=(ppid,), daemon=True,
                     name="toksearch-parent-watch").start()
    # Keep C-library output off the parent's stdout, which may be a protocol
    # channel (an MCP stdio server, a piped CLI).
    os.dup2(2, 1)
    worker = Worker(packages)
    conn.send({"ready": True})
    while True:
        try:
            req = conn.recv()
        except EOFError:
            return
        try:
            reply = worker.handle(req)
        except BaseException as e:   # never let an op (or SIGINT) end the worker
            reply = {"error": f"{type(e).__name__}: {e}"}
        conn.send(reply)
