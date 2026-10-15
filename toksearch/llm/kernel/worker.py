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
dicts over a Pipe. SIGINT is ignored except while user code runs, where it
raises KeyboardInterrupt so a cell can be stopped without losing the
namespace.
"""

import contextlib
import io
import os
import signal
import traceback

from .checkpoint import load_namespace, save_namespace, user_names
from .figures import save_figure


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
        self.ns = build_namespace(packages)
        self.baseline = frozenset(self.ns)
        self.origins: dict[str, int] = {}
        self.captured: list = []
        _install_plotly_capture(self.captured)

    def _ids(self) -> dict[str, int]:
        return {k: id(self.ns[k]) for k in user_names(self.ns, self.baseline)}

    def execute(self, req: dict) -> dict:
        out, err = io.StringIO(), io.StringIO()
        is_error = interrupted = False
        self.captured.clear()
        before = self._ids()
        signal.signal(signal.SIGINT, signal.default_int_handler)
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                exec(req["code"], self.ns)  # noqa: S102 -- intentional
        except KeyboardInterrupt:
            is_error = interrupted = True
            err.write("Interrupted.\n")
        except BaseException:  # includes SystemExit: exit() must not end the worker
            is_error = True
            err.write(traceback.format_exc())
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        after = self._ids()
        bound = sorted(k for k, i in after.items() if before.get(k) != i)
        for k in bound:
            self.origins[k] = req["cell"]
        for k in [k for k in self.origins if k not in after]:
            del self.origins[k]

        mpl, close_mpl = _matplotlib_figures()
        figures = []
        for i, (kind, payload) in enumerate(mpl + list(self.captured), start=1):
            try:
                figures.append(save_figure(kind, payload, req["fig_dir"],
                                           f"c{req['cell']}-{i}"))
            except Exception as e:
                figures.append({"kind": kind,
                                "error": f"{type(e).__name__}: {e}"})
        close_mpl()
        self.captured.clear()
        return {"stdout": out.getvalue(), "stderr": err.getvalue(),
                "is_error": is_error, "interrupted": interrupted,
                "figures": figures, "bound": bound}

    def checkpoint(self, req: dict) -> dict:
        return {"index": save_namespace(
            self.ns, self.baseline, self.origins, req["dir"],
            per_var_limit=req["per_var_limit"],
            total_limit=req["total_limit"])}

    def restore(self, req: dict) -> dict:
        report = load_namespace(self.ns, req["dir"])
        self.origins.update(report["origins"])
        return {"report": report}

    def handle(self, req: dict) -> dict:
        try:
            return getattr(self, req["op"])(req)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}


def worker_main(conn, packages: list[str] | None) -> None:
    # Keep C-library output off the parent's stdout, which may be a protocol
    # channel (an MCP stdio server, a piped CLI).
    os.dup2(2, 1)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    worker = Worker(packages)
    conn.send({"ready": True})
    while True:
        try:
            req = conn.recv()
        except EOFError:
            return
        conn.send(worker.handle(req))
