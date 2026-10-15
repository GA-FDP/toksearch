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
"""KernelExecutor: run_python's out-of-process executor.

Numbers cells, reports each executed cell to ``on_cell`` (the conversation
log), and turns an ExecResult into the text the model sees.
"""

from pathlib import Path

from ..tools import ToolOutput
from .checkpoint import describe_not_restored


def _restart_text(r) -> str:
    rep = r.restore
    lines = []
    if r.restarted:
        lines.append("The Python worker was restarted.")
    if rep and rep.get("found"):
        lead = ("Variables restored from the last checkpoint" if r.restarted
                else "Variables loaded from the saved conversation")
        lines.append(f"{lead}: {', '.join(rep['restored']) or 'none'}.")
        if rep["not_restored"]:
            lines.append("Not restored: "
                         + describe_not_restored(rep["not_restored"])
                         + ". Re-run those cells if they are needed.")
        if r.restarted:
            lines.append("Anything computed after that checkpoint is lost.")
    elif r.restarted:
        lines.append("No checkpoint existed, so all variables were lost.")
    return " ".join(lines)


def format_result(r) -> str:
    parts = []
    if r.stdout:
        parts.append(f"stdout:\n{r.stdout}")
    if r.stderr:
        parts.append(f"stderr:\n{r.stderr}")
    if r.restarted or r.restore:
        parts.append(_restart_text(r))
    for i, fig in enumerate(r.figures, start=1):
        if fig.get("error"):
            parts.append(f"Figure {i} could not be saved: {fig['error']}")
        else:
            paths = [p for p in (fig.get("png_path"), fig.get("html_path")) if p]
            parts.append(f"Figure {i} saved: {', '.join(paths)}")
    return "\n".join(parts) if parts else "(no output)"


class KernelExecutor:
    def __init__(self, client, fig_dir, *, first_cell: int = 1,
                 timeout: float = 600.0, on_cell=None):
        self.client = client
        self.fig_dir = Path(fig_dir)
        self.next_cell = first_cell
        self.timeout = timeout
        self.on_cell = on_cell

    def run(self, code: str, thought: str | None = None) -> ToolOutput:
        cell = self.next_cell
        self.next_cell += 1
        r = self.client.run(code, cell=cell, fig_dir=str(self.fig_dir),
                            timeout=self.timeout)
        if self.on_cell is not None:
            self.on_cell({"cell": cell, "code": code, "thought": thought,
                          "stdout": r.stdout, "stderr": r.stderr,
                          "is_error": r.is_error,
                          "interrupted": r.interrupted,
                          "restarted": r.restarted, "figures": r.figures,
                          "bound": r.bound, "restore": r.restore})
        return ToolOutput(text=format_result(r), is_error=r.is_error,
                          interrupted=r.interrupted)

    def interrupt(self) -> None:
        self.client.interrupt()

    def restart(self) -> None:
        self.client.restart()

    def close(self) -> None:
        self.client.close()
