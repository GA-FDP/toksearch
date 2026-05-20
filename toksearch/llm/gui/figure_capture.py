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
"""Auto-capture for matplotlib and plotly figures from run_python."""

from typing import Any, Callable

from ..tools import ToolOutput

ToolHandler = Callable[[dict, Any], ToolOutput]
OnFigure = Callable[[str, object], None]
"""Callback signature: ``on_figure(kind, payload)``.

``kind`` is the literal ``"matplotlib"`` or ``"plotly"``; payload is
either a ``matplotlib.figure.Figure`` or a ``dict`` (plotly's
fig_dict)."""


def wrap_run_python_handler(handler: ToolHandler,
                              on_figure: OnFigure) -> ToolHandler:
    """Wrap a ``run_python`` handler to emit matplotlib figures.

    After the inner handler returns, every figure registered with
    ``pyplot.get_fignums()`` is delivered to ``on_figure`` and then
    closed. The inner handler's return value is passed through
    unchanged.
    """

    def wrapped(args, session):
        result = handler(args, session)
        import matplotlib.pyplot as plt
        for num in plt.get_fignums():
            on_figure("matplotlib", plt.figure(num))
        plt.close("all")
        return result

    return wrapped
