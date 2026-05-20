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
"""Auto-capture for matplotlib and plotly figures from run_python.

The figure-capture hooks publish to a module-level ``_active_figure_emitter``
that callers swap for the duration of a chat turn. Before any turn is
active the emitter is ``None`` and figures are silently dropped (matches
CLI-only invocations of toksearch.llm).
"""

from typing import Any, Callable

from ..tools import ToolOutput

ToolHandler = Callable[[dict, Any], ToolOutput]
OnFigure = Callable[[str, object], None]
"""Callback signature: ``on_figure(kind, payload)``.

``kind`` is the literal ``"matplotlib"`` or ``"plotly"``; payload is
either a ``matplotlib.figure.Figure`` or a ``dict`` (plotly's
fig_dict)."""


_active_figure_emitter: OnFigure | None = None


def _set_active_figure_emitter(emit_fn: OnFigure | None) -> None:
    """Install the function the matplotlib + plotly hooks call.

    Each chat turn installs an emitter that publishes onto its
    per-turn queue, then restores the previous emitter on exit so
    figures generated outside a turn are silently dropped.
    """
    global _active_figure_emitter
    _active_figure_emitter = emit_fn


def _dispatch_to_active_emitter(kind: str, payload: object) -> None:
    """Forward a captured figure to whichever emitter is currently active.

    Looked up at call time (not at closure-binding time) so a turn's
    emitter installed AFTER ``wrap_run_python_handler`` /
    ``install_plotly_renderer`` were called still wins.
    """
    if _active_figure_emitter is not None:
        _active_figure_emitter(kind, payload)


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


_RENDERER_NAME = "toksearch_gui"


def install_plotly_renderer(on_figure: OnFigure) -> Callable[[], None]:
    """Register a plotly renderer that fires ``on_figure("plotly", fig_dict)``.

    Returns a callable that restores the previous default renderer
    when invoked. Safe to call ``install_plotly_renderer`` multiple
    times — the latest callback wins.
    """
    import plotly.io as pio
    from plotly.io._base_renderers import ExternalRenderer

    class _GuiRenderer(ExternalRenderer):
        def render(self, fig_dict):
            on_figure("plotly", fig_dict)

    pio.renderers[_RENDERER_NAME] = _GuiRenderer()
    previous = pio.renderers.default
    pio.renderers.default = _RENDERER_NAME

    def uninstall() -> None:
        pio.renderers.default = previous

    return uninstall
