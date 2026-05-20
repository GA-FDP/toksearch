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
"""Local Gradio chat GUI for toksearch.llm."""

import dataclasses as _dc

from ..cli import build_session
from .app import build_app
from . import figure_capture


def launch_gui(args, *, host: str = "127.0.0.1",
                port: int | None = None,
                open_browser: bool = True) -> None:
    """Build a Session from CLI args and launch the Gradio app.

    Wraps the session's ``run_python`` handler so matplotlib figures
    are auto-captured into the chat as inline plot bubbles, and
    installs a custom plotly renderer that routes ``fig.show()``
    calls into the same channel. Both captures publish via
    :data:`figure_capture._active_figure_emitter`, which the chat
    handler swaps in at the start of each turn.
    """
    session = build_session(args)

    def on_figure(kind, payload):
        # Late-bound lookup so each turn's emitter (installed by
        # chat_fn) wins, not the value at wrap time.
        figure_capture._dispatch_to_active_emitter(kind, payload)

    run_python_spec = session._tools_by_name["run_python"]
    wrapped_spec = _dc.replace(
        run_python_spec,
        handler=figure_capture.wrap_run_python_handler(
            run_python_spec.handler, on_figure),
    )
    session._tools_by_name["run_python"] = wrapped_spec
    session.tool_specs = list(session._tools_by_name.values())

    figure_capture.install_plotly_renderer(on_figure)

    blocks = build_app(session)
    blocks.launch(
        server_name=host,
        server_port=port,
        inbrowser=open_browser,
        share=False,
    )


__all__ = ["launch_gui"]
