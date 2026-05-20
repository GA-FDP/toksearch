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
"""Gradio chat GUI for ``toksearch.llm``.

Built on ``gr.ChatInterface`` so Gradio owns the input box, history,
streaming UI, and message rendering. We provide a single generator
that yields the list of ``gr.ChatMessage`` instances comprising the
assistant's response: assistant text, tool-call expanders (rendered
via ``ChatMessage.metadata``), and inline plots
(``ChatMessage(content=gr.Plot(fig))``).
"""

import queue
import sys
import threading
import traceback

import gradio as gr

from .figure_capture import _set_active_figure_emitter


_ARG_FALLBACKS = ("skill_name", "name", "query")


def _tool_title(call) -> str:
    """Build the title shown on a tool-call expander."""
    detail = (getattr(call, "thought", None) or "").strip()
    if not detail:
        args = getattr(call, "args", {}) or {}
        for key in _ARG_FALLBACKS:
            value = args.get(key)
            if value:
                detail = str(value)
                break
    suffix = f": {detail}" if detail else ""
    return f"🔧 {call.name}{suffix}"


def _truncate(body: str, limit: int = 2000) -> str:
    if len(body) <= limit:
        return body
    return body[:limit] + f"\n…(truncated, {len(body) - limit} more chars)…"


def _to_plotly_figure(payload):
    """Coerce a captured plotly payload (fig_dict) into a go.Figure."""
    if isinstance(payload, dict):
        import plotly.graph_objects as go
        return go.Figure(payload)
    return payload


def _build_chat_fn(session):
    """Construct the streaming ChatInterface function for this Session."""

    def chat_fn(message, history):
        q: queue.Queue = queue.Queue()

        def runner():
            try:
                session.send(
                    message,
                    on_text=lambda t: q.put(("text", t)),
                    on_tool_call=lambda c: q.put(("tool_call", c)),
                    on_tool_result=lambda r: q.put(("tool_result", r)),
                )
            except Exception as e:
                q.put(("error", (str(e), traceback.format_exc())))
            finally:
                q.put(("done", None))

        from . import figure_capture as _fc
        previous_emitter = _fc._active_figure_emitter
        _set_active_figure_emitter(
            lambda kind, payload: q.put(("figure", (kind, payload))))

        threading.Thread(target=runner, daemon=True).start()

        messages: list[gr.ChatMessage] = []
        pending: dict[str, int] = {}  # tool_use_id -> index into messages

        try:
            while True:
                kind, payload = q.get()
                if kind == "done":
                    return

                if kind == "text":
                    if (messages
                            and not messages[-1].metadata
                            and isinstance(messages[-1].content, str)):
                        # Extend the trailing text bubble.
                        messages[-1] = gr.ChatMessage(
                            role="assistant",
                            content=messages[-1].content + payload,
                        )
                    else:
                        messages.append(gr.ChatMessage(
                            role="assistant", content=payload,
                        ))

                elif kind == "tool_call":
                    title = _tool_title(payload)
                    messages.append(gr.ChatMessage(
                        role="assistant",
                        content="(running…)",
                        metadata={"title": title, "status": "pending"},
                    ))
                    pending[payload.id] = len(messages) - 1

                elif kind == "tool_result":
                    idx = pending.pop(payload.id, None)
                    if idx is not None:
                        old = messages[idx]
                        body = _truncate(payload.output or "(no output)")
                        messages[idx] = gr.ChatMessage(
                            role="assistant",
                            content=f"```\n{body}\n```",
                            metadata={
                                "title": old.metadata["title"],
                                "status": ("error" if payload.is_error
                                           else "done"),
                            },
                        )

                elif kind == "figure":
                    fig_kind, fig_payload = payload
                    if fig_kind == "plotly":
                        fig = _to_plotly_figure(fig_payload)
                    else:
                        fig = fig_payload
                    messages.append(gr.ChatMessage(
                        role="assistant",
                        content=gr.Plot(fig),
                    ))

                elif kind == "error":
                    err_msg, err_tb = payload
                    print(f"[gui] chat_fn error: {err_msg}\n{err_tb}",
                          file=sys.stderr, flush=True)
                    messages.append(gr.ChatMessage(
                        role="assistant",
                        content=f"```\n{err_tb}\n```",
                        metadata={"title": f"⛔ {err_msg}",
                                  "status": "error"},
                    ))

                yield messages
        finally:
            _set_active_figure_emitter(previous_emitter)

    return chat_fn


def build_app(session, fn=None):
    """Construct the chat GUI Blocks app for a given Session.

    ``fn`` defaults to a streaming generator that drives
    ``session.send``. Tests can pass a synchronous stub to isolate the
    UI wiring from the LLM call.
    """
    chat_fn = fn if fn is not None else _build_chat_fn(session)
    with gr.Blocks(title="toksearch chat") as blocks:
        gr.ChatInterface(
            fn=chat_fn,
            chatbot=gr.Chatbot(
                label="Chat",
                height=700,
                # Permit gr.Plot content inside ChatMessages.
                sanitize_html=False,
            ),
            title="toksearch chat",
        )
    return blocks
