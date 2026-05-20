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
"""Tests for toksearch.llm.gui.app."""

import unittest
from unittest import mock


class _FakeCall:
    def __init__(self, id, name, thought=None, args=None):
        self.id = id
        self.name = name
        self.thought = thought
        self.args = args or {}


class _FakeResult:
    def __init__(self, id, output, is_error=False):
        self.id = id
        self.output = output
        self.is_error = is_error


class TestChatFn(unittest.TestCase):
    def _drain(self, gen):
        """Collect all yields from the generator."""
        return list(gen)

    def test_text_only_response_yields_one_text_bubble(self):
        from toksearch.llm.gui.app import _build_chat_fn

        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                on_text("Hello ")
                on_text("world")

        fn = _build_chat_fn(FakeSession())
        yields = self._drain(fn("hi", []))
        # Last yield's messages list must end with the accumulated text.
        last_messages = yields[-1]
        self.assertEqual(last_messages[-1].content, "Hello world")
        self.assertEqual(last_messages[-1].role, "assistant")
        # Should NOT have metadata (it's a regular text bubble).
        self.assertFalse(getattr(last_messages[-1], "metadata", None))

    def test_tool_call_then_result_renders_as_done_expander(self):
        from toksearch.llm.gui.app import _build_chat_fn

        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                on_tool_call(_FakeCall(
                    id="t1", name="run_python",
                    thought="Fetch shots"))
                on_tool_result(_FakeResult(
                    id="t1", output="ok", is_error=False))

        fn = _build_chat_fn(FakeSession())
        yields = self._drain(fn("do stuff", []))
        last = yields[-1]
        # One bubble that's the completed tool-call.
        self.assertEqual(len(last), 1)
        msg = last[0]
        self.assertIn("run_python", msg.metadata["title"])
        self.assertIn("Fetch shots", msg.metadata["title"])
        self.assertEqual(msg.metadata["status"], "done")
        self.assertIn("ok", msg.content)

    def test_tool_call_with_no_thought_uses_arg_fallback(self):
        from toksearch.llm.gui.app import _build_chat_fn

        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                on_tool_call(_FakeCall(
                    id="t1", name="lookup_docs", thought=None,
                    args={"skill_name": "toksearch-quickstart"}))
                on_tool_result(_FakeResult(
                    id="t1", output="docs body", is_error=False))

        fn = _build_chat_fn(FakeSession())
        yields = self._drain(fn("docs", []))
        last = yields[-1]
        self.assertIn("toksearch-quickstart", last[0].metadata["title"])

    def test_tool_result_is_error_uses_error_title_prefix(self):
        from toksearch.llm.gui.app import _build_chat_fn

        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                on_tool_call(_FakeCall(
                    id="t1", name="run_python", thought="try a thing"))
                on_tool_result(_FakeResult(
                    id="t1", output="Traceback...", is_error=True))

        fn = _build_chat_fn(FakeSession())
        yields = self._drain(fn("do it", []))
        last = yields[-1]
        self.assertEqual(last[0].metadata["status"], "done")
        self.assertTrue(last[0].metadata["title"].startswith("⛔"))

    def test_error_event_appends_red_message(self):
        from toksearch.llm.gui.app import _build_chat_fn

        class BoomSession:
            def send(self, prompt, **_):
                raise RuntimeError("boom")

        fn = _build_chat_fn(BoomSession())
        yields = self._drain(fn("hi", []))
        last = yields[-1]
        # Gradio's ChatMessage.metadata.status only accepts
        # "pending"/"done"; the error marker lives in the title.
        self.assertEqual(last[-1].metadata["status"], "done")
        self.assertIn("⛔", last[-1].metadata["title"])
        self.assertIn("boom", last[-1].metadata["title"])

    def test_plotly_figure_renders_in_an_iframe(self):
        """plotly figures are inlined as gr.HTML containing an
        <iframe srcdoc="..."> so the embedded plotly.js script
        actually executes (the chatbot strips <script> tags from
        directly-inlined HTML even with sanitize_html=False)."""
        from toksearch.llm.gui.app import _build_chat_fn

        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                from toksearch.llm.gui import figure_capture as _fc
                _fc._active_figure_emitter(
                    "plotly", {"data": [], "layout": {}})

        import gradio as gr
        fn = _build_chat_fn(FakeSession())
        yields = self._drain(fn("plot", []))
        last = yields[-1]
        plot_bubble = last[0]
        self.assertIsInstance(plot_bubble.content, gr.HTML)
        html = plot_bubble.content.value
        self.assertIn("<iframe", html)
        self.assertIn("srcdoc=", html)
        # The escaped srcdoc payload should reference plotly.
        self.assertIn("plotly", html.lower())

    def test_matplotlib_figure_renders_as_static_gr_plot(self):
        """Matplotlib figures stay on the gr.Plot path -- they aren't
        interactive in the browser anyway, so a static render is the
        right choice."""
        import matplotlib
        matplotlib.use("Agg")
        # Use the bare Figure constructor (not plt.figure()) so we
        # don't register a figure with pyplot's global state — the
        # matplotlib-capture test in test_llm_gui_figure_capture
        # asserts on plt.get_fignums() and would see a leaked figure.
        from matplotlib.figure import Figure
        from toksearch.llm.gui.app import _build_chat_fn
        import gradio as gr

        fig = Figure()

        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                from toksearch.llm.gui import figure_capture as _fc
                _fc._active_figure_emitter("matplotlib", fig)

        fn = _build_chat_fn(FakeSession())
        yields = self._drain(fn("plot", []))
        last = yields[-1]
        plot_bubble = last[0]
        self.assertIsInstance(plot_bubble.content, gr.Plot)


class TestBuildApp(unittest.TestCase):
    def test_build_app_returns_blocks(self):
        from toksearch.llm.gui.app import build_app
        import gradio as gr

        # Use a no-op chat_fn so build_app doesn't try to spawn a real
        # session.
        def noop_fn(message, history):
            yield []

        blocks = build_app(session=mock.Mock(), fn=noop_fn)
        self.assertIsInstance(blocks, gr.Blocks)


if __name__ == "__main__":
    unittest.main()
