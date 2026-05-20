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

    def test_error_event_appends_red_message(self):
        from toksearch.llm.gui.app import _build_chat_fn

        class BoomSession:
            def send(self, prompt, **_):
                raise RuntimeError("boom")

        fn = _build_chat_fn(BoomSession())
        yields = self._drain(fn("hi", []))
        last = yields[-1]
        self.assertEqual(last[-1].metadata["status"], "error")
        self.assertIn("boom", last[-1].metadata["title"])

    def test_figure_event_renders_as_inline_plot_bubble(self):
        from toksearch.llm.gui.app import _build_chat_fn
        from toksearch.llm.gui.figure_capture import _active_figure_emitter

        class FakeSession:
            def send(self, prompt, *, on_text, on_tool_call,
                     on_tool_result, **_):
                # Use the active emitter installed by chat_fn to
                # publish a figure mid-turn (simulates what a real
                # run_python call's matplotlib capture would do).
                from toksearch.llm.gui import figure_capture as _fc
                _fc._active_figure_emitter(
                    "plotly", {"data": [], "layout": {}})

        import gradio as gr
        fn = _build_chat_fn(FakeSession())
        yields = self._drain(fn("plot", []))
        last = yields[-1]
        self.assertEqual(len(last), 1)
        plot_bubble = last[0]
        # The content should be a gr.Plot component carrying a Figure.
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
