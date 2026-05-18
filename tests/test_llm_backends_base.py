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
"""Tests for the Backend ABC and _ToolLoopBackend shared loop.

We construct a minimal subclass (``ScriptedToolLoopBackend``) that returns
pre-scripted ``AssistantTurn``s from ``_send_request``, which lets us verify
the loop's behavior (tool execution, history appending, max_iterations, etc.)
without any provider SDK.
"""

import unittest
from types import SimpleNamespace
from typing import Iterable

from toksearch.llm.backends.base import (
    Backend,
    AssistantTurn,
    Callbacks,
    _ToolLoopBackend,
)
from toksearch.llm.events import TurnComplete
from toksearch.llm.messages import (
    Message, TextBlock, ToolUseBlock, ToolResultBlock,
)
from toksearch.llm.tools import ToolSpec, ToolOutput


def _ok_tool(name="echo"):
    """A trivial tool that returns its input as text."""
    return ToolSpec(
        name=name,
        description="echo",
        input_schema={"type": "object",
                      "properties": {"x": {"type": "string"}}},
        handler=lambda args, sess: ToolOutput(text=args["x"], is_error=False),
    )


class ScriptedToolLoopBackend(_ToolLoopBackend):
    """A _ToolLoopBackend whose _send_request returns scripted turns."""

    name = "scripted"
    default_model = "scripted-1"

    def __init__(self, turns: Iterable[AssistantTurn]):
        self._turns = list(turns)

    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None):
        return self._turns.pop(0)


def _make_session(tools=None):
    """Minimal Session-shaped stub for the loop to drive."""
    s = SimpleNamespace()
    s.history = []
    s.namespace = {}
    s.system_prompt = "sys"
    s.model = "scripted-1"
    s.tool_specs = tools or []
    s._tools_by_name = {t.name: t for t in s.tool_specs}
    def append_user(msg):
        s.history.append(Message(role="user", content=[TextBlock(text=msg)]))
    def append_assistant(blocks):
        s.history.append(Message(role="assistant", content=list(blocks)))
    def append_tool_result(tool_use_id, output, is_error):
        s.history.append(Message(role="user", content=[
            ToolResultBlock(tool_use_id=tool_use_id,
                            output=output, is_error=is_error)]))
    def execute_tool(block: ToolUseBlock) -> ToolOutput:
        return s._tools_by_name[block.name].handler(block.args, s)
    s._append_user = append_user
    s._append_assistant = append_assistant
    s._append_tool_result = append_tool_result
    s._execute_tool = execute_tool
    return s


class TestBackendABC(unittest.TestCase):
    def test_backend_is_abstract(self):
        with self.assertRaises(TypeError):
            Backend()


class TestToolLoop(unittest.TestCase):
    def test_end_turn_with_only_text(self):
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[TextBlock(text="all done")],
                          stop_reason="end_turn"),
        ])
        sess = _make_session()
        cbs = Callbacks()
        result = backend.run_conversation(sess, "hello", cbs, max_iterations=5)
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.final_text, "all done")
        # History: user prompt + assistant turn
        self.assertEqual(len(sess.history), 2)
        self.assertEqual(sess.history[0].role, "user")
        self.assertEqual(sess.history[1].role, "assistant")

    def test_tool_use_then_end(self):
        tool = _ok_tool("echo")
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[ToolUseBlock(id="t1", name="echo",
                                               args={"x": "hi"})],
                          stop_reason="tool_use"),
            AssistantTurn(blocks=[TextBlock(text="ok")],
                          stop_reason="end_turn"),
        ])
        sess = _make_session(tools=[tool])
        calls, results = [], []
        cbs = Callbacks(on_tool_call=calls.append,
                        on_tool_result=results.append)
        out = backend.run_conversation(sess, "go", cbs, max_iterations=5)
        self.assertEqual(out.stop_reason, "end_turn")
        # One tool call, one tool result fired
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "echo")
        self.assertEqual(results[0].output, "hi")
        # History: user, assistant(tool_use), user(tool_result), assistant(text)
        self.assertEqual([m.role for m in sess.history],
                         ["user", "assistant", "user", "assistant"])

    def test_max_iterations_caps_loop(self):
        # Backend keeps emitting tool_use forever
        tool = _ok_tool("echo")
        forever = [AssistantTurn(blocks=[ToolUseBlock(id=f"t{i}", name="echo",
                                                      args={"x": str(i)})],
                                 stop_reason="tool_use")
                   for i in range(10)]
        backend = ScriptedToolLoopBackend(forever)
        sess = _make_session(tools=[tool])
        out = backend.run_conversation(sess, "go", Callbacks(),
                                        max_iterations=3)
        self.assertEqual(out.stop_reason, "max_iterations")

    def test_confirm_false_aborts(self):
        tool = _ok_tool("echo")
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[ToolUseBlock(id="t1", name="echo",
                                               args={"x": "hi"})],
                          stop_reason="tool_use"),
        ])
        sess = _make_session(tools=[tool])
        cbs = Callbacks(confirm=lambda call: False)
        out = backend.run_conversation(sess, "go", cbs, max_iterations=5)
        self.assertEqual(out.stop_reason, "interrupted")

    def test_on_event_catch_all_fires_for_everything(self):
        tool = _ok_tool("echo")
        backend = ScriptedToolLoopBackend([
            AssistantTurn(blocks=[ToolUseBlock(id="t1", name="echo",
                                               args={"x": "hi"})],
                          stop_reason="tool_use"),
            AssistantTurn(blocks=[TextBlock(text="done")],
                          stop_reason="end_turn"),
        ])
        sess = _make_session(tools=[tool])
        events = []
        cbs = Callbacks(on_event=events.append)
        backend.run_conversation(sess, "go", cbs, max_iterations=5)
        kinds = [type(e).__name__ for e in events]
        # ToolCall, ToolResult, TurnComplete at minimum (text deltas optional)
        self.assertIn("ToolCall", kinds)
        self.assertIn("ToolResult", kinds)
        self.assertIn("TurnComplete", kinds)


if __name__ == "__main__":
    unittest.main()
