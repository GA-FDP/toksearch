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
"""Tests for FakeBackend (the testing seam)."""

import unittest
from types import SimpleNamespace

from toksearch.llm.backends.base import AssistantTurn, Callbacks
from toksearch.llm.backends.fake import FakeBackend
from toksearch.llm.messages import (
    Message, TextBlock, ToolUseBlock, ToolResultBlock,
)
from toksearch.llm.tools import ToolSpec, ToolOutput


def _stub_session(tools=None):
    s = SimpleNamespace()
    s.history = []
    s.namespace = {}
    s.system_prompt = "sys"
    s.model = "fake-1"
    s.tool_specs = tools or []
    s._tools_by_name = {t.name: t for t in s.tool_specs}
    s._append_user = lambda msg: s.history.append(
        Message(role="user", content=[TextBlock(text=msg)]))
    s._append_assistant = lambda blocks: s.history.append(
        Message(role="assistant", content=list(blocks)))
    s._append_tool_result = lambda tid, out, err: s.history.append(
        Message(role="user", content=[ToolResultBlock(
            tool_use_id=tid, output=out, is_error=err)]))
    s._execute_tool = lambda block: s._tools_by_name[block.name].handler(
        block.args, s)
    return s


class TestFakeBackend(unittest.TestCase):
    def test_simple_text_response(self):
        backend = FakeBackend(scripted_turns=[
            AssistantTurn(blocks=[TextBlock(text="hi")],
                          stop_reason="end_turn"),
        ])
        sess = _stub_session()
        out = backend.run_conversation(sess, "hello", Callbacks(),
                                        max_iterations=5)
        self.assertEqual(out.stop_reason, "end_turn")
        self.assertEqual(out.final_text, "hi")

    def test_recording_inspects_calls(self):
        record = []
        backend = FakeBackend(scripted_turns=[
            AssistantTurn(blocks=[TextBlock(text="ok")],
                          stop_reason="end_turn"),
        ], record=record)
        sess = _stub_session()
        backend.run_conversation(sess, "do thing", Callbacks(),
                                  max_iterations=5)
        self.assertEqual(len(record), 1)
        self.assertEqual(record[0]["user_message"], "do thing")
        self.assertEqual(record[0]["model"], "fake-1")

    def test_script_exhausted_raises(self):
        backend = FakeBackend(scripted_turns=[])
        sess = _stub_session()
        with self.assertRaises(RuntimeError):
            backend.run_conversation(sess, "hello", Callbacks(),
                                      max_iterations=5)


if __name__ == "__main__":
    unittest.main()
