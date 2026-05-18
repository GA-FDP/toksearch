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
"""Tests for the provider-neutral Message / ContentBlock types."""

import unittest

from toksearch.llm.messages import (
    Message,
    TextBlock,
    ToolUseBlock,
    ToolResultBlock,
    block_to_dict,
    dict_to_block,
    message_to_dict,
    dict_to_message,
)


class TestContentBlocks(unittest.TestCase):
    def test_text_block_kind(self):
        b = TextBlock(text="hi")
        self.assertEqual(b.kind, "text")
        self.assertEqual(b.text, "hi")

    def test_tool_use_block_kind(self):
        b = ToolUseBlock(id="t1", name="run_python", args={"code": "1"})
        self.assertEqual(b.kind, "tool_use")

    def test_tool_result_block_kind(self):
        b = ToolResultBlock(tool_use_id="t1", output="1", is_error=False)
        self.assertEqual(b.kind, "tool_result")


class TestMessage(unittest.TestCase):
    def test_user_text_message(self):
        m = Message(role="user", content=[TextBlock(text="hello")])
        self.assertEqual(m.role, "user")
        self.assertEqual(len(m.content), 1)

    def test_assistant_mixed_content(self):
        m = Message(role="assistant", content=[
            TextBlock(text="let me run code"),
            ToolUseBlock(id="t1", name="run_python",
                         args={"code": "x = 1", "thought": "set x"}),
        ])
        self.assertEqual(len(m.content), 2)


class TestSerialization(unittest.TestCase):
    """Round-trip support for future /save persistence."""

    def test_text_block_round_trip(self):
        b = TextBlock(text="hi")
        d = block_to_dict(b)
        self.assertEqual(d, {"kind": "text", "text": "hi"})
        self.assertEqual(dict_to_block(d), b)

    def test_tool_use_block_round_trip(self):
        b = ToolUseBlock(id="t1", name="run_python", args={"code": "x"})
        d = block_to_dict(b)
        self.assertEqual(d, {"kind": "tool_use", "id": "t1",
                             "name": "run_python", "args": {"code": "x"}})
        self.assertEqual(dict_to_block(d), b)

    def test_tool_result_block_round_trip(self):
        b = ToolResultBlock(tool_use_id="t1", output="ok", is_error=False)
        d = block_to_dict(b)
        self.assertEqual(d, {"kind": "tool_result", "tool_use_id": "t1",
                             "output": "ok", "is_error": False})
        self.assertEqual(dict_to_block(d), b)

    def test_message_round_trip(self):
        m = Message(role="assistant", content=[
            TextBlock(text="a"),
            ToolUseBlock(id="t1", name="run_python", args={"code": "1"}),
        ])
        d = message_to_dict(m)
        self.assertEqual(d["role"], "assistant")
        self.assertEqual(len(d["content"]), 2)
        self.assertEqual(dict_to_message(d), m)

    def test_dict_to_block_rejects_unknown_kind(self):
        with self.assertRaises(ValueError):
            dict_to_block({"kind": "bogus"})


if __name__ == "__main__":
    unittest.main()
