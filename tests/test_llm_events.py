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
"""Tests for the event dataclasses dispatched by Session.send()."""

import unittest

from toksearch.llm.events import (
    TextDelta,
    ToolCall,
    ToolResult,
    TurnComplete,
)


class TestEvents(unittest.TestCase):
    def test_text_delta_holds_text(self):
        e = TextDelta(text="hello")
        self.assertEqual(e.text, "hello")

    def test_tool_call_fields(self):
        e = ToolCall(id="abc", name="run_python",
                     args={"code": "x = 1", "thought": "set x"},
                     thought="set x")
        self.assertEqual(e.id, "abc")
        self.assertEqual(e.name, "run_python")
        self.assertEqual(e.args, {"code": "x = 1", "thought": "set x"})
        self.assertEqual(e.thought, "set x")

    def test_tool_call_thought_optional(self):
        e = ToolCall(id="abc", name="lookup_docs",
                     args={"skill_name": "toksearch-pipeline"},
                     thought=None)
        self.assertIsNone(e.thought)

    def test_tool_result_fields(self):
        e = ToolResult(id="abc", output="42", is_error=False)
        self.assertFalse(e.is_error)
        self.assertEqual(e.output, "42")

    def test_turn_complete_fields(self):
        e = TurnComplete(stop_reason="end_turn", final_text="done")
        self.assertEqual(e.stop_reason, "end_turn")
        self.assertEqual(e.final_text, "done")

    def test_events_are_frozen(self):
        e = TextDelta(text="hi")
        with self.assertRaises(Exception):  # FrozenInstanceError or AttributeError
            e.text = "bye"

    def test_events_compare_by_value(self):
        a = TextDelta(text="hi")
        b = TextDelta(text="hi")
        c = TextDelta(text="bye")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
