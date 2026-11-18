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

"""Tests for the recap and the hidden restore note."""

import unittest

from toksearch.llm.conversation.recap import build_recap
from toksearch.llm.conversation.restore_note import restore_messages

EVENTS = [
    {"type": "user", "text": "Plot Ip for MAST 30420"},
    {"type": "tool_call", "id": "t1", "name": "run_python", "args": {}},
    {"type": "cell", "cell": 1, "code": "sig = fetch(30420)"},
    {"type": "assistant_text", "text": "Here is the plot."},
    {"type": "turn_end", "status": "end_turn"},
]


class TestRecap(unittest.TestCase):
    def test_contents_in_order(self):
        text = build_recap(EVENTS)
        self.assertTrue(text.startswith("Summary of the earlier conversation"))
        u = text.index("User: Plot Ip for MAST 30420")
        c = text.index("Cell 1:\n```python\nsig = fetch(30420)\n```")
        a = text.index("Assistant: Here is the plot.")
        self.assertLess(u, c)
        self.assertLess(c, a)

    def test_budget_keeps_newest(self):
        events = [{"type": "user", "text": f"question {i} " + "x" * 100}
                  for i in range(100)]
        text = build_recap(events, budget=1_000)
        self.assertIn("question 99", text)
        self.assertNotIn("question 0 ", text)
        self.assertIn("earliest items are omitted", text)
        self.assertLess(len(text), 1_300)


class TestRestoreMessages(unittest.TestCase):
    INDEX = {"variables": [
        {"name": "sig", "status": "saved", "cell": 1},
        {"name": "df", "status": "too_large", "cell": 7},
        {"name": "conn", "status": "unpicklable", "cell": 2,
         "error": "TypeError: no"}]}

    def test_full_note(self):
        note, line = restore_messages(self.INDEX, "1.34.0", "1.35.0")
        self.assertTrue(note.startswith("[Note from the app, not the user:"))
        self.assertIn("Python variables available: sig.", note)
        self.assertIn("`df` (too large to save; bound by cell 7)", note)
        self.assertIn("`conn` (could not be saved; bound by cell 2)", note)
        self.assertIn("Environment changed: fdp-core 1.34.0 → 1.35.0.", note)
        self.assertTrue(note.endswith("]"))
        self.assertEqual(
            line,
            "Conversation restored. 1 variable restored. Not restored: "
            "`df` (too large to save; bound by cell 7), `conn` (could not "
            "be saved; bound by cell 2). Environment changed: fdp-core "
            "1.34.0 → 1.35.0.")

    def test_no_checkpoint_same_env(self):
        note, line = restore_messages(None, "1.35.0", "1.35.0")
        self.assertIn("No Python variables had been saved.", note)
        self.assertNotIn("Environment changed", note)
        self.assertEqual(line, "Conversation restored. No variables saved.")

    def test_with_recap(self):
        note, line = restore_messages(None, None, None, recap="RECAP")
        self.assertIn("a summary follows", note)
        self.assertTrue(note.endswith("]\n\nRECAP"))
        self.assertIn("rebuilt from a summary", line)


if __name__ == "__main__":
    unittest.main()
