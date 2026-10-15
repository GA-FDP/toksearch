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

"""Tests for the conversation folder, its event log and history file."""

import json
import os
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from toksearch.llm.conversation.events import EventLog
from toksearch.llm.conversation.folder import (
    ConversationFolder, default_root, list_conversations,
)
from toksearch.llm.conversation.history import load_history, save_history
from toksearch.llm.messages import Message, TextBlock, ToolUseBlock


class TestFolder(unittest.TestCase):
    def test_create_names_and_meta(self):
        with TemporaryDirectory() as tmp:
            with mock.patch.dict(os.environ, {"FDP_CORE_VERSION": "1.40.0"}):
                f = ConversationFolder.create(
                    tmp, now=datetime(2026, 10, 15, 9, 30, 5))
            self.assertEqual(f.path, Path(tmp) / "2026-10-15-093005")
            meta = f.read_meta()
            self.assertEqual(len(meta["id"]), 36)
            self.assertEqual(meta["title"], "")
            self.assertEqual(meta["fdp_core_version"], "1.40.0")
            self.assertIsNone(meta["claude_session_id"])
            self.assertEqual(f.namespace_dir, f.path / "namespace")
            self.assertEqual(f.figures_dir, f.path / "figures")
            self.assertEqual(f.claude_dir, f.path / "claude")

    def test_same_second_gets_a_suffix(self):
        with TemporaryDirectory() as tmp:
            now = datetime(2026, 10, 15, 9, 30, 5)
            a = ConversationFolder.create(tmp, now=now)
            b = ConversationFolder.create(tmp, now=now)
            self.assertNotEqual(a.path, b.path)
            self.assertEqual(b.path.name, "2026-10-15-093005-2")

    def test_update_meta(self):
        with TemporaryDirectory() as tmp:
            f = ConversationFolder.create(tmp)
            f.update_meta(title="MAST Ip")
            self.assertEqual(f.read_meta()["title"], "MAST Ip")

    def test_list_newest_first_with_sizes(self):
        with TemporaryDirectory() as tmp:
            a = ConversationFolder.create(tmp, now=datetime(2026, 1, 1))
            b = ConversationFolder.create(tmp, now=datetime(2026, 1, 2))
            a.update_meta(last_active="2026-01-03T00:00:00")
            (Path(tmp) / "stray").mkdir()
            items = list_conversations(tmp)
            self.assertEqual([i["path"] for i in items],
                             [str(a.path), str(b.path)])
            self.assertGreater(items[0]["size_bytes"], 0)
            self.assertEqual(list_conversations(Path(tmp) / "none"), [])

    def test_default_root(self):
        with mock.patch.dict(os.environ, {"TOKSEARCH_SESSIONS_DIR": "/x/y"}):
            self.assertEqual(default_root(), Path("/x/y"))
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(default_root(), Path.home() / "FDP" / "sessions")


class TestEventLog(unittest.TestCase):
    def test_round_trip_and_timestamps(self):
        with TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.jsonl")
            self.assertEqual(log.read(), [])
            log.append({"type": "user", "text": "hi"})
            log.append({"type": "turn_end", "status": "end_turn"})
            log.sync()
            events = log.read()
            self.assertEqual([e["type"] for e in events],
                             ["user", "turn_end"])
            self.assertIn("t", events[0])

    def test_truncated_last_line_is_ignored(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "events.jsonl"
            log = EventLog(p)
            log.append({"type": "user", "text": "hi"})
            with open(p, "a") as f:
                f.write('{"type": "assist')
            self.assertEqual(len(log.read()), 1)


class TestHistory(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "history.jsonl"
            self.assertIsNone(load_history(p))
            hist = [Message(role="user", content=[TextBlock(text="q")]),
                    Message(role="assistant", content=[ToolUseBlock(
                        id="t", name="run_python", args={"code": "1"})])]
            save_history(p, hist)
            self.assertEqual(load_history(p), hist)


if __name__ == "__main__":
    unittest.main()
