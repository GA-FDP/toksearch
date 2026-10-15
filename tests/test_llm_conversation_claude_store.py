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

"""Tests for FolderSessionStore, the claude-agent-sdk transcript mirror."""

import asyncio
import unittest
import uuid
from pathlib import Path
from tempfile import TemporaryDirectory

from toksearch.llm.conversation.claude_store import FolderSessionStore

SID = str(uuid.uuid4())


def run(coro):
    return asyncio.run(coro)


class TestFolderSessionStore(unittest.TestCase):
    def test_append_load_and_dedup(self):
        with TemporaryDirectory() as tmp:
            s = FolderSessionStore(Path(tmp) / "claude")
            key = {"project_key": "anything", "session_id": SID}
            self.assertIsNone(run(s.load(key)))
            self.assertFalse(s.has_transcript(SID))
            run(s.append(key, [{"type": "user", "uuid": "a"},
                               {"type": "title"}]))
            run(s.append(key, [{"type": "user", "uuid": "a"},
                               {"type": "assistant", "uuid": "b"}]))
            self.assertEqual(run(s.load(key)),
                             [{"type": "user", "uuid": "a"}, {"type": "title"},
                              {"type": "assistant", "uuid": "b"}])
            self.assertTrue(s.has_transcript(SID))

    def test_project_key_is_ignored(self):
        with TemporaryDirectory() as tmp:
            s = FolderSessionStore(tmp)
            run(s.append({"project_key": "p1", "session_id": SID},
                         [{"type": "user", "uuid": "a"}]))
            self.assertEqual(
                len(run(s.load({"project_key": "p2", "session_id": SID}))), 1)

    def test_subkeys(self):
        with TemporaryDirectory() as tmp:
            s = FolderSessionStore(tmp)
            sub = {"project_key": "p", "session_id": SID,
                   "subpath": "subagents/agent-1"}
            run(s.append(sub, [{"type": "user", "uuid": "z"}]))
            self.assertEqual(run(s.load(sub)), [{"type": "user", "uuid": "z"}])
            self.assertEqual(
                run(s.list_subkeys({"project_key": "p", "session_id": SID})),
                ["subagents/agent-1"])
            self.assertIsNone(
                run(s.load({"project_key": "p", "session_id": SID})))

    def test_truncated_line_skipped(self):
        with TemporaryDirectory() as tmp:
            s = FolderSessionStore(tmp)
            key = {"project_key": "p", "session_id": SID}
            run(s.append(key, [{"type": "user", "uuid": "a"}]))
            with open(Path(tmp) / f"{SID}.jsonl", "a") as f:
                f.write('{"type": "assi')
            self.assertEqual(len(run(s.load(key))), 1)

    def test_rejects_bad_keys(self):
        with TemporaryDirectory() as tmp:
            s = FolderSessionStore(tmp)
            with self.assertRaises(ValueError):
                run(s.load({"project_key": "p", "session_id": "../x"}))
            with self.assertRaises(ValueError):
                run(s.load({"project_key": "p", "session_id": SID,
                            "subpath": "../escape"}))

    def test_list_sessions(self):
        with TemporaryDirectory() as tmp:
            s = FolderSessionStore(tmp)
            run(s.append({"project_key": "p", "session_id": SID},
                         [{"type": "user"}]))
            listed = run(s.list_sessions("p"))
            self.assertEqual([x["session_id"] for x in listed], [SID])


class TestSdkMaterialisesFromStore(unittest.TestCase):
    """The SDK's own resume path reads our store (no ~/.claude involved)."""

    def test_materialize_resume_session(self):
        try:
            import claude_agent_sdk as sdk
            from claude_agent_sdk._internal.session_resume import (
                materialize_resume_session,
            )
        except ImportError:
            self.skipTest("claude-agent-sdk not installed")
        with TemporaryDirectory() as tmp:
            s = FolderSessionStore(Path(tmp) / "claude")
            run(s.append({"project_key": "x", "session_id": SID},
                         [{"type": "user", "uuid": "a",
                           "message": {"role": "user", "content": "hi"}}]))
            opts = sdk.ClaudeAgentOptions(resume=SID, session_store=s,
                                          cwd=tmp)
            m = run(materialize_resume_session(opts))
            self.assertIsNotNone(m)
            files = list(Path(m.config_dir).rglob(f"{SID}.jsonl"))
            self.assertEqual(len(files), 1)
            m.cleanup()


if __name__ == "__main__":
    unittest.main()
