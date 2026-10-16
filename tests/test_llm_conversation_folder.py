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
    ConversationFolder, ConversationInUse, default_root, list_conversations,
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


class TestEventLogConcurrency(unittest.TestCase):
    def test_threads_appending_large_lines_never_interleave(self):
        import threading
        with TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.jsonl")
            payload = "x" * 20_000

            def worker(n):
                for i in range(200):
                    log.append({"type": "cell", "w": n, "i": i,
                                "stdout": payload})
            threads = [threading.Thread(target=worker, args=(n,))
                       for n in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            lines = log.path.read_text().splitlines()
            self.assertEqual(len(lines), 8 * 200)
            for line in lines:
                json.loads(line)        # every line parses
            self.assertEqual(len(log.read()), 8 * 200)

    def test_each_event_is_one_write_of_a_whole_line(self):
        import toksearch.llm.conversation.events as events_mod
        real_write = os.write
        writes = []

        def spy(fd, data):
            writes.append(bytes(data))
            return real_write(fd, data)
        with TemporaryDirectory() as tmp:
            log = EventLog(Path(tmp) / "events.jsonl")
            with mock.patch.object(events_mod.os, "write", side_effect=spy):
                log.append({"type": "user", "text": "hi"})
                log.append({"type": "cell", "stdout": "y" * 50_000})
            self.assertEqual(len(writes), 2)
            for w in writes:
                self.assertTrue(w.endswith(b"\n"))
                json.loads(w)


def _dead_pid() -> int:
    import subprocess
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


class TestLock(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.folder = ConversationFolder.create(self._tmp.name)

    def _write_lock(self, **info):
        (self.folder.path / ".lock").write_text(json.dumps(info))

    def test_acquire_writes_owner_and_release_removes_it(self):
        import socket
        self.folder.acquire_lock()
        info = json.loads((self.folder.path / ".lock").read_text())
        self.assertEqual(info["pid"], os.getpid())
        self.assertEqual(info["host"], socket.gethostname())
        self.assertIn("since", info)
        self.folder.release_lock()
        self.assertFalse((self.folder.path / ".lock").exists())
        self.folder.release_lock()      # idempotent

    def test_live_lock_is_in_use(self):
        self.folder.acquire_lock()
        other = ConversationFolder(self.folder.path)
        with self.assertRaises(ConversationInUse) as cm:
            other.acquire_lock()
        self.assertIn(str(os.getpid()), str(cm.exception))
        self.assertIsInstance(cm.exception, RuntimeError)
        self.folder.release_lock()
        other.acquire_lock()
        other.release_lock()

    def test_dead_pid_on_this_host_is_taken_over(self):
        import socket
        self._write_lock(pid=_dead_pid(), host=socket.gethostname(),
                         since="2026-01-01T00:00:00")
        self.folder.acquire_lock()
        info = json.loads((self.folder.path / ".lock").read_text())
        self.assertEqual(info["pid"], os.getpid())
        self.folder.release_lock()

    def test_lock_from_another_host_is_in_use(self):
        self._write_lock(pid=_dead_pid(), host="elsewhere.example",
                         since="2026-01-01T00:00:00")
        with self.assertRaises(ConversationInUse) as cm:
            self.folder.acquire_lock()
        self.assertIn("elsewhere.example", str(cm.exception))

    def test_release_leaves_someone_elses_lock(self):
        self._write_lock(pid=1, host="elsewhere.example", since="x")
        self.folder.release_lock()
        self.assertTrue((self.folder.path / ".lock").exists())

    def test_lock_records_machine_and_start_time(self):
        from toksearch.llm.conversation import folder as folder_mod
        self.folder.acquire_lock()
        info = json.loads((self.folder.path / ".lock").read_text())
        self.folder.release_lock()
        self.assertEqual(info.get("machine"), folder_mod._machine_id())
        self.assertEqual(info.get("start"),
                         folder_mod._process_start(os.getpid()))
        self.assertIsNotNone(info.get("start"))     # Linux /proc

    def test_same_machine_other_hostname_dead_pid_is_taken_over(self):
        from toksearch.llm.conversation import folder as folder_mod
        machine = folder_mod._machine_id()
        if machine is None:
            self.skipTest("no hardware machine id here")
        self._write_lock(pid=_dead_pid(), host="renamed.example",
                         machine=machine, since="x")
        self.folder.acquire_lock()
        info = json.loads((self.folder.path / ".lock").read_text())
        self.assertEqual(info["pid"], os.getpid())
        self.folder.release_lock()

    def test_other_machine_same_hostname_is_in_use(self):
        import socket
        self._write_lock(pid=_dead_pid(), host=socket.gethostname(),
                         machine="0000000000aa", since="x")
        with self.assertRaises(ConversationInUse):
            self.folder.acquire_lock()

    def test_live_pid_with_another_start_time_is_taken_over(self):
        import socket
        from toksearch.llm.conversation import folder as folder_mod
        self._write_lock(pid=os.getpid(), host=socket.gethostname(),
                         machine=folder_mod._machine_id(),
                         start="not-this-process", since="x")
        self.folder.acquire_lock()
        info = json.loads((self.folder.path / ".lock").read_text())
        self.assertEqual(info["start"], folder_mod._process_start(os.getpid()))
        self.folder.release_lock()

    def test_in_use_carries_holder_details(self):
        self._write_lock(pid=4242, host="elsewhere.example",
                         machine="0000000000aa", since="2026-01-01T00:00:00")
        with self.assertRaises(ConversationInUse) as cm:
            self.folder.acquire_lock()
        e = cm.exception
        self.assertEqual((e.host, e.pid, e.since),
                         ("elsewhere.example", 4242, "2026-01-01T00:00:00"))

    def test_break_lock(self):
        self._write_lock(pid=4242, host="elsewhere.example", since="x")
        self.folder.break_lock()
        self.assertFalse((self.folder.path / ".lock").exists())
        self.folder.break_lock()        # nothing to break is fine
        self.folder.acquire_lock()
        self.folder.release_lock()

    def test_list_ignores_the_lock(self):
        before = list_conversations(self._tmp.name)[0]["size_bytes"]
        self.folder.acquire_lock()
        items = list_conversations(self._tmp.name)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["size_bytes"], before)
        self.folder.release_lock()


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
