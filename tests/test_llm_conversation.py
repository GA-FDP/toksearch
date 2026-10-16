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

"""Tests for Conversation: logging, checkpoints, and restore after a stop.

FakeBackend drives the turns; the kernel is the real spawned worker, so
"restore" is tested across processes. packages=[] keeps worker start-up fast.
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from toksearch.llm.backends.base import AssistantTurn
from toksearch.llm.backends.fake import FakeBackend
from toksearch.llm.conversation import Conversation, list_conversations
from toksearch.llm.errors import LLMResumeError
from toksearch.llm.kernel.client import KernelClient, KernelError
from toksearch.llm.messages import TextBlock, ToolUseBlock
from toksearch.llm.session import Session


def _text(s):
    return AssistantTurn(blocks=[TextBlock(text=s)], stop_reason="end_turn")


def _py(code, id_="t1"):
    return AssistantTurn(blocks=[ToolUseBlock(
        id=id_, name="run_python", args={"code": code, "thought": "t"})],
        stop_reason="tool_use")


def _session(turns, record=None, backend_cls=FakeBackend):
    return Session(backend=backend_cls(scripted_turns=turns, record=record),
                   packages=[])


class _Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestCreateAndSend(_Tmp):
    def test_create(self):
        sess = _session([])
        conv = Conversation.create(sess, self.root, packages=[])
        self.addCleanup(conv.close)
        meta = conv.folder.read_meta()
        self.assertEqual(meta["backend"], "fake")
        self.assertEqual(meta["model"], "fake-1")
        self.assertIs(sess.executor, conv.executor)
        self.assertEqual(conv.events.read(), [])

    def test_send_logs_saves_and_checkpoints(self):
        sess = _session([_py("x = 41"), _text("done")])
        conv = Conversation.create(sess, self.root, packages=[])
        self.addCleanup(conv.close)
        conv.send("set x please")
        types_ = [e["type"] for e in conv.events.read()]
        self.assertEqual(types_, ["user", "tool_call", "cell", "tool_result",
                                  "assistant_text", "turn_end", "checkpoint"])
        meta = conv.folder.read_meta()
        self.assertEqual(meta["title"], "set x please")
        self.assertTrue(conv.folder.history_path.exists())
        index = json.loads((conv.folder.namespace_dir
                            / "index.json").read_text())
        self.assertEqual([(e["name"], e["cell"]) for e in index["variables"]],
                         [("x", 1)])

    def test_title_is_truncated(self):
        sess = _session([_text("ok")])
        conv = Conversation.create(sess, self.root, packages=[])
        self.addCleanup(conv.close)
        conv.send("q" * 200)
        self.assertEqual(len(conv.folder.read_meta()["title"]), 80)


class TestRestore(_Tmp):
    def _first(self):
        sess = _session([_py("x = 41"), _text("done")])
        conv = Conversation.create(sess, self.root, packages=[])
        conv.send("set x")
        conv.close()
        sess.close()
        return conv.folder.path

    def test_open_restores_history_note_and_variables(self):
        path = self._first()
        record = []
        sess = _session([_py("print(x + 1)", id_="t2"), _text("42")],
                        record=record)
        conv = Conversation.open(sess, path, packages=[])
        self.addCleanup(conv.close)
        self.assertEqual(len(sess.history), 4)
        self.assertIn("1 variable restored", conv.restore_line)
        conv.send("what is x+1?")
        prompt = record[0]["user_message"]
        self.assertTrue(prompt.startswith("[Note from the app"))
        self.assertIn("Python variables available: x.", prompt)
        self.assertTrue(prompt.endswith("what is x+1?"))
        self.assertNotIn("Summary of the earlier", prompt)
        cells = [e for e in conv.events.read() if e["type"] == "cell"]
        self.assertEqual(cells[-1]["cell"], 2)
        self.assertEqual(cells[-1]["stdout"], "42\n")
        users = [e["text"] for e in conv.events.read() if e["type"] == "user"]
        self.assertEqual(users, ["set x", "what is x+1?"])

    def test_note_is_sent_once(self):
        path = self._first()
        record = []
        sess = _session([_text("a"), _text("b")], record=record)
        conv = Conversation.open(sess, path, packages=[])
        self.addCleanup(conv.close)
        conv.send("one")
        conv.send("two")
        self.assertEqual(record[1]["user_message"], "two")

    def test_missing_history_uses_recap(self):
        path = self._first()
        (path / "history.jsonl").unlink()
        record = []
        sess = _session([_text("ok")], record=record)
        conv = Conversation.open(sess, path, packages=[])
        self.addCleanup(conv.close)
        self.assertEqual(sess.history, [])
        conv.send("again")
        prompt = record[0]["user_message"]
        self.assertIn("Summary of the earlier conversation", prompt)
        self.assertIn("User: set x", prompt)
        self.assertIn("rebuilt from a summary", conv.restore_line)

    def test_changed_backend_uses_recap(self):
        path = self._first()
        meta = json.loads((path / "conversation.json").read_text())
        meta["backend"] = "anthropic"
        (path / "conversation.json").write_text(json.dumps(meta))
        sess = _session([])
        conv = Conversation.open(sess, path, packages=[])
        self.addCleanup(conv.close)
        self.assertEqual(sess.history, [])
        self.assertIn("rebuilt from a summary", conv.restore_line)

    def test_environment_change_is_reported(self):
        with mock.patch.dict(os.environ, {"FDP_CORE_VERSION": "1.34.0"}):
            path = self._first()
        with mock.patch.dict(os.environ, {"FDP_CORE_VERSION": "1.35.0"}):
            conv = Conversation.open(_session([]), path, packages=[])
        self.addCleanup(conv.close)
        self.assertIn("fdp-core 1.34.0 → 1.35.0", conv.restore_line)

    def test_interrupted_turn_is_closed_on_open(self):
        path = self._first()
        with open(path / "events.jsonl", "a") as f:
            f.write(json.dumps({"type": "user", "text": "cut off"}) + "\n")
        conv = Conversation.open(_session([]), path, packages=[])
        self.addCleanup(conv.close)
        events = conv.events.read()
        self.assertEqual(events[-2], {**events[-2], "type": "turn_end",
                                      "status": "interrupted"})
        self.assertEqual(events[-1]["type"], "restore")


def _texts(history):
    return [b.text for m in history for b in m.content
            if isinstance(b, TextBlock)]


class TestReusedSession(_Tmp):
    """One Session serves several conversations in turn (the chat app)."""

    def _a_then_b(self, record):
        sess = _session([_text("a1"), _text("b1"), _text("a2")],
                        record=record)
        self.addCleanup(sess.close)
        a = Conversation.create(sess, self.root, packages=[])
        a.send("alpha")
        a.close()
        b = Conversation.create(sess, self.root, packages=[])
        b.send("bravo")
        b.close()
        return sess, a.folder.path

    def test_new_conversation_starts_with_empty_history(self):
        record = []
        self._a_then_b(record)
        self.assertEqual(_texts(record[1]["history"]), ["bravo"])

    def test_reopening_after_another_conversation_has_none_of_it(self):
        record = []
        sess, a_path = self._a_then_b(record)
        (a_path / "history.jsonl").unlink()     # forces the recap path
        a2 = Conversation.open(sess, a_path, packages=[])
        self.addCleanup(a2.close)
        a2.send("again")
        texts = _texts(record[2]["history"])
        self.assertFalse(any("bravo" in t or "b1" == t for t in texts), texts)


class _ResumeFailingBackend(FakeBackend):
    """Stands in for claude-max: persistence hooks, first send fails to resume."""

    name = "claude-max"

    def __init__(self, scripted_turns=None, record=None):
        super().__init__(scripted_turns, record)
        self.configured = []
        self.session_id = None
        self.fail_next = False

    def configure_persistence(self, *, store, cwd, session_id=None,
                              resume=None):
        self.configured.append({"session_id": session_id, "resume": resume})
        self.session_id = session_id or resume
        self.fail_next = resume is not None

    def close(self):
        self.session_id = None

    def run_conversation(self, session, new_user_message, callbacks,
                         max_iterations):
        if self.fail_next:
            self.fail_next = False
            session._append_user(new_user_message)
            raise LLMResumeError("gone")
        return super().run_conversation(session, new_user_message, callbacks,
                                        max_iterations)


class TestClaudeStylePersistence(_Tmp):
    def test_new_conversation_records_the_sdk_session_id(self):
        sess = _session([_text("hi")], backend_cls=_ResumeFailingBackend)
        conv = Conversation.create(sess, self.root, packages=[])
        self.addCleanup(conv.close)
        sid = sess.backend.configured[0]["session_id"]
        self.assertEqual(conv.folder.read_meta()["claude_session_id"], sid)

    def test_resume_is_tried_when_a_transcript_exists(self):
        sess = _session([_text("hi")], backend_cls=_ResumeFailingBackend)
        conv = Conversation.create(sess, self.root, packages=[])
        sid = conv.folder.read_meta()["claude_session_id"]
        conv.folder.claude_dir.mkdir()
        (conv.folder.claude_dir / f"{sid}.jsonl").write_text('{"type":"x"}\n')
        conv.close()
        sess2 = _session([], backend_cls=_ResumeFailingBackend)
        conv2 = Conversation.open(sess2, conv.folder.path, packages=[])
        self.addCleanup(conv2.close)
        self.assertEqual(sess2.backend.configured[-1]["resume"], sid)
        self.assertNotIn("summary", conv2.restore_line)

    def test_no_transcript_starts_fresh_with_recap(self):
        sess = _session([_text("hi")], backend_cls=_ResumeFailingBackend)
        conv = Conversation.create(sess, self.root, packages=[])
        conv.send("first")
        conv.close()
        sess2 = _session([], backend_cls=_ResumeFailingBackend)
        conv2 = Conversation.open(sess2, conv.folder.path, packages=[])
        self.addCleanup(conv2.close)
        self.assertIsNotNone(sess2.backend.configured[-1]["session_id"])
        self.assertIn("rebuilt from a summary", conv2.restore_line)

    def test_resume_failure_falls_back_once_without_duplicating(self):
        sess = _session([_text("hi")], backend_cls=_ResumeFailingBackend)
        conv = Conversation.create(sess, self.root, packages=[])
        conv.send("first")
        sid = conv.folder.read_meta()["claude_session_id"]
        conv.folder.claude_dir.mkdir()
        (conv.folder.claude_dir / f"{sid}.jsonl").write_text('{"type":"x"}\n')
        conv.close()
        record = []
        sess2 = _session([_text("back")], record=record,
                         backend_cls=_ResumeFailingBackend)
        conv2 = Conversation.open(sess2, conv.folder.path, packages=[])
        self.addCleanup(conv2.close)
        out = conv2.send("second")
        self.assertEqual(out.final_text, "back")
        self.assertIn("User: first", record[0]["user_message"])
        self.assertTrue(record[0]["user_message"].endswith("second"))
        user_msgs = [m for m in sess2.history if m.role == "user"]
        self.assertEqual(sum("second" in m.content[0].text
                             for m in user_msgs), 1)
        new_sid = sess2.backend.configured[-1]["session_id"]
        self.assertNotEqual(new_sid, sid)
        self.assertEqual(conv2.folder.read_meta()["claude_session_id"],
                         new_sid)
        restores = [e for e in conv2.events.read() if e["type"] == "restore"]
        self.assertTrue(restores[-1]["recap_used"])


class TestClose(_Tmp):
    def _conv_with(self, client):
        return Conversation.create(_session([]), self.root, packages=[],
                                   client_factory=lambda packages,
                                   checkpoint_dir: client)

    def test_close_stops_the_cell_then_checkpoints_with_a_bound(self):
        client = mock.create_autospec(KernelClient, instance=True)
        client.checkpoint.return_value = None
        conv = self._conv_with(client)
        conv.close()
        names = [c[0] for c in client.mock_calls]
        self.assertEqual(names[:3], ["begin_close", "interrupt", "checkpoint"])
        self.assertEqual(client.checkpoint.call_args.kwargs, {"wait": 30})
        client.close.assert_called_once_with()

    def test_close_during_a_stuck_cell_is_bounded_and_starts_no_worker(self):
        import threading
        import time
        clients = []

        def factory(packages, checkpoint_dir):
            c = KernelClient(packages, checkpoint_dir=checkpoint_dir, grace=2)
            clients.append(c)
            return c

        conv = Conversation.create(_session([]), self.root, packages=[],
                                   client_factory=factory)
        client = clients[0]
        spawns = []
        real_spawn = client._spawn
        client._spawn = lambda: (spawns.append(1), real_spawn())
        started = self.root / "started"
        results = []
        t = threading.Thread(target=lambda: results.append(conv.executor.run(
            "import signal, time\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            f"open({str(started)!r}, 'w').close()\n"
            "time.sleep(120)")))
        t.start()
        deadline = time.monotonic() + 60
        while not started.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(started.exists(), "the cell never started")
        t0 = time.monotonic()
        conv.close()
        elapsed = time.monotonic() - t0
        t.join(30)
        self.assertLess(elapsed, 2 + 15)
        self.assertEqual(spawns, [1])        # only the first start
        self.assertFalse(client.running)
        self.assertTrue(results[0].interrupted)

    def test_busy_checkpoint_on_close_is_logged_not_raised(self):
        client = mock.create_autospec(KernelClient, instance=True)
        client.checkpoint.side_effect = KernelError("busy: a cell is running")
        conv = self._conv_with(client)
        conv.close()
        errors = [e["text"] for e in conv.events.read()
                  if e["type"] == "error"]
        self.assertEqual(errors,
                         ["Checkpoint failed: busy: a cell is running"])
        client.close.assert_called_once_with()


class TestList(_Tmp):
    def test_list_conversations_is_reexported(self):
        conv = Conversation.create(_session([]), self.root, packages=[])
        conv.close()
        self.assertEqual(len(list_conversations(self.root)), 1)


if __name__ == "__main__":
    unittest.main()
