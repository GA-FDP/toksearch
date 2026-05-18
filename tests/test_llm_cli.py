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
"""Tests for the toksearch chat / toksearch query CLI.

The Session class is mocked so the CLI tests verify wiring (subcommand
dispatch, flag plumbing, slash-command handling) without going near a real
backend.
"""

import io
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock


class TestCliQuery(unittest.TestCase):
    def _run_cli(self, argv, send_returns=None, stdin_text=""):
        from toksearch.llm import cli
        fake_session = mock.MagicMock()
        fake_session.send.return_value = send_returns or mock.MagicMock(
            stop_reason="end_turn", final_text="ok")
        buf = io.StringIO()
        exit_code = None
        with mock.patch.object(sys, "argv", argv), \
                mock.patch.object(cli, "build_session",
                                  return_value=fake_session) as build, \
                mock.patch.object(sys, "stdin",
                                  new=io.StringIO(stdin_text)), \
                redirect_stdout(buf):
            try:
                cli.main()
            except SystemExit as e:
                exit_code = e.code
        return fake_session, build, buf.getvalue(), exit_code

    def test_query_dispatches_send(self):
        sess, _, _, _ = self._run_cli(
            ["toksearch", "query", "hello"])
        sess.send.assert_called_once()
        prompt = sess.send.call_args.args[0]
        self.assertEqual(prompt, "hello")

    def test_query_backend_flag_forwarded_to_build(self):
        _, build, _, _ = self._run_cli(
            ["toksearch", "query", "--backend", "openai", "hi"])
        # build_session called with args namespace; check the attribute
        ns = build.call_args.args[0]
        self.assertEqual(ns.backend, "openai")

    def test_query_max_iterations_flag(self):
        _, build, _, _ = self._run_cli(
            ["toksearch", "query", "-n", "3", "hi"])
        ns = build.call_args.args[0]
        self.assertEqual(ns.max_iterations, 3)


class TestCliChatSlashCommands(unittest.TestCase):
    def _run_chat(self, stdin_text):
        from toksearch.llm import cli
        fake_session = mock.MagicMock()
        fake_session.send.return_value = mock.MagicMock(
            stop_reason="end_turn", final_text="agent says hi")
        buf = io.StringIO()
        exit_code = None
        with mock.patch.object(sys, "argv", ["toksearch", "chat"]), \
                mock.patch.object(cli, "build_session",
                                  return_value=fake_session), \
                mock.patch.object(sys, "stdin",
                                  new=io.StringIO(stdin_text)), \
                redirect_stdout(buf):
            try:
                cli.main()
            except SystemExit as e:
                exit_code = e.code
        return fake_session, buf.getvalue(), exit_code

    def test_chat_sends_each_nonempty_line(self):
        sess, out, _ = self._run_chat("hello\nhow are you?\n")
        self.assertEqual(sess.send.call_count, 2)

    def test_slash_quit_exits_loop(self):
        sess, out, _ = self._run_chat("hello\n/quit\nignored\n")
        self.assertEqual(sess.send.call_count, 1)

    def test_slash_reset_calls_session_reset(self):
        sess, out, _ = self._run_chat("/reset\n")
        sess.reset.assert_called_once()

    def test_slash_help_prints(self):
        sess, out, _ = self._run_chat("/help\n")
        self.assertIn("/help", out)
        self.assertIn("/reset", out)
        self.assertIn("/quit", out)

    def test_eof_exits_cleanly(self):
        # Empty stdin = immediate EOF
        sess, out, exit_code = self._run_chat("")
        self.assertIn(exit_code, (None, 0))


if __name__ == "__main__":
    unittest.main()
