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
"""Tests for KernelExecutor and Session's executor seam."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from toksearch.llm.backends.base import AssistantTurn
from toksearch.llm.backends.fake import FakeBackend
from toksearch.llm.kernel.client import ExecResult, KernelClient
from toksearch.llm.kernel.executor import KernelExecutor, format_result
from toksearch.llm.messages import TextBlock, ToolUseBlock
from toksearch.llm.session import Session


class TestFormatResult(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(format_result(ExecResult()), "(no output)")
        self.assertEqual(format_result(ExecResult(stdout="1\n", stderr="w\n")),
                         "stdout:\n1\n\nstderr:\nw\n")

    def test_figures(self):
        r = ExecResult(figures=[
            {"kind": "plotly", "png_path": "/f/c1-1.png",
             "html_path": "/f/c1-1.html", "png_error": None},
            {"kind": "matplotlib", "error": "ValueError: bad"}])
        text = format_result(r)
        self.assertIn("Figure 1 saved: /f/c1-1.png, /f/c1-1.html", text)
        self.assertIn("Figure 2 could not be saved: ValueError: bad", text)

    def test_restart_with_restore(self):
        r = ExecResult(restarted=True, is_error=True, restore={
            "found": True, "restored": ["a"],
            "not_restored": [{"name": "df", "reason": "too_large",
                              "cell": 7}]})
        text = format_result(r)
        self.assertIn("The Python worker was restarted.", text)
        self.assertIn("Variables restored from the last checkpoint: a.", text)
        self.assertIn("`df` (too large to save; bound by cell 7)", text)
        self.assertIn("Anything computed after that checkpoint is lost.", text)

    def test_restart_without_checkpoint(self):
        text = format_result(ExecResult(restarted=True))
        self.assertIn("all variables were lost", text)

    def test_first_load_after_open(self):
        r = ExecResult(restore={"found": True, "restored": ["x"],
                                "not_restored": []})
        self.assertIn("Variables loaded from the saved conversation: x.",
                      format_result(r))


class TestSessionWithKernel(unittest.TestCase):
    def test_run_python_goes_to_the_worker(self):
        cells = []
        with TemporaryDirectory() as tmp:
            client = KernelClient(packages=[])
            executor = KernelExecutor(client, Path(tmp) / "figures",
                                      on_cell=cells.append)
            backend = FakeBackend(scripted_turns=[
                AssistantTurn(blocks=[ToolUseBlock(
                    id="t1", name="run_python",
                    args={"code": "kx = 5\nprint(kx)", "thought": "set kx"})],
                    stop_reason="tool_use"),
                AssistantTurn(blocks=[TextBlock(text="done")],
                              stop_reason="end_turn")])
            sess = Session(backend=backend, executor=executor)
            self.addCleanup(sess.close)
            sess.send("go")
            self.assertNotIn("kx", sess.namespace)
            tool_result = sess.history[-2].content[0]
            self.assertIn("stdout:\n5", tool_result.output)
            self.assertEqual(len(cells), 1)
            self.assertEqual(cells[0]["cell"], 1)
            self.assertEqual(cells[0]["bound"], ["kx"])
            self.assertEqual(cells[0]["thought"], "set kx")
            self.assertEqual(executor.next_cell, 2)


if __name__ == "__main__":
    unittest.main()
