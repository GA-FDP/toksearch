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
"""Session-level tests using FakeBackend.

The Session class is responsible for: building the system prompt, owning the
namespace and history, registering tools, dispatching to the backend, and
providing reset/introspection.
"""

import unittest

from toksearch.llm.backends.base import AssistantTurn
from toksearch.llm.backends.fake import FakeBackend
from toksearch.llm.messages import TextBlock, ToolUseBlock
from toksearch.llm.session import Session


def _text(s):
    return AssistantTurn(blocks=[TextBlock(text=s)], stop_reason="end_turn")


def _tool_use(name, args, id_="t1"):
    return AssistantTurn(
        blocks=[ToolUseBlock(id=id_, name=name, args=args)],
        stop_reason="tool_use",
    )


class TestSessionBasics(unittest.TestCase):
    def test_send_returns_turn_complete(self):
        backend = FakeBackend(scripted_turns=[_text("hello")])
        sess = Session(backend=backend)
        out = sess.send("hi")
        self.assertEqual(out.stop_reason, "end_turn")
        self.assertEqual(out.final_text, "hello")

    def test_namespace_pre_populated(self):
        backend = FakeBackend(scripted_turns=[_text("ok")])
        sess = Session(backend=backend)
        # toksearch and numpy are required; pandas and matplotlib are optional
        # in the test env (they live in the [llm] extra).
        self.assertIn("toksearch", sess.namespace)
        self.assertIn("np", sess.namespace)

    def test_extra_namespace_merged(self):
        backend = FakeBackend(scripted_turns=[_text("ok")])
        sess = Session(backend=backend, extra_namespace={"answer": 42})
        self.assertEqual(sess.namespace["answer"], 42)


class TestSessionPersistence(unittest.TestCase):
    def test_namespace_persists_across_send_calls(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "x = 99", "thought": "set"}),
            _text("set"),
            _tool_use("run_python", {"code": "print(x)", "thought": "read"}, id_="t2"),
            _text("read"),
        ])
        sess = Session(backend=backend)
        sess.send("set x")
        sess.send("read x")
        self.assertEqual(sess.namespace["x"], 99)

    def test_reset_clears_namespace_and_history(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "x = 1", "thought": "set"}),
            _text("done"),
        ])
        sess = Session(backend=backend)
        sess.send("set x")
        self.assertIn("x", sess.namespace)
        self.assertGreater(len(sess.history), 0)
        sess.reset()
        self.assertNotIn("x", sess.namespace)
        self.assertEqual(len(sess.history), 0)
        # Standard names are still there
        self.assertIn("toksearch", sess.namespace)


class TestSessionCallbacks(unittest.TestCase):
    def test_on_tool_call_fires_before_result(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "print('hi')", "thought": "x"}),
            _text("done"),
        ])
        sess = Session(backend=backend)
        order = []
        sess.send("go",
                  on_tool_call=lambda c: order.append(("call", c.name)),
                  on_tool_result=lambda r: order.append(("result", r.output)))
        self.assertEqual(order[0][0], "call")
        self.assertEqual(order[1][0], "result")
        self.assertIn("hi", order[1][1])

    def test_confirm_false_aborts(self):
        backend = FakeBackend(scripted_turns=[
            _tool_use("run_python", {"code": "print('x')", "thought": "x"}),
        ])
        sess = Session(backend=backend)
        out = sess.send("go", confirm=lambda call: False)
        self.assertEqual(out.stop_reason, "interrupted")


class TestSessionTools(unittest.TestCase):
    def test_run_python_and_lookup_docs_registered(self):
        backend = FakeBackend(scripted_turns=[_text("ok")])
        sess = Session(backend=backend)
        names = {t.name for t in sess.tool_specs}
        self.assertEqual(names, {"run_python", "lookup_docs"})


if __name__ == "__main__":
    unittest.main()
