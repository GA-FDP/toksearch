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

"""End-to-end integration tests for toksearch.llm against real provider APIs.

These tests are skipped by default; they only run when:

  1. ``TOKSEARCH_INTEGRATION`` is set to ``"yes"`` in the environment.
     ``tests/testit.py`` sets this for you unless you pass ``--mock``.
  2. The relevant API key env var is set (``ANTHROPIC_API_KEY`` for the
     Anthropic backend, ``OPENAI_API_KEY`` for the OpenAI backend).

Each backend gets one trivial prompt that exercises the full loop:
``run_python`` tool call, namespace persistence, end_turn.  Failure here means
the wire format / tool-result encoding diverged from what the provider
actually expects — something the mocked unit tests cannot catch.

Cost note: each test does one short prompt + one tool round-trip, ~a few
hundred tokens.  Cheap, but not free.
"""

import os
import unittest

from toksearch.llm import Session
from toksearch.llm.events import TurnComplete


_INTEGRATION_ON = os.environ.get("TOKSEARCH_INTEGRATION") == "yes"
_HAS_ANTHROPIC = bool(os.environ.get("ANTHROPIC_API_KEY"))
_HAS_OPENAI = bool(os.environ.get("OPENAI_API_KEY"))


PROMPT = (
    "Use the run_python tool to compute 2 + 2 and print the result. "
    "Then report the answer in your final text response."
)


def _assert_simple_arithmetic(tc: TurnComplete, sess: Session):
    """Common assertions for the trivial '2 + 2' prompt across backends."""
    assert tc.stop_reason == "end_turn", f"unexpected stop_reason: {tc.stop_reason}"
    # The agent should have called run_python at least once.
    tool_uses = [
        b for m in sess.history if m.role == "assistant"
        for b in m.content if getattr(b, "kind", None) == "tool_use"
    ]
    assert any(b.name == "run_python" for b in tool_uses), (
        f"no run_python call in history; got tool uses: "
        f"{[b.name for b in tool_uses]}")
    # The answer "4" should appear in the final text.
    assert "4" in tc.final_text, (
        f"final_text did not mention 4: {tc.final_text!r}")


@unittest.skipUnless(_INTEGRATION_ON and _HAS_ANTHROPIC,
                     "TOKSEARCH_INTEGRATION=yes and ANTHROPIC_API_KEY required")
class TestAnthropicIntegration(unittest.TestCase):
    def test_simple_arithmetic_end_to_end(self):
        from toksearch.llm.backends.anthropic import AnthropicBackend
        backend = AnthropicBackend(api_key=os.environ["ANTHROPIC_API_KEY"])
        sess = Session(backend=backend, max_iterations=5)
        result = sess.send(PROMPT)
        _assert_simple_arithmetic(result, sess)


@unittest.skipUnless(_INTEGRATION_ON and _HAS_OPENAI,
                     "TOKSEARCH_INTEGRATION=yes and OPENAI_API_KEY required")
class TestOpenAIIntegration(unittest.TestCase):
    def test_simple_arithmetic_end_to_end(self):
        from toksearch.llm.backends.openai import OpenAIBackend
        backend = OpenAIBackend(api_key=os.environ["OPENAI_API_KEY"])
        sess = Session(backend=backend, max_iterations=5)
        result = sess.send(PROMPT)
        _assert_simple_arithmetic(result, sess)


_HAS_CLAUDE_MAX = os.environ.get("TOKSEARCH_CLAUDE_MAX") == "yes"


@unittest.skipUnless(_INTEGRATION_ON and _HAS_CLAUDE_MAX,
                     "TOKSEARCH_INTEGRATION=yes and TOKSEARCH_CLAUDE_MAX=yes "
                     "required; the `claude` CLI must be installed and "
                     "logged in.")
class TestClaudeMaxIntegration(unittest.TestCase):
    def test_simple_arithmetic_end_to_end(self):
        from toksearch.llm.backends.claude_sdk import ClaudeSDKBackend
        backend = ClaudeSDKBackend()
        sess = Session(backend=backend, max_iterations=5)
        result = sess.send(PROMPT)
        _assert_simple_arithmetic(result, sess)


if __name__ == "__main__":
    unittest.main()
