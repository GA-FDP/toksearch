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
"""Tests for AnthropicBackend.

The anthropic SDK is mocked at the module-attribute level (``backend._client``)
so tests don't need the SDK installed and don't make network calls.
"""

import sys
import types
import unittest
from unittest import mock


# Install a stub `anthropic` module before any imports of the backend so the
# `import anthropic` inside backend.__init__ resolves to our stub.
_stub_anthropic = types.ModuleType("anthropic")
_stub_anthropic.Anthropic = mock.MagicMock
_stub_anthropic.AuthenticationError = type("AuthenticationError", (Exception,), {})
_stub_anthropic.RateLimitError = type("RateLimitError", (Exception,), {})
_stub_anthropic.APIStatusError = type("APIStatusError", (Exception,), {})
_stub_anthropic.APIConnectionError = type("APIConnectionError", (Exception,), {})
sys.modules.setdefault("anthropic", _stub_anthropic)

from toksearch.llm.backends.anthropic import AnthropicBackend  # noqa: E402
from toksearch.llm.backends.base import AssistantTurn  # noqa: E402
from toksearch.llm.errors import LLMAuthError  # noqa: E402
from toksearch.llm.messages import TextBlock, ToolUseBlock  # noqa: E402
from toksearch.llm.tools import ToolSpec, ToolOutput  # noqa: E402


def _make_response(text=None, tool_use=None, stop_reason="end_turn"):
    """Build a fake anthropic.types.Message-shaped response."""
    blocks = []
    if text is not None:
        blocks.append(mock.MagicMock(type="text", text=text))
    if tool_use is not None:
        b = mock.MagicMock(type="tool_use")
        b.id = tool_use["id"]
        b.name = tool_use["name"]
        b.input = tool_use["input"]
        blocks.append(b)
    resp = mock.MagicMock()
    resp.content = blocks
    resp.stop_reason = stop_reason
    return resp


class TestAnthropicSendRequest(unittest.TestCase):
    def setUp(self):
        self.backend = AnthropicBackend(api_key="sk-test")
        self.backend._client = mock.MagicMock()

    def test_text_response_becomes_assistant_turn(self):
        self.backend._client.messages.create.return_value = _make_response(
            text="hi", stop_reason="end_turn")
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="claude-x")
        self.assertEqual(turn.stop_reason, "end_turn")
        self.assertEqual(len(turn.blocks), 1)
        self.assertIsInstance(turn.blocks[0], TextBlock)
        self.assertEqual(turn.blocks[0].text, "hi")

    def test_tool_use_response_becomes_tool_use_block(self):
        self.backend._client.messages.create.return_value = _make_response(
            tool_use={"id": "t1", "name": "run_python",
                      "input": {"code": "1+1", "thought": "x"}},
            stop_reason="tool_use")
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="claude-x")
        self.assertEqual(turn.stop_reason, "tool_use")
        self.assertEqual(len(turn.blocks), 1)
        b = turn.blocks[0]
        self.assertIsInstance(b, ToolUseBlock)
        self.assertEqual(b.id, "t1")
        self.assertEqual(b.name, "run_python")
        self.assertEqual(b.args, {"code": "1+1", "thought": "x"})

    def test_tools_translated_to_anthropic_schema(self):
        spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object",
                          "properties": {"x": {"type": "string"}}},
            handler=lambda a, s: ToolOutput(text="x"),
        )
        self.backend._client.messages.create.return_value = _make_response(
            text="", stop_reason="end_turn")
        self.backend._send_request(
            system_prompt="sys", history=[], tools=[spec], model="claude-x")
        kwargs = self.backend._client.messages.create.call_args.kwargs
        sent_tools = kwargs["tools"]
        self.assertEqual(sent_tools[0]["name"], "echo")
        self.assertEqual(sent_tools[0]["description"], "echo")
        self.assertEqual(sent_tools[0]["input_schema"],
                         {"type": "object",
                          "properties": {"x": {"type": "string"}}})


class TestAnthropicAuthError(unittest.TestCase):
    def test_auth_error_translated_to_llm_auth_error(self):
        # Build a fresh exception type and patch it onto the anthropic module
        # so the backend's `except anthropic.AuthenticationError` matches it
        # regardless of whether the real SDK is installed.
        import anthropic as _anthropic
        fake_auth = type("FakeAuth", (Exception,), {})
        backend = AnthropicBackend(api_key="x")
        backend._client = mock.MagicMock()
        with mock.patch.object(_anthropic, "AuthenticationError", fake_auth):
            backend._client.messages.create.side_effect = fake_auth("401")
            with self.assertRaises(LLMAuthError):
                backend._send_request(system_prompt="s", history=[],
                                       tools=[], model="m")


class TestAnthropicBaseUrl(unittest.TestCase):
    def test_base_url_passed_to_client(self):
        with mock.patch.object(_stub_anthropic, "Anthropic") as ctor:
            AnthropicBackend(api_key="x",
                              base_url="https://custom.example")._build_client()
            self.assertEqual(
                ctor.call_args.kwargs["base_url"],
                "https://custom.example")


if __name__ == "__main__":
    unittest.main()
