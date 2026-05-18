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
"""Tests for OpenAIBackend.

The openai SDK is mocked at the module-attribute level so tests don't need
the SDK installed and don't make network calls.
"""

import json
import sys
import types
import unittest
from unittest import mock


# Stub openai module + its error types
_stub_openai = types.ModuleType("openai")
_stub_openai.OpenAI = mock.MagicMock
_stub_openai.AuthenticationError = type("AuthenticationError", (Exception,), {})
_stub_openai.RateLimitError = type("RateLimitError", (Exception,), {})
_stub_openai.APIConnectionError = type("APIConnectionError", (Exception,), {})
_stub_openai.APIStatusError = type("APIStatusError", (Exception,), {})
sys.modules.setdefault("openai", _stub_openai)

from toksearch.llm.backends.openai import OpenAIBackend  # noqa: E402
from toksearch.llm.errors import LLMAuthError  # noqa: E402
from toksearch.llm.messages import TextBlock, ToolUseBlock  # noqa: E402
from toksearch.llm.tools import ToolSpec, ToolOutput  # noqa: E402


def _make_chat_response(text=None, tool_calls=None, finish_reason="stop"):
    """Build a fake openai chat completion response."""
    msg = mock.MagicMock()
    msg.content = text
    msg.tool_calls = []
    for tc in (tool_calls or []):
        m = mock.MagicMock()
        m.id = tc["id"]
        m.type = "function"
        m.function.name = tc["name"]
        m.function.arguments = json.dumps(tc["args"])
        msg.tool_calls.append(m)
    choice = mock.MagicMock()
    choice.message = msg
    choice.finish_reason = finish_reason
    resp = mock.MagicMock()
    resp.choices = [choice]
    return resp


class TestOpenAISendRequest(unittest.TestCase):
    def setUp(self):
        self.backend = OpenAIBackend(api_key="sk-test")
        self.backend._client = mock.MagicMock()

    def test_text_response_becomes_assistant_turn(self):
        self.backend._client.chat.completions.create.return_value = (
            _make_chat_response(text="hello", finish_reason="stop"))
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="gpt-4o")
        self.assertEqual(turn.stop_reason, "end_turn")
        self.assertEqual(len(turn.blocks), 1)
        self.assertIsInstance(turn.blocks[0], TextBlock)
        self.assertEqual(turn.blocks[0].text, "hello")

    def test_tool_call_response(self):
        self.backend._client.chat.completions.create.return_value = (
            _make_chat_response(text=None,
                                tool_calls=[{"id": "t1", "name": "run_python",
                                             "args": {"code": "1+1",
                                                      "thought": "x"}}],
                                finish_reason="tool_calls"))
        turn = self.backend._send_request(
            system_prompt="sys", history=[], tools=[], model="gpt-4o")
        self.assertEqual(turn.stop_reason, "tool_use")
        # Should produce one ToolUseBlock (no TextBlock since content is None)
        tool_blocks = [b for b in turn.blocks if isinstance(b, ToolUseBlock)]
        self.assertEqual(len(tool_blocks), 1)
        self.assertEqual(tool_blocks[0].args,
                         {"code": "1+1", "thought": "x"})

    def test_tools_translated_to_openai_schema(self):
        spec = ToolSpec(
            name="echo",
            description="echo",
            input_schema={"type": "object",
                          "properties": {"x": {"type": "string"}}},
            handler=lambda a, s: ToolOutput(text="x"),
        )
        self.backend._client.chat.completions.create.return_value = (
            _make_chat_response(text="", finish_reason="stop"))
        self.backend._send_request(
            system_prompt="sys", history=[], tools=[spec], model="gpt-4o")
        kwargs = self.backend._client.chat.completions.create.call_args.kwargs
        sent_tools = kwargs["tools"]
        self.assertEqual(sent_tools[0]["type"], "function")
        self.assertEqual(sent_tools[0]["function"]["name"], "echo")
        self.assertEqual(sent_tools[0]["function"]["parameters"],
                         {"type": "object",
                          "properties": {"x": {"type": "string"}}})

    def test_auth_error_translated_to_llm_auth_error(self):
        import openai as _openai
        fake_auth = type("FakeAuth", (Exception,), {})
        with mock.patch.object(_openai, "AuthenticationError", fake_auth):
            self.backend._client.chat.completions.create.side_effect = (
                fake_auth("401"))
            with self.assertRaises(LLMAuthError):
                self.backend._send_request(system_prompt="s", history=[],
                                            tools=[], model="m")


if __name__ == "__main__":
    unittest.main()
