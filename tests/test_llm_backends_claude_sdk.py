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
"""Tests for ClaudeSDKBackend.

The claude_agent_sdk module is real (conda-forge install), but we never
construct an actual ClaudeSDKClient -- tests inject mocks to verify the
backend wires options + MCP tools correctly and translates SDK messages
into our event types.

The test module gracefully skips itself when claude-agent-sdk is not
installed (e.g. the conda-recipe's minimal test env that omits the
``[llm]`` optional-dependency extra).
"""

import unittest

try:
    import asyncio
    import claude_agent_sdk  # noqa: F401
    from types import SimpleNamespace
    from unittest import mock

    from toksearch.llm.backends.base import Callbacks
    from toksearch.llm.backends.claude_sdk import ClaudeSDKBackend
    from toksearch.llm.messages import (
        Message, TextBlock, ToolUseBlock, ToolResultBlock,
    )
    from toksearch.llm.tools import ToolOutput
    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False


def _stub_session():
    """Minimal Session-shaped stub."""
    s = SimpleNamespace()
    s.history = []
    s.namespace = {}
    s.system_prompt = "sys"
    s.model = None
    s.tool_specs = []
    s.skills = {}
    s._executed = []
    def execute_tool(block):
        s._executed.append(block)
        if block.name == "run_python":
            return ToolOutput(text=f"executed: {block.args.get('code')}",
                              is_error=False)
        if block.name == "lookup_docs":
            return ToolOutput(text=f"docs for {block.args.get('skill_name')}",
                              is_error=False)
        return ToolOutput(text="unknown", is_error=True)
    s._append_user = lambda msg: s.history.append(
        Message(role="user", content=[TextBlock(text=msg)]))
    s._append_assistant = lambda blocks: s.history.append(
        Message(role="assistant", content=list(blocks)))
    s._append_tool_result = lambda tid, out, err: s.history.append(
        Message(role="user", content=[ToolResultBlock(
            tool_use_id=tid, output=out, is_error=err)]))
    s._execute_tool = execute_tool
    return s


@unittest.skipUnless(_SDK_AVAILABLE, "claude-agent-sdk not installed")
class TestMcpToolsRunPython(unittest.TestCase):
    def test_run_python_tool_proxies_to_session(self):
        backend = ClaudeSDKBackend()
        sess = _stub_session()
        calls, results = [], []
        backend._current_session = sess
        backend._current_callbacks = Callbacks(
            on_tool_call=calls.append,
            on_tool_result=results.append,
        )
        # _run_python_handler is the inner coroutine the MCP tool wraps.
        result = asyncio.run(backend._run_python_handler(
            {"code": "x = 1", "thought": "set x"}))
        # MCP tools return {"content": [{"type": "text", "text": ...}]}
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertIn("executed: x = 1", result["content"][0]["text"])
        # Callbacks fired in order:
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "run_python")
        self.assertEqual(calls[0].thought, "set x")
        self.assertEqual(len(results), 1)
        self.assertIn("executed: x = 1", results[0].output)
        # Tool result appended to session history
        self.assertEqual(len(sess.history), 1)
        block = sess.history[0].content[0]
        self.assertIsInstance(block, ToolResultBlock)

    def test_run_python_isError_propagates(self):
        backend = ClaudeSDKBackend()
        sess = _stub_session()
        sess._execute_tool = lambda b: ToolOutput(
            text="ZeroDivisionError", is_error=True)
        backend._current_session = sess
        backend._current_callbacks = Callbacks()
        result = asyncio.run(backend._run_python_handler(
            {"code": "1/0", "thought": "boom"}))
        # is_error reflected in the MCP response (text content marked)
        self.assertTrue(result.get("isError", False))


@unittest.skipUnless(_SDK_AVAILABLE, "claude-agent-sdk not installed")
class TestMcpToolsLookupDocs(unittest.TestCase):
    def test_lookup_docs_tool_proxies_to_session(self):
        backend = ClaudeSDKBackend()
        sess = _stub_session()
        backend._current_session = sess
        backend._current_callbacks = Callbacks()
        result = asyncio.run(backend._lookup_docs_handler(
            {"skill_name": "toksearch-pipeline"}))
        self.assertEqual(result["content"][0]["type"], "text")
        self.assertIn("docs for toksearch-pipeline",
                      result["content"][0]["text"])


@unittest.skipUnless(_SDK_AVAILABLE, "claude-agent-sdk not installed")
class TestMcpToolsConfirm(unittest.TestCase):
    def test_confirm_false_returns_isError(self):
        """confirm() returning False aborts the tool call with an error result."""
        backend = ClaudeSDKBackend()
        sess = _stub_session()
        backend._current_session = sess
        backend._current_callbacks = Callbacks(confirm=lambda call: False)
        result = asyncio.run(backend._run_python_handler(
            {"code": "x = 1", "thought": "x"}))
        self.assertTrue(result.get("isError", False))
        # Session's tool was NOT executed
        self.assertEqual(sess._executed, [])


@unittest.skipUnless(_SDK_AVAILABLE, "claude-agent-sdk not installed")
class TestBuildMcpServer(unittest.TestCase):
    def test_build_mcp_server_returns_config(self):
        backend = ClaudeSDKBackend()
        server = backend._build_mcp_server()
        # McpSdkServerConfig is a TypedDict (subclass of dict); isinstance
        # against a TypedDict raises TypeError, so we check the underlying type.
        from claude_agent_sdk.types import McpSdkServerConfig
        self.assertIsInstance(server, dict)
        self.assertEqual(server.get("type"), "sdk")


@unittest.skipUnless(_SDK_AVAILABLE, "claude-agent-sdk not installed")
class TestRunConversation(unittest.TestCase):
    """Verify run_conversation drives ClaudeSDKClient and translates messages."""

    def _make_async_iter(self, items):
        async def gen():
            for item in items:
                yield item
        return gen()

    def _mock_client(self, response_messages):
        """Build a mock ClaudeSDKClient with the given response stream."""
        client = mock.MagicMock()
        client.connect = mock.AsyncMock()
        client.query = mock.AsyncMock()
        client.receive_response = mock.MagicMock(
            return_value=self._make_async_iter(response_messages))
        return client

    def test_single_turn_text_response(self):
        from claude_agent_sdk import (
            AssistantMessage, TextBlock as SDKTextBlock, ResultMessage,
        )
        msgs = [
            AssistantMessage(content=[SDKTextBlock(text="hi")], model="m",
                              parent_tool_use_id=None, error=None,
                              usage=None, message_id="m1",
                              stop_reason="end_turn", session_id="s1",
                              uuid="u1"),
            ResultMessage(subtype="success", duration_ms=10,
                           duration_api_ms=5, is_error=False, num_turns=1,
                           session_id="s1", stop_reason="end_turn",
                           total_cost_usd=0.001, usage=None,
                           result="hi", structured_output=None,
                           model_usage=None, permission_denials=None,
                           deferred_tool_use=None, errors=None,
                           api_error_status=None, uuid="u2"),
        ]
        client = self._mock_client(msgs)
        backend = ClaudeSDKBackend()
        sess = _stub_session()
        with mock.patch(
            "toksearch.llm.backends.claude_sdk.sdk.ClaudeSDKClient",
            return_value=client,
        ):
            result = backend.run_conversation(
                sess, "hello", Callbacks(), max_iterations=5)
        self.assertEqual(result.stop_reason, "end_turn")
        self.assertEqual(result.final_text, "hi")
        # client.query was called with the user message
        client.query.assert_called_once()
        # User + assistant in history
        roles = [m.role for m in sess.history]
        self.assertEqual(roles, ["user", "assistant"])

    def test_options_configure_mcp_and_allowed_tools(self):
        """The SDK should be constructed with our MCP server + allowed_tools."""
        from claude_agent_sdk import ResultMessage
        msgs = [ResultMessage(
            subtype="success", duration_ms=1, duration_api_ms=1,
            is_error=False, num_turns=1, session_id="s",
            stop_reason="end_turn", total_cost_usd=0.0, usage=None,
            result="", structured_output=None, model_usage=None,
            permission_denials=None, deferred_tool_use=None, errors=None,
            api_error_status=None, uuid="u")]
        client = self._mock_client(msgs)
        backend = ClaudeSDKBackend()
        sess = _stub_session()
        with mock.patch(
            "toksearch.llm.backends.claude_sdk.sdk.ClaudeSDKClient",
            return_value=client,
        ) as ctor:
            backend.run_conversation(sess, "hi", Callbacks(),
                                      max_iterations=5)
        opts = ctor.call_args.kwargs["options"]
        self.assertIn("toksearch", opts.mcp_servers)
        self.assertEqual(
            sorted(opts.allowed_tools),
            ["mcp__toksearch__lookup_docs", "mcp__toksearch__run_python"])
        self.assertEqual(opts.permission_mode, "bypassPermissions")

    def test_connect_failure_raises_auth_error(self):
        from claude_agent_sdk import CLINotFoundError
        from toksearch.llm.errors import LLMAuthError
        client = mock.MagicMock()
        async def _boom():
            raise CLINotFoundError("not found")
        client.connect = _boom
        backend = ClaudeSDKBackend()
        sess = _stub_session()
        with mock.patch(
            "toksearch.llm.backends.claude_sdk.sdk.ClaudeSDKClient",
            return_value=client,
        ), self.assertRaises(LLMAuthError):
            backend.run_conversation(sess, "hi", Callbacks(),
                                      max_iterations=5)


if __name__ == "__main__":
    unittest.main()
