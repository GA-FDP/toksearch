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
"""ClaudeSDKBackend - drive toksearch.llm against the Claude Agent SDK.

This backend exists so users with a Claude Max plan can use toksearch.llm
without paying API costs.  It is deliberately RESTRICTED -- the SDK is
configured with allowed_tools=["mcp__toksearch__run_python",
"mcp__toksearch__lookup_docs"] only.  Claude Code's built-in Bash/Read/Edit
tools are NOT enabled; users who want those should run `claude` directly.

Implementation notes:
- The SDK is async-first; toksearch.llm.Session.send() is sync.  Bridging
  is done by running a persistent asyncio event loop on a daemon thread
  and dispatching coroutines via run_coroutine_threadsafe.
- The ClaudeSDKClient is reused across send() calls within a Session so
  the SDK can maintain its own conversation state.
- The MCP server is in-process (create_sdk_mcp_server returns an
  McpSdkServerConfig); no subprocess beyond the `claude` CLI itself.
- Auth is via the `claude` CLI (run `claude login` or set
  CLAUDE_CODE_OAUTH_TOKEN).  Connection failures raise LLMAuthError.
"""

import asyncio
import threading
from typing import TYPE_CHECKING

import claude_agent_sdk as sdk

from ..errors import LLMAuthError, LLMBackendError
from ..events import TurnComplete
from .base import Backend, Callbacks

if TYPE_CHECKING:
    from ..session import Session


class ClaudeSDKBackend(Backend):
    name = "claude-max"
    default_model = "claude-sonnet-4-6"  # SDK will pick a reasonable default if None

    def __init__(self, api_key=None, base_url=None):
        # api_key and base_url are accepted for interface uniformity with
        # the other backends; both are ignored.  The SDK gets credentials
        # from `claude login` or CLAUDE_CODE_OAUTH_TOKEN at process start.
        self._loop = None
        self._thread = None
        self._client = None
        self._current_callbacks = None
        self._current_session = None

    # ------------------------------------------------------------------
    # MCP tool handlers (called by the SDK's internal loop via in-process MCP)
    # ------------------------------------------------------------------

    async def _run_python_handler(self, args: dict) -> dict:
        """MCP tool: run_python.  Proxies to session._execute_tool."""
        from ..events import ToolCall, ToolResult
        from ..messages import ToolUseBlock
        import uuid as _uuid
        sess = self._current_session
        cbs = self._current_callbacks
        thought = args.get("thought")
        call_id = f"sdk-{_uuid.uuid4().hex[:8]}"
        cbs.fire_tool_call(ToolCall(
            id=call_id, name="run_python", args=args, thought=thought,
        ))
        if cbs.confirm is not None and not cbs.confirm(ToolCall(
                id=call_id, name="run_python", args=args, thought=thought)):
            return {
                "content": [{"type": "text", "text": "(interrupted)"}],
                "isError": True,
            }
        block = ToolUseBlock(id=call_id, name="run_python", args=args)
        output = sess._execute_tool(block)
        cbs.fire_tool_result(ToolResult(
            id=call_id, output=output.text, is_error=output.is_error,
        ))
        sess._append_tool_result(call_id, output.text, output.is_error)
        return {
            "content": [{"type": "text", "text": output.text}],
            "isError": output.is_error,
        }

    async def _lookup_docs_handler(self, args: dict) -> dict:
        """MCP tool: lookup_docs.  Proxies to session._execute_tool."""
        from ..events import ToolCall, ToolResult
        from ..messages import ToolUseBlock
        import uuid as _uuid
        sess = self._current_session
        cbs = self._current_callbacks
        call_id = f"sdk-{_uuid.uuid4().hex[:8]}"
        cbs.fire_tool_call(ToolCall(
            id=call_id, name="lookup_docs", args=args, thought=None,
        ))
        block = ToolUseBlock(id=call_id, name="lookup_docs", args=args)
        output = sess._execute_tool(block)
        cbs.fire_tool_result(ToolResult(
            id=call_id, output=output.text, is_error=output.is_error,
        ))
        sess._append_tool_result(call_id, output.text, output.is_error)
        return {
            "content": [{"type": "text", "text": output.text}],
            "isError": output.is_error,
        }

    # ------------------------------------------------------------------
    # MCP server construction
    # ------------------------------------------------------------------

    def _build_mcp_server(self):
        """Construct the in-process MCP server exposing our two tools."""
        @sdk.tool(
            "run_python",
            "Execute a Python code string in the persistent toksearch "
            "session namespace.",
            {"code": str, "thought": str},
        )
        async def _rp(args):
            return await self._run_python_handler(args)

        @sdk.tool(
            "lookup_docs",
            "Read a documentation skill registered with the session.",
            {"skill_name": str},
        )
        async def _ld(args):
            return await self._lookup_docs_handler(args)

        return sdk.create_sdk_mcp_server(
            name="toksearch", version="1.0", tools=[_rp, _ld],
        )

    # ------------------------------------------------------------------
    # Async bridge
    # ------------------------------------------------------------------

    def _ensure_loop(self):
        """Start the persistent daemon-thread event loop on first use."""
        if self._loop is not None:
            return
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True,
            name="toksearch-llm-claude-sdk")
        self._thread.start()

    def _run_coro(self, coro):
        self._ensure_loop()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # ------------------------------------------------------------------
    # Conversation entry point
    # ------------------------------------------------------------------

    def run_conversation(self, session, new_user_message, callbacks,
                          max_iterations) -> TurnComplete:
        # Stash per-call state for the MCP handlers to read.
        self._current_session = session
        self._current_callbacks = callbacks
        session._append_user(new_user_message)
        return self._run_coro(self._async_turn(
            session, new_user_message, callbacks, max_iterations))

    async def _async_turn(self, session, prompt, callbacks, max_iterations):
        from ..messages import TextBlock as _TextBlock, ToolUseBlock as _ToolUseBlock
        client = await self._get_or_create_client(session, max_iterations)
        await client.query(prompt)
        # Collect assistant blocks across all messages from this turn.
        assistant_blocks: list = []
        final_text = ""
        stop_reason = "end_turn"
        async for msg in client.receive_response():
            if isinstance(msg, sdk.AssistantMessage):
                for b in msg.content:
                    if isinstance(b, sdk.TextBlock):
                        # Native text deltas can be streamed; fire on_text once
                        # per AssistantMessage in PR 3 (no chunking).
                        callbacks.fire_text(b.text)
                        assistant_blocks.append(_TextBlock(text=b.text))
                        final_text = b.text
                    elif isinstance(b, sdk.ToolUseBlock):
                        # The SDK already invoked the tool via our MCP server
                        # (which appended the tool_result to history). Record
                        # the tool_use block in our history too.
                        assistant_blocks.append(_ToolUseBlock(
                            id=b.id, name=b.name, args=dict(b.input)))
            elif isinstance(msg, sdk.ResultMessage):
                if msg.stop_reason:
                    if msg.stop_reason in ("end_turn", "max_iterations",
                                            "interrupted"):
                        stop_reason = msg.stop_reason
                    else:
                        stop_reason = "end_turn"
                if msg.result:
                    final_text = msg.result
                break
        if assistant_blocks:
            session._append_assistant(assistant_blocks)
        result = TurnComplete(stop_reason=stop_reason, final_text=final_text)
        callbacks.fire_turn_complete(result)
        return result

    async def _get_or_create_client(self, session, max_iterations):
        if self._client is not None:
            return self._client
        options = sdk.ClaudeAgentOptions(
            system_prompt=session.system_prompt,
            mcp_servers={"toksearch": self._build_mcp_server()},
            allowed_tools=[
                "mcp__toksearch__run_python",
                "mcp__toksearch__lookup_docs",
            ],
            permission_mode="bypassPermissions",
            model=session.model,
            max_turns=max_iterations,
        )
        try:
            self._client = sdk.ClaudeSDKClient(options=options)
            await self._client.connect()
        except sdk.CLINotFoundError as e:
            raise LLMAuthError(
                "Could not find the `claude` CLI. Install Claude Code "
                "(see https://docs.claude.com/en/docs/claude-code) and "
                "run `claude login` to use the claude-max backend.") from e
        except sdk.CLIConnectionError as e:
            raise LLMAuthError(
                "Failed to connect to the `claude` CLI. Run `claude login` "
                "or set CLAUDE_CODE_OAUTH_TOKEN. Original error: " + str(e)
            ) from e
        except sdk.ClaudeSDKError as e:
            raise LLMBackendError(f"Claude Agent SDK error: {e}") from e
        return self._client
