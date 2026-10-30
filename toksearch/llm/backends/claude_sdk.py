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
import re
import threading
from typing import TYPE_CHECKING

import claude_agent_sdk as sdk

from ..errors import LLMAuthError, LLMBackendError, LLMConfigError, LLMResumeError
from ..events import TurnComplete
from .base import Backend, Callbacks

if TYPE_CHECKING:
    from ..session import Session

# Errors that say nothing about the saved transcript: retrying with a recap
# would only lose the real context, so they are reported as they are.
# Status codes match only as whole tokens, and only after session UUIDs and
# paths are removed, since those often contain the digits 401 or 429.
_AUTH_PHRASES = ("please run /login", "claude login", "not logged in",
                 "invalid api key", "authentication_error", "oauth token",
                 "authentication", "unauthorized")
_LIMIT_PHRASES = ("rate limit", "rate_limit", "usage limit", "limit reached",
                  "overloaded")
_ACCOUNT_ERROR_MARKERS = (_AUTH_PHRASES + _LIMIT_PHRASES, ("401", "429"))
_AUTH_ERROR_MARKERS = (_AUTH_PHRASES, ("401",))
_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}"
                      r"-[0-9a-f]{12}", re.IGNORECASE)
_PATH_RE = re.compile(r"\S*[/\\]\S*")


def _mentions(text: str, markers) -> bool:
    phrases, codes = markers
    text = _UUID_RE.sub(" ", text).lower()
    if any(p in text for p in phrases):
        return True
    bare = _PATH_RE.sub(" ", text)
    return any(re.search(rf"\b{c}\b", bare) for c in codes)


def _result_error_text(msg) -> str:
    parts = [msg.result, msg.subtype, getattr(msg, "errors", None)]
    return " ".join(str(p) for p in parts if p)


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
        self._persist: dict = {}
        self._resuming = False
        self._tool_called = False   # set by the MCP handlers during a turn
        self.session_id: str | None = None

    def configure_persistence(self, *, store, cwd, session_id=None,
                              resume=None) -> None:
        """Keep the transcript in ``store``; set before the first send.

        Pass ``session_id`` (a new UUID) to start a conversation or
        ``resume`` (an existing one) to continue it.
        """
        if self._client is not None:
            raise LLMConfigError(
                "configure_persistence must be called before the first send")
        if (session_id is None) == (resume is None):
            raise ValueError("pass exactly one of session_id and resume")
        self._persist = {"session_store": store, "cwd": str(cwd),
                         "session_id": session_id, "resume": resume}
        self._resuming = resume is not None
        self.session_id = session_id or resume

    def close(self) -> None:
        """Disconnect the CLI and stop the event loop.

        The next send starts a new loop and a new SDK conversation.
        """
        client, self._client = self._client, None
        if client is not None and self._loop is not None:
            try:
                self._run_coro(client.disconnect())
            except Exception:  # noqa: BLE001 -- closing must not fail
                pass
        self._persist = {}
        self._resuming = False
        self.session_id = None
        self._stop_loop()

    def interrupt(self) -> None:
        """Ask the CLI to stop the current turn; safe from any thread.

        Waits up to 5 s; errors are swallowed. No-op before the first send.
        """
        client, loop = self._client, self._loop
        if client is None or loop is None or loop.is_closed():
            return
        try:
            future = asyncio.run_coroutine_threadsafe(client.interrupt(), loop)
            future.result(5)
        except Exception:  # noqa: BLE001 -- interrupting must not fail
            pass

    def _stop_loop(self) -> None:
        """Stop and close the event-loop thread; the next send starts a new one."""
        loop, thread = self._loop, self._thread
        self._loop = self._thread = None
        if loop is None:
            return
        if not loop.is_closed():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(5)
        if not loop.is_running() and not loop.is_closed():
            loop.close()

    # ------------------------------------------------------------------
    # MCP tool handlers (called by the SDK's internal loop via in-process MCP)
    # ------------------------------------------------------------------

    async def _run_python_handler(self, args: dict) -> dict:
        """MCP tool: run_python.  Proxies to session._execute_tool."""
        from ..events import ToolCall, ToolResult
        from ..messages import ToolUseBlock
        import uuid as _uuid
        self._tool_called = True
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
        self._tool_called = True
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
        assistant_blocks: list = []
        final_text = ""
        stop_reason = "end_turn"
        self._tool_called = False

        def nothing_happened():
            # A resume failure can only be told apart from an ordinary
            # failure before the model has replied or run a tool.
            return not assistant_blocks and not self._tool_called
        try:
            client = await self._get_or_create_client(session, max_iterations)
            await client.query(prompt)
            async for msg in client.receive_response():
                if isinstance(msg, sdk.AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, sdk.TextBlock):
                            callbacks.fire_text(b.text)
                            assistant_blocks.append(_TextBlock(text=b.text))
                            final_text = b.text
                        elif isinstance(b, sdk.ToolUseBlock):
                            # The SDK already invoked the tool via our MCP
                            # server (which appended the tool_result to
                            # history). Record the tool_use block too.
                            assistant_blocks.append(_ToolUseBlock(
                                id=b.id, name=b.name, args=dict(b.input)))
                elif isinstance(msg, sdk.ResultMessage):
                    if msg.session_id:
                        self.session_id = msg.session_id
                    if (self._resuming and msg.is_error
                            and nothing_happened()):
                        text = _result_error_text(msg)
                        if _mentions(text, _ACCOUNT_ERROR_MARKERS):
                            raise LLMBackendError(
                                f"Claude Agent SDK error: {text}")
                        raise LLMResumeError(
                            "The saved conversation could not be resumed: "
                            f"{msg.result or msg.subtype}")
                    if msg.stop_reason:
                        if msg.stop_reason in ("end_turn", "max_iterations",
                                                "interrupted"):
                            stop_reason = msg.stop_reason
                        else:
                            stop_reason = "end_turn"
                    if msg.result:
                        final_text = msg.result
                    break
        except sdk.ClaudeSDKError as e:
            if self._resuming and nothing_happened():
                raise LLMResumeError(
                    f"The saved conversation could not be resumed: {e}") from e
            raise LLMBackendError(f"Claude Agent SDK error: {e}") from e
        self._resuming = False
        if assistant_blocks:
            session._append_assistant(assistant_blocks)
        result = TurnComplete(stop_reason=stop_reason, final_text=final_text)
        callbacks.fire_turn_complete(result)
        return result

    async def _get_or_create_client(self, session, max_iterations):
        if self._client is not None:
            return self._client
        persist = {k: v for k, v in self._persist.items() if v is not None}
        if persist:
            persist["session_store_flush"] = "eager"
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
            **persist,
        )
        client = sdk.ClaudeSDKClient(options=options)
        try:
            await client.connect()
        except Exception as e:
            try:
                await client.disconnect()
            except Exception:  # noqa: BLE001 -- best effort
                pass
            translated = self._connect_error(e)
            if translated is None:
                raise
            raise translated from e
        self._client = client
        return client

    def _connect_error(self, e):
        """The LLM error for a failed connect(), or None to re-raise ``e``."""
        if isinstance(e, sdk.CLINotFoundError):
            return LLMAuthError(
                "Could not find the `claude` CLI. Install Claude Code "
                "(see https://docs.claude.com/en/docs/claude-code) and "
                "run `claude login` to use the claude-max backend.")
        login = LLMAuthError(
            "Failed to connect to the `claude` CLI. Run `claude login` "
            "or set CLAUDE_CODE_OAUTH_TOKEN. Original error: " + str(e))
        if isinstance(e, (sdk.CLIConnectionError, RuntimeError)):
            # With a session_store, connect() also loads the saved
            # transcript, so while resuming a failure is a resume failure
            # unless it says the login is the problem.
            if self._resuming:
                if _mentions(str(e), _AUTH_ERROR_MARKERS):
                    return login
                return LLMResumeError(
                    f"The saved conversation could not be resumed: {e}")
            if isinstance(e, sdk.CLIConnectionError):
                return login
            return LLMBackendError(f"Claude Agent SDK error: {e}")
        if isinstance(e, sdk.ClaudeSDKError):
            return LLMBackendError(f"Claude Agent SDK error: {e}")
        return None
