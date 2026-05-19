# `toksearch.llm` PR 3 — Claude Agent SDK Backend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Add `ClaudeSDKBackend` so users with a Claude Max plan can drive `toksearch.llm` against their subscription instead of paying API costs. The SDK is intentionally restricted to our two MCP tools (`run_python`, `lookup_docs`) — built-in Claude Code tools (Bash, Read, Edit, Glob, Grep) are NOT enabled. The SDK is "just a way to bill Claude Max for tokens," not a richer agent.

**Architecture:** `ClaudeSDKBackend` does NOT subclass `_ToolLoopBackend` — the SDK runs its own internal loop, and our loop logic doesn't fit. The backend implements `Backend.run_conversation` directly using `claude_agent_sdk.ClaudeSDKClient`. To preserve multi-turn conversational state across `Session.send()` calls (Session is sync; the SDK is async-first), a persistent asyncio event loop runs on a background thread, and each `send()` posts a coroutine via `run_coroutine_threadsafe`. The SDK's internal tool loop calls back into our process via an in-process MCP server (`create_sdk_mcp_server`) whose `run_python` and `lookup_docs` handlers proxy to `session._execute_tool` and dispatch `ToolCall`/`ToolResult` events through the active callbacks.

**Tech Stack:** Python 3.11, `claude-agent-sdk` 0.2+, `mcp` 1.23+ (conda-forge), `asyncio`, `unittest`, `unittest.mock`.

**Reference spec:** `docs/superpowers/specs/2026-05-18-toksearch-llm-design.md` (the `ClaudeSDKBackend` section).

**Branch:** `feat/llm-pr3` off `feat/llm-pr2` (will rebase onto main when PR 2 merges).

## Scope notes

- `allowed_tools` restricts the SDK to exactly two MCP tools we expose: `mcp__toksearch__run_python` and `mcp__toksearch__lookup_docs`. Built-in Claude Code tools are disabled. (Optional `--full-tools` flag deferred.)
- `permission_mode="bypassPermissions"` — we already surface code via callbacks; the SDK's own permission prompts would be redundant noise.
- The `claude` CLI must be installed and authenticated (e.g. `claude login`) or `CLAUDE_CODE_OAUTH_TOKEN` set. Auth failure surfaces as `LLMAuthError` with explicit remediation.
- `claude-agent-sdk` + `mcp` are added to `[project.optional-dependencies].llm` so `pip install toksearch[llm]` gets the SDK. They are lazy-imported in `claude_sdk.py`; the registry catches `ImportError` and re-raises as `LLMConfigError` with install guidance.
- Persistent event loop runs on a daemon thread; `Session` doesn't expose a `close()` and the loop dies with the process. A `close()` method would be a follow-up if needed.
- The `confirm=` callback fires INSIDE our MCP tool handlers — same UX as raw API backends.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `toksearch/llm/backends/claude_sdk.py` | Create | `ClaudeSDKBackend` class, MCP server builder, async bridge. |
| `toksearch/llm/backends/__init__.py` | Modify | Add `"claude-max"` to `get_backend_class` lazy lookup. |
| `toksearch/llm/presets.py` | Modify | Add `"claude-max"` to `BUILTIN_PRESETS`. |
| `toksearch/llm/cli.py` | Modify | `_resolve_api_key` returns None for `claude-max` (SDK uses OAuth, not an API key). |
| `pyproject.toml` | Modify | Add `claude-agent-sdk` + `mcp` to `[project.optional-dependencies].llm`. |
| `tests/test_llm_backends_claude_sdk.py` | Create | Mocked-SDK tests for MCP wiring, message translation, run_conversation. |
| `tests/test_llm_backends_registry.py` | Modify | Add `claude-max` registry test. |
| `tests/test_llm_presets.py` | Modify | Add `claude-max` builtin preset test. |
| `tests/test_llm_integration.py` | Modify | Add gated integration test for Claude Max backend. |

---

## Task 1: `claude-max` preset, registry entry, and `ClaudeSDKBackend` skeleton

**Files:**
- Modify: `toksearch/llm/presets.py`
- Modify: `toksearch/llm/backends/__init__.py`
- Modify: `toksearch/llm/cli.py`
- Create: `toksearch/llm/backends/claude_sdk.py` (skeleton with raising stub)
- Modify: `tests/test_llm_presets.py`
- Modify: `tests/test_llm_backends_registry.py`

Use the FULL 13-line Apache 2.0 header (matching `setup.py` lines 1-13) in any new file.

- [ ] **Step 1: Add builtin preset test**

Append to `tests/test_llm_presets.py` (BEFORE `if __name__ == "__main__":`):

```python
class TestClaudeMaxPreset(unittest.TestCase):
    def test_claude_max_in_builtins(self):
        self.assertIn("claude-max", BUILTIN_PRESETS)
        p = BUILTIN_PRESETS["claude-max"]
        self.assertEqual(p.backend, "claude-max")
        # Uses OAuth via `claude` CLI, not an API key env var:
        self.assertIsNone(p.api_key_env)
        self.assertIsNone(p.api_key_file)

    def test_resolve_claude_max(self):
        p = resolve_preset("claude-max", Config())
        self.assertEqual(p.backend, "claude-max")
```

- [ ] **Step 2: Add registry test**

Append to `tests/test_llm_backends_registry.py` (BEFORE `if __name__`):

```python
class TestClaudeMaxBackendClass(unittest.TestCase):
    def test_get_backend_class_claude_max(self):
        from toksearch.llm.backends import get_backend_class
        from toksearch.llm.backends.claude_sdk import ClaudeSDKBackend
        self.assertIs(get_backend_class("claude-max"), ClaudeSDKBackend)
```

- [ ] **Step 3: Verify both tests fail**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest test_llm_presets.TestClaudeMaxPreset test_llm_backends_registry.TestClaudeMaxBackendClass -v'
```

Expected: `KeyError` / `LLMConfigError` for presets; `ModuleNotFoundError: No module named 'toksearch.llm.backends.claude_sdk'` for registry.

- [ ] **Step 4: Add `claude-max` to `BUILTIN_PRESETS`**

In `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/presets.py`, find the `BUILTIN_PRESETS` dict. Add `"claude-max"` entry after the existing `"openai"` entry. The final dict:

```python
BUILTIN_PRESETS: dict[str, Preset] = {
    "anthropic": Preset(
        backend="anthropic",
        model="claude-sonnet-4-6",
        api_key_env="ANTHROPIC_API_KEY",
    ),
    "openai": Preset(
        backend="openai",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
    ),
    "claude-max": Preset(
        backend="claude-max",
        model=None,  # SDK picks the default; users may override via CLI.
        api_key_env=None,
        api_key_file=None,
    ),
}
```

- [ ] **Step 5: Add `claude-max` to backend registry lazy lookup**

Open `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/backends/__init__.py`. Replace `get_backend_class` with:

```python
def get_backend_class(name: str):
    """Resolve a backend-class name to its class.

    Imports the concrete class lazily so the registry import doesn't
    require ``anthropic``, ``openai``, or ``claude-agent-sdk`` SDKs to be
    installed.  An ImportError on lazy load is re-raised as ``LLMConfigError``
    with installation guidance.
    """
    if name == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend
    if name == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend
    if name == "claude-max":
        try:
            from .claude_sdk import ClaudeSDKBackend
        except ImportError as e:
            raise LLMConfigError(
                "The claude-max backend requires the claude-agent-sdk and "
                "mcp packages. Install them via `pip install toksearch[llm]` "
                "or, in pixi-managed envs, add them under [dependencies] in "
                "pixi.toml.") from e
        return ClaudeSDKBackend
    raise LLMConfigError(
        f"Unknown backend: {name!r}. Known: anthropic, openai, claude-max.")
```

- [ ] **Step 6: Adjust `_resolve_api_key` for `claude-max`**

Open `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/cli.py`. Find `_resolve_api_key`. After the existing `preset.api_key_env` / `preset.api_key_file` lookups, BEFORE the `if preset.backend == "anthropic"` block, add:

```python
    if preset.backend == "claude-max":
        # ClaudeSDKBackend uses OAuth via the `claude` CLI; no API key.
        return None
```

This is conceptually redundant (the function already returns None at the bottom for unknown backends) but it makes the intent explicit and prevents accidental fallthrough if someone later adds a config-level `claude_max_api_key` field.

- [ ] **Step 7: Create `claude_sdk.py` skeleton**

Create `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/backends/claude_sdk.py` with FULL Apache header, then:

```python
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

    def run_conversation(self, session, new_user_message, callbacks,
                          max_iterations):
        raise NotImplementedError(
            "ClaudeSDKBackend.run_conversation is implemented in Task 3.")
```

- [ ] **Step 8: Verify all tests pass**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest discover -p "test_llm*" 2>&1 | tail -5'
```

Expected: 125+ tests pass, 2 skipped. The new preset and registry tests should be green.

- [ ] **Step 9: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add \
  toksearch/llm/presets.py \
  toksearch/llm/backends/__init__.py \
  toksearch/llm/backends/claude_sdk.py \
  toksearch/llm/cli.py \
  tests/test_llm_presets.py \
  tests/test_llm_backends_registry.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Add claude-max preset and ClaudeSDKBackend skeleton

Registry lazily imports claude_sdk and re-raises ImportError as
LLMConfigError with install guidance. The backend class itself is
just a stub at this point; Task 2 wires the MCP server and Task 3
implements run_conversation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: MCP server with `run_python` and `lookup_docs` tools

**Files:**
- Modify: `toksearch/llm/backends/claude_sdk.py`
- Create: `tests/test_llm_backends_claude_sdk.py`

This task builds the in-process MCP server that the SDK's internal tool loop will call back into. Each MCP tool fires the active `Callbacks`, executes via `session._execute_tool`, and returns an MCP-shaped result.

- [ ] **Step 1: Write failing tests**

Create `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/tests/test_llm_backends_claude_sdk.py` with FULL Apache header, then:

```python
"""Tests for ClaudeSDKBackend.

The claude_agent_sdk module is real (conda-forge install), but we never
construct an actual ClaudeSDKClient -- tests inject mocks to verify the
backend wires options + MCP tools correctly and translates SDK messages
into our event types.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest import mock

from toksearch.llm.backends.base import Callbacks
from toksearch.llm.backends.claude_sdk import ClaudeSDKBackend
from toksearch.llm.messages import (
    Message, TextBlock, ToolUseBlock, ToolResultBlock,
)
from toksearch.llm.tools import ToolOutput


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


class TestBuildMcpServer(unittest.TestCase):
    def test_build_mcp_server_returns_config(self):
        backend = ClaudeSDKBackend()
        server = backend._build_mcp_server()
        # Should be an McpSdkServerConfig (returned by create_sdk_mcp_server).
        from claude_agent_sdk.types import McpSdkServerConfig
        self.assertIsInstance(server, McpSdkServerConfig)
```

- [ ] **Step 2: Verify failing**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest test_llm_backends_claude_sdk -v'
```

Expected: `AttributeError: 'ClaudeSDKBackend' object has no attribute '_run_python_handler'` and `_lookup_docs_handler` and `_build_mcp_server`.

- [ ] **Step 3: Add MCP tool handlers and server builder to `ClaudeSDKBackend`**

Open `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/backends/claude_sdk.py`. Add the following methods to `ClaudeSDKBackend` (place them BEFORE `run_conversation`):

```python
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
```

- [ ] **Step 4: Verify all tests pass**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest test_llm_backends_claude_sdk -v'
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add toksearch/llm/backends/claude_sdk.py tests/test_llm_backends_claude_sdk.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Add MCP server with run_python and lookup_docs for ClaudeSDKBackend

The two MCP tools proxy to session._execute_tool, fire callbacks before
and after execution, and honor confirm() by returning an isError result.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement `run_conversation` (SDK lifecycle + async-to-sync bridge)

**Files:**
- Modify: `toksearch/llm/backends/claude_sdk.py`
- Modify: `tests/test_llm_backends_claude_sdk.py`

This task wires the SDK's `ClaudeSDKClient` into our Session and bridges async-to-sync via a persistent event loop on a daemon thread.

- [ ] **Step 1: Add tests for `run_conversation`**

Append to `tests/test_llm_backends_claude_sdk.py` (BEFORE `if __name__`):

```python
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
```

- [ ] **Step 2: Verify failing**

- [ ] **Step 3: Implement `run_conversation` and helpers**

Append to `ClaudeSDKBackend` in `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/toksearch/llm/backends/claude_sdk.py` (replacing the `raise NotImplementedError` stub from Task 1):

```python
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
                        assistant_blocks.append(
                            __import__("toksearch.llm.messages",
                                        fromlist=["TextBlock"]).TextBlock(
                                text=b.text))
                        final_text = b.text
                    elif isinstance(b, sdk.ToolUseBlock):
                        # The SDK already invoked the tool via our MCP server
                        # (which appended the tool_result to history).  Record
                        # the tool_use block in our history too.
                        TUB = __import__(
                            "toksearch.llm.messages",
                            fromlist=["ToolUseBlock"]).ToolUseBlock
                        assistant_blocks.append(TUB(
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
        # Record the full assistant turn at once (after the loop) so the
        # provider-neutral history shape stays consistent across backends.
        # Tool-use blocks emitted by the SDK come AFTER their tool_result was
        # already appended (because the SDK called our MCP handler first);
        # we keep the chronological order by inserting the assistant turn
        # before those tool_results.  Simpler approach for PR 3: append the
        # assistant blocks at the end -- history is informational; backends
        # don't replay it for the SDK (the SDK owns its session state).
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
```

- [ ] **Step 4: Run all tests**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest test_llm_backends_claude_sdk -v'
```

Expected: 7 tests pass (4 from Task 2 + 3 new).

Run the full LLM suite:
```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest discover -p "test_llm*" 2>&1 | tail -3'
```

Expected: 128+ tests, 2 skipped, all green.

- [ ] **Step 5: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add toksearch/llm/backends/claude_sdk.py tests/test_llm_backends_claude_sdk.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Implement ClaudeSDKBackend.run_conversation

Bridges sync Session.send() to the async ClaudeSDKClient via a persistent
event loop on a daemon thread. The client is reused across send() calls
so the SDK maintains its own conversation state. CLI auth failures
(claude CLI missing or not logged in) surface as LLMAuthError with
clear remediation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Add SDKs to `[llm]` extra and integration test

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_llm_integration.py`

- [ ] **Step 1: Add `claude-agent-sdk` and `mcp` to `[project.optional-dependencies].llm`**

In `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/pyproject.toml`, find:

```toml
[project.optional-dependencies]
llm = ["anthropic>=0.34", "openai>=1.50", "matplotlib"]
```

Replace with:

```toml
[project.optional-dependencies]
llm = [
    "anthropic>=0.34",
    "openai>=1.50",
    "matplotlib",
    "claude-agent-sdk>=0.2",
    "mcp>=1.23",
]
```

- [ ] **Step 2: Add gated integration test**

Open `/fusion/projects/dt/sammuli/fdp_dev/repos/toksearch/tests/test_llm_integration.py`. After the existing `TestOpenAIIntegration` class, append:

```python
_HAS_CLAUDE_MAX = os.environ.get("TOKSEARCH_CLAUDE_MAX") == "yes"


@unittest.skipUnless(_INTEGRATION_ON and _HAS_CLAUDE_MAX,
                     "TOKSEARCH_INTEGRATION=yes and TOKSEARCH_CLAUDE_MAX=yes required; "
                     "the `claude` CLI must be installed and logged in.")
class TestClaudeMaxIntegration(unittest.TestCase):
    def test_simple_arithmetic_end_to_end(self):
        from toksearch.llm.backends.claude_sdk import ClaudeSDKBackend
        backend = ClaudeSDKBackend()
        sess = Session(backend=backend, max_iterations=5)
        result = sess.send(PROMPT)
        _assert_simple_arithmetic(result, sess)
```

Note: the Claude Max integration test uses a separate gate (`TOKSEARCH_CLAUDE_MAX=yes`) instead of an API key env var because the SDK uses OAuth via the `claude` CLI. Setting the env var is an explicit opt-in that the user has `claude login` already done.

- [ ] **Step 3: Confirm the test still skips cleanly without the gate**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest test_llm_integration -v 2>&1 | tail -10'
```

Expected: all 3 integration tests skip (2 from PR 1 + 1 new).

- [ ] **Step 4: Confirm full LLM suite still passes**

```bash
cd /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch && pixi run bash -c 'cd tests && python -m unittest discover -p "test_llm*" 2>&1 | tail -3'
```

Expected: 128+ tests, 3 skipped, all green.

- [ ] **Step 5: Commit**

```bash
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch add pyproject.toml tests/test_llm_integration.py
git -C /fusion/projects/dt/sammuli/fdp_dev/repos/toksearch commit -m "$(cat <<'EOF'
Add claude-agent-sdk to [llm] extra and Claude Max integration test

The integration test is gated on TOKSEARCH_INTEGRATION=yes plus a
separate TOKSEARCH_CLAUDE_MAX=yes opt-in, because the SDK uses OAuth
via the `claude` CLI rather than an API key env var -- the opt-in
makes it explicit that the user has `claude login` set up.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Manual smoke test (operator-run, not automated)

**Files:** none modified.

Requires `claude` CLI installed and `claude login` completed (or `CLAUDE_CODE_OAUTH_TOKEN` set).

- [ ] **Step 1: One-shot query via the SDK**

```bash
pixi run toksearch query --backend claude-max "Use run_python to compute 2 ** 10 and report the result."
```

Expected: per-iteration `[run_python] ...` lines, then a final text mentioning `1024`.

- [ ] **Step 2: Multi-turn chat preserves the namespace**

```bash
pixi run toksearch chat --backend claude-max
```

In the REPL:
```
you> use run_python to set x = 5 and confirm the value
[...agent output...]
you> what is x squared?
[...agent should reference x and compute 25...]
you> /quit
```

Expected: second turn references `x` from the first; the persistent namespace works through the MCP-tool path.

- [ ] **Step 3: Negative test — missing `claude` CLI**

```bash
PATH=/usr/bin:/bin pixi run toksearch query --backend claude-max "hi"
```

Expected: prints `error: Could not find the \`claude\` CLI...` and exits non-zero.

If any step fails, the failure is either in `run_conversation` (translation/wiring) or `_get_or_create_client` (auth). Both are covered by unit tests; a new smoke failure is worth a follow-up unit test case.

---

## Self-Review Notes

Coverage against the PR 3 scope:
- ✓ `ClaudeSDKBackend` with `claude-max` backend name → Task 1.
- ✓ Lazy import in registry with helpful error on missing SDK → Task 1.
- ✓ Built-in `claude-max` preset → Task 1.
- ✓ In-process MCP server with `run_python` and `lookup_docs` → Task 2.
- ✓ `allowed_tools` restricted to our two MCP tools (Bash/Read/Edit/Glob/Grep disabled) → Task 3.
- ✓ `permission_mode="bypassPermissions"` → Task 3.
- ✓ Persistent client across `send()` calls via background event loop → Task 3.
- ✓ `confirm=` callback honored inside MCP handler → Task 2.
- ✓ Auth error remediation when `claude` CLI missing or unauthenticated → Task 3.
- ✓ Integration test gated on opt-in env vars → Task 4.

Deferred (explicitly out of scope for PR 3):
- `--full-tools` flag to also enable Claude Code's built-in tools.
- Backend `close()` method for explicit cleanup of the event-loop thread.
- Streaming text deltas (the SDK supports `include_partial_messages=True`; PR 3 fires `on_text` once per AssistantMessage).
- Cost / usage reporting from `ResultMessage` (we ignore those fields; future PR could surface them via a new event type).
