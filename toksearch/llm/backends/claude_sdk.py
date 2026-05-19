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

    def run_conversation(self, session, new_user_message, callbacks,
                          max_iterations):
        raise NotImplementedError(
            "ClaudeSDKBackend.run_conversation is implemented in Task 3.")
