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
"""AnthropicBackend — driver for the Anthropic Messages API.

Also serves AmSC and any other Anthropic-compatible endpoint via the
``base_url`` constructor argument (set by preset resolution).

PR 1 uses the blocking, non-streaming API.  Streaming text deltas are deferred
to a follow-up PR.
"""

import anthropic

from ..errors import LLMAuthError, LLMBackendError, LLMRateLimitError
from ..messages import (
    Message, TextBlock, ToolResultBlock, ToolUseBlock,
)
from .base import AssistantTurn, _ToolLoopBackend


class AnthropicBackend(_ToolLoopBackend):
    name = "anthropic"
    default_model = "claude-sonnet-4-6"

    def __init__(
        self,
        api_key: str | None,
        base_url: str | None = None,
        max_tokens: int = 4096,
    ):
        self._api_key = api_key
        self._base_url = base_url
        self._max_tokens = max_tokens
        self._client = None  # lazy; tests inject a mock

    def _build_client(self):
        kwargs = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def _ensure_client(self):
        if self._client is None:
            self._build_client()
        return self._client

    # ---- Translation: ToolSpec -> Anthropic tool schema ----

    @staticmethod
    def _spec_to_native(spec) -> dict:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }

    # ---- Translation: our Message list -> Anthropic messages= ----

    @staticmethod
    def _history_to_native(history: list[Message]) -> list[dict]:
        out = []
        for m in history:
            native_blocks = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    native_blocks.append({"type": "text", "text": b.text})
                elif isinstance(b, ToolUseBlock):
                    native_blocks.append({
                        "type": "tool_use",
                        "id": b.id, "name": b.name, "input": b.args,
                    })
                elif isinstance(b, ToolResultBlock):
                    native_blocks.append({
                        "type": "tool_result",
                        "tool_use_id": b.tool_use_id,
                        "content": b.output,
                        "is_error": b.is_error,
                    })
            out.append({"role": m.role, "content": native_blocks})
        return out

    # ---- Translation: Anthropic response.content -> our ContentBlock list ----

    @staticmethod
    def _response_to_blocks(content) -> list:
        out = []
        for blk in content:
            if blk.type == "text":
                out.append(TextBlock(text=blk.text))
            elif blk.type == "tool_use":
                out.append(ToolUseBlock(id=blk.id, name=blk.name,
                                         args=dict(blk.input)))
        return out

    # ---- Main entrypoint called by _ToolLoopBackend ----

    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None) -> AssistantTurn:
        client = self._ensure_client()
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=self._history_to_native(history),
                tools=[self._spec_to_native(t) for t in tools],
            )
        except anthropic.AuthenticationError as e:
            raise LLMAuthError(
                "Anthropic auth failed. Set ANTHROPIC_API_KEY or pass "
                "api_key=... to the backend.") from e
        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except anthropic.APIConnectionError as e:
            raise LLMBackendError(f"Network error: {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMBackendError(f"Anthropic API error: {e}") from e
        blocks = self._response_to_blocks(resp.content)
        text_total = "".join(b.text for b in blocks if isinstance(b, TextBlock))
        if text_total and on_text is not None:
            on_text(text_total)
        return AssistantTurn(blocks=blocks, stop_reason=resp.stop_reason)
