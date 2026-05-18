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
"""OpenAIBackend — driver for the OpenAI Chat Completions API.

PR 1: blocking (non-streaming) API only.  Tool-arg streaming reassembly is
deferred to a follow-up PR.
"""

import json

import openai

from ..errors import LLMAuthError, LLMBackendError, LLMRateLimitError
from ..messages import (
    Message, TextBlock, ToolResultBlock, ToolUseBlock,
)
from .base import AssistantTurn, _ToolLoopBackend


class OpenAIBackend(_ToolLoopBackend):
    name = "openai"
    default_model = "gpt-4o"

    def __init__(self, api_key: str | None, base_url: str | None = None):
        self._api_key = api_key
        self._base_url = base_url
        self._client = None  # lazy

    def _build_client(self):
        kwargs = {}
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if self._base_url is not None:
            kwargs["base_url"] = self._base_url
        self._client = openai.OpenAI(**kwargs)
        return self._client

    def _ensure_client(self):
        if self._client is None:
            self._build_client()
        return self._client

    # ---- Translation: ToolSpec -> OpenAI tool schema ----

    @staticmethod
    def _spec_to_native(spec) -> dict:
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }

    # ---- Translation: our Message list -> OpenAI messages= ----

    @classmethod
    def _history_to_native(cls, system_prompt: str,
                            history: list[Message]) -> list[dict]:
        out = [{"role": "system", "content": system_prompt}]
        for m in history:
            if m.role == "user":
                # Could be a plain text user message OR a tool_result-only user
                # message.  OpenAI represents tool results as separate
                # role="tool" messages, so split them out.
                text_parts = [b.text for b in m.content
                              if isinstance(b, TextBlock)]
                if text_parts:
                    out.append({"role": "user",
                                "content": "\n".join(text_parts)})
                for b in m.content:
                    if isinstance(b, ToolResultBlock):
                        content = b.output
                        if b.is_error:
                            content = "[error]\n" + content
                        out.append({"role": "tool",
                                    "tool_call_id": b.tool_use_id,
                                    "content": content})
            elif m.role == "assistant":
                text_parts = [b.text for b in m.content
                              if isinstance(b, TextBlock)]
                tool_uses = [b for b in m.content
                             if isinstance(b, ToolUseBlock)]
                msg = {"role": "assistant",
                       "content": "\n".join(text_parts) if text_parts else None}
                if tool_uses:
                    msg["tool_calls"] = [{
                        "id": b.id,
                        "type": "function",
                        "function": {"name": b.name,
                                      "arguments": json.dumps(b.args)},
                    } for b in tool_uses]
                out.append(msg)
        return out

    # ---- Translation: OpenAI response -> our ContentBlock list ----

    @staticmethod
    def _response_to_blocks(message) -> list:
        out = []
        if message.content:
            out.append(TextBlock(text=message.content))
        for tc in (message.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            out.append(ToolUseBlock(id=tc.id, name=tc.function.name,
                                     args=args))
        return out

    # ---- Stop-reason translation ----

    @staticmethod
    def _stop_reason(finish_reason: str) -> str:
        return "tool_use" if finish_reason == "tool_calls" else "end_turn"

    # ---- Main entrypoint ----

    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None) -> AssistantTurn:
        client = self._ensure_client()
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=self._history_to_native(system_prompt, history),
                tools=[self._spec_to_native(t) for t in tools],
            )
        except openai.AuthenticationError as e:
            raise LLMAuthError(
                "OpenAI auth failed. Set OPENAI_API_KEY or pass api_key=... "
                "to the backend.") from e
        except openai.RateLimitError as e:
            raise LLMRateLimitError(str(e)) from e
        except openai.APIConnectionError as e:
            raise LLMBackendError(f"Network error: {e}") from e
        except openai.APIStatusError as e:
            raise LLMBackendError(f"OpenAI API error: {e}") from e
        choice = resp.choices[0]
        blocks = self._response_to_blocks(choice.message)
        text_total = "".join(b.text for b in blocks if isinstance(b, TextBlock))
        if text_total and on_text is not None:
            on_text(text_total)
        return AssistantTurn(blocks=blocks,
                              stop_reason=self._stop_reason(choice.finish_reason))
