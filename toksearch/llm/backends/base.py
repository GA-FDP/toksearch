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
"""Backend ABC and the shared tool-use loop.

``_ToolLoopBackend`` holds the loop that all raw-API backends (Anthropic,
OpenAI, AmSC-via-preset) share.  Subclasses implement only:

- ``_send_request(system_prompt, history, tools, model, on_text)`` returning
  an ``AssistantTurn``.
- Translation helpers between our ``ContentBlock`` / ``ToolSpec`` taxonomy and
  the provider's native shapes.

Streaming text is deferred to a follow-up PR.  In PR 1, ``_send_request`` is
expected to be non-streaming; it MAY call ``on_text`` once at the end with the
full assistant text, but the loop does not require it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal

from ..events import TextDelta, ToolCall, ToolResult, TurnComplete
from ..messages import Message, TextBlock, ToolUseBlock, ToolResultBlock


@dataclass
class AssistantTurn:
    """One assistant round-trip from a backend.

    ``blocks`` is the assistant's content this turn (text + tool_use mixed).
    ``stop_reason`` mirrors the provider's stop reason; ``"tool_use"`` means
    the loop should execute tools and call ``_send_request`` again.
    """

    blocks: list  # list[ContentBlock]
    stop_reason: Literal["tool_use", "end_turn", "max_tokens", "stop_sequence"]


@dataclass
class Callbacks:
    on_text: Callable[[str], None] | None = None
    on_tool_call: Callable[[ToolCall], None] | None = None
    on_tool_result: Callable[[ToolResult], None] | None = None
    on_event: Callable[[object], None] | None = None
    confirm: Callable[[ToolCall], bool] | None = None

    def fire_text(self, text: str) -> None:
        if self.on_text is not None:
            self.on_text(text)
        if self.on_event is not None:
            self.on_event(TextDelta(text=text))

    def fire_tool_call(self, e: ToolCall) -> None:
        if self.on_tool_call is not None:
            self.on_tool_call(e)
        if self.on_event is not None:
            self.on_event(e)

    def fire_tool_result(self, e: ToolResult) -> None:
        if self.on_tool_result is not None:
            self.on_tool_result(e)
        if self.on_event is not None:
            self.on_event(e)

    def fire_turn_complete(self, e: TurnComplete) -> None:
        if self.on_event is not None:
            self.on_event(e)


class Backend(ABC):
    """Pluggable LLM provider.

    Subclasses advance the conversation by one user-message worth of work,
    dispatching ``Callbacks`` events along the way.
    """

    name: str
    default_model: str

    @abstractmethod
    def run_conversation(
        self,
        session,
        new_user_message: str,
        callbacks: Callbacks,
        max_iterations: int,
    ) -> TurnComplete:
        ...


class _ToolLoopBackend(Backend):
    """Shared tool-use loop for raw-API backends (Anthropic, OpenAI)."""

    @abstractmethod
    def _send_request(self, *, system_prompt, history, tools, model,
                      on_text=None) -> AssistantTurn:
        ...

    def run_conversation(self, session, new_user_message, callbacks,
                          max_iterations):
        session._append_user(new_user_message)
        final_text = ""
        for _ in range(max_iterations):
            turn = self._send_request(
                system_prompt=session.system_prompt,
                history=session.history,
                tools=session.tool_specs,
                model=session.model,
                on_text=callbacks.fire_text,
            )
            session._append_assistant(turn.blocks)
            tool_use_blocks = [b for b in turn.blocks
                               if isinstance(b, ToolUseBlock)]
            text_blocks = [b for b in turn.blocks
                           if isinstance(b, TextBlock)]
            if text_blocks:
                final_text = text_blocks[-1].text
            for block in tool_use_blocks:
                call = ToolCall(
                    id=block.id, name=block.name, args=block.args,
                    thought=block.args.get("thought")
                            if block.name == "run_python" else None,
                )
                callbacks.fire_tool_call(call)
                if callbacks.confirm is not None and not callbacks.confirm(call):
                    result = TurnComplete(stop_reason="interrupted", final_text="")
                    callbacks.fire_turn_complete(result)
                    return result
                output = session._execute_tool(block)
                event = ToolResult(id=block.id, output=output.text,
                                    is_error=output.is_error)
                callbacks.fire_tool_result(event)
                session._append_tool_result(block.id, output.text,
                                             output.is_error)
            if turn.stop_reason != "tool_use":
                result = TurnComplete(stop_reason="end_turn",
                                       final_text=final_text)
                callbacks.fire_turn_complete(result)
                return result
        result = TurnComplete(stop_reason="max_iterations",
                               final_text=final_text)
        callbacks.fire_turn_complete(result)
        return result
