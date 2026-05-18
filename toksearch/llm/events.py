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
"""Event dataclasses dispatched to Session.send() callbacks.

Events are frozen and compare by value.  They are emitted by the active
``Backend`` while the conversation advances; ``Session.send()`` routes them
to the appropriate ``on_<kind>`` callback (and to ``on_event`` if provided).
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TextDelta:
    """An incremental (or whole, in non-streaming backends) chunk of assistant text."""

    text: str


@dataclass(frozen=True)
class ToolCall:
    """The assistant is requesting a tool invocation.

    ``thought`` is populated for ``run_python`` (its schema requires a
    ``thought`` field) and is ``None`` for tools without one.
    """

    id: str
    name: str
    args: dict
    thought: str | None


@dataclass(frozen=True)
class ToolResult:
    """The output of a tool invocation, about to be sent back to the model."""

    id: str
    output: str
    is_error: bool


@dataclass(frozen=True)
class TurnComplete:
    """The assistant has finished this turn.

    ``stop_reason``:
    - ``"end_turn"``: assistant stopped emitting tool calls.
    - ``"max_iterations"``: hit ``Session.max_iterations`` before end_turn.
    - ``"interrupted"``: user aborted (ctrl-C) or ``confirm()`` returned False.
    """

    stop_reason: Literal["end_turn", "max_iterations", "interrupted"]
    final_text: str


Event = TextDelta | ToolCall | ToolResult | TurnComplete
