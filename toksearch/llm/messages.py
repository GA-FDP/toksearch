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
"""Provider-neutral conversation history representation.

``Message`` is one turn in the history.  Its ``content`` is a list of
``ContentBlock``s — a tagged union (``kind`` field) of text, tool-use, and
tool-result blocks.  Backends translate to and from their native shapes on
their request/response boundary; the Session and tests see only this taxonomy.

``*_to_dict`` and ``dict_to_*`` helpers provide JSON-safe round-trip so
``/save`` (future) can serialize the history without provider-specific objects.
"""

from dataclasses import dataclass, field
from typing import Literal, Union


@dataclass
class TextBlock:
    text: str
    kind: Literal["text"] = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    args: dict
    kind: Literal["tool_use"] = "tool_use"


@dataclass
class ToolResultBlock:
    tool_use_id: str
    output: str
    is_error: bool
    kind: Literal["tool_result"] = "tool_result"


ContentBlock = Union[TextBlock, ToolUseBlock, ToolResultBlock]


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: list[ContentBlock] = field(default_factory=list)


def block_to_dict(b: ContentBlock) -> dict:
    if isinstance(b, TextBlock):
        return {"kind": "text", "text": b.text}
    if isinstance(b, ToolUseBlock):
        return {"kind": "tool_use", "id": b.id, "name": b.name, "args": b.args}
    if isinstance(b, ToolResultBlock):
        return {"kind": "tool_result", "tool_use_id": b.tool_use_id,
                "output": b.output, "is_error": b.is_error}
    raise ValueError(f"Unknown block type: {type(b).__name__}")


def dict_to_block(d: dict) -> ContentBlock:
    kind = d.get("kind")
    if kind == "text":
        return TextBlock(text=d["text"])
    if kind == "tool_use":
        return ToolUseBlock(id=d["id"], name=d["name"], args=d["args"])
    if kind == "tool_result":
        return ToolResultBlock(tool_use_id=d["tool_use_id"],
                               output=d["output"], is_error=d["is_error"])
    raise ValueError(f"Unknown block kind: {kind!r}")


def message_to_dict(m: Message) -> dict:
    return {"role": m.role,
            "content": [block_to_dict(b) for b in m.content]}


def dict_to_message(d: dict) -> Message:
    return Message(role=d["role"],
                   content=[dict_to_block(b) for b in d["content"]])
