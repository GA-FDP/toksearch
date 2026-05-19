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
"""toksearch.llm — Conversational LLM interface for TokSearch.

See ``docs/superpowers/specs/2026-05-18-toksearch-llm-design.md``.

Quickstart
==========

::

    from toksearch.llm import Session
    from toksearch.llm.backends.anthropic import AnthropicBackend

    sess = Session(backend=AnthropicBackend(api_key="sk-..."))
    result = sess.send("Fetch ip for shot 200000 and report peak in MA.",
                       on_tool_call=lambda c: print(c.name, c.thought),
                       on_tool_result=lambda r: print(r.output))
    print(result.final_text)
"""

from .errors import (
    LLMError,
    LLMConfigError,
    LLMAuthError,
    LLMBackendError,
    LLMRateLimitError,
    LLMUserAbort,
)
from .events import TextDelta, ToolCall, ToolResult, TurnComplete
from .config import Config, load_config
from .presets import Preset, BUILTIN_PRESETS, resolve_preset
from .session import Session

from pathlib import Path as _Path

CORE_SKILLS_DIR = _Path(__file__).parent.parent / "skills"

__all__ = [
    "Session",
    "Config", "load_config",
    "Preset", "BUILTIN_PRESETS", "resolve_preset",
    "TextDelta", "ToolCall", "ToolResult", "TurnComplete",
    "LLMError", "LLMConfigError", "LLMAuthError",
    "LLMBackendError", "LLMRateLimitError", "LLMUserAbort",
    "CORE_SKILLS_DIR",
]
