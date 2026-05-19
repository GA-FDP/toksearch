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
"""Backend registry and class lookup for toksearch.llm."""

from ..errors import LLMConfigError


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
