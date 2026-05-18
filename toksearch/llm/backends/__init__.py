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
    """Resolve a backend-class name (e.g. 'anthropic', 'openai') to its class.

    Imports the concrete class lazily so the import of ``toksearch.llm.backends``
    doesn't require ``anthropic`` / ``openai`` SDKs to be installed.
    """
    if name == "anthropic":
        from .anthropic import AnthropicBackend
        return AnthropicBackend
    if name == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend
    raise LLMConfigError(
        f"Unknown backend: {name!r}. Known: anthropic, openai. "
        "(claude-max is added in PR 3.)")
