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
"""Tests for the backend registry."""

import unittest

from toksearch.llm.errors import LLMConfigError

try:
    import claude_agent_sdk  # noqa: F401
    _HAS_CLAUDE_SDK = True
except ImportError:
    _HAS_CLAUDE_SDK = False


class TestBackendRegistry(unittest.TestCase):
    def test_get_backend_class_anthropic(self):
        from toksearch.llm.backends import get_backend_class
        from toksearch.llm.backends.anthropic import AnthropicBackend
        self.assertIs(get_backend_class("anthropic"), AnthropicBackend)

    def test_get_backend_class_openai(self):
        from toksearch.llm.backends import get_backend_class
        from toksearch.llm.backends.openai import OpenAIBackend
        self.assertIs(get_backend_class("openai"), OpenAIBackend)

    def test_unknown_raises(self):
        from toksearch.llm.backends import get_backend_class
        with self.assertRaises(LLMConfigError):
            get_backend_class("nonexistent")


@unittest.skipUnless(_HAS_CLAUDE_SDK, "claude-agent-sdk not installed")
class TestClaudeMaxBackendClass(unittest.TestCase):
    def test_get_backend_class_claude_max(self):
        from toksearch.llm.backends import get_backend_class
        from toksearch.llm.backends.claude_sdk import ClaudeSDKBackend
        self.assertIs(get_backend_class("claude-max"), ClaudeSDKBackend)


class TestClaudeMaxImportErrorMessage(unittest.TestCase):
    """When claude-agent-sdk is missing, get_backend_class('claude-max') should
    raise LLMConfigError with installation guidance (rather than ImportError)."""

    def test_helpful_error_when_sdk_missing(self):
        from toksearch.llm.backends import get_backend_class
        import sys
        # Simulate "SDK not installed" by patching the import.
        from unittest import mock
        # Remove any cached import so the lazy load actually runs.
        sdk_modules = [m for m in list(sys.modules)
                       if m.startswith("toksearch.llm.backends.claude_sdk")
                       or m.startswith("claude_agent_sdk")]
        with mock.patch.dict(sys.modules, {m: None for m in sdk_modules}):
            with self.assertRaises(LLMConfigError) as cm:
                get_backend_class("claude-max")
            msg = str(cm.exception)
            self.assertIn("claude-agent-sdk", msg)
            self.assertIn("[llm]", msg)


if __name__ == "__main__":
    unittest.main()
