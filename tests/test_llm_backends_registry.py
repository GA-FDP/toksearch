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


if __name__ == "__main__":
    unittest.main()
