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

"""Public-surface tests for toksearch.llm."""

import unittest


class TestPackageImports(unittest.TestCase):
    def test_import_toksearch_llm(self):
        import toksearch.llm  # noqa: F401

    def test_import_toksearch_llm_backends(self):
        import toksearch.llm.backends  # noqa: F401


class TestPublicSurface(unittest.TestCase):
    def test_session_importable(self):
        from toksearch.llm import Session  # noqa: F401

    def test_event_types_importable(self):
        from toksearch.llm import (
            TextDelta, ToolCall, ToolResult, TurnComplete,
        )  # noqa: F401

    def test_error_types_importable(self):
        from toksearch.llm import (
            LLMError, LLMConfigError, LLMAuthError,
            LLMBackendError, LLMRateLimitError, LLMUserAbort,
        )  # noqa: F401

    def test_config_and_presets_importable(self):
        from toksearch.llm import (
            Config, load_config, Preset, BUILTIN_PRESETS, resolve_preset,
        )  # noqa: F401


class TestConsoleScript(unittest.TestCase):
    def test_cli_main_callable(self):
        from toksearch.llm.cli import main
        self.assertTrue(callable(main))


if __name__ == "__main__":
    unittest.main()
