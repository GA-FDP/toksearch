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
"""Tests for Config and load_config()."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from toksearch.llm.config import Config, load_config


class TestConfigDefaults(unittest.TestCase):
    def test_default_backend_is_none(self):
        cfg = Config()
        self.assertIsNone(cfg.backend)

    def test_default_max_iterations(self):
        cfg = Config()
        self.assertEqual(cfg.max_iterations, 20)

    def test_anthropic_key_default(self):
        cfg = Config()
        self.assertIsNone(cfg.anthropic_api_key)


class TestLoadConfig(unittest.TestCase):
    def setUp(self):
        # Isolate env so test order doesn't matter
        self._saved_env = {k: os.environ.pop(k, None)
                           for k in ("FDP_LLM_BACKEND",
                                     "ANTHROPIC_API_KEY",
                                     "OPENAI_API_KEY")}

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v

    def test_no_file_no_env_returns_defaults(self):
        cfg = load_config(config_path=Path("/does/not/exist"))
        self.assertIsNone(cfg.backend)
        self.assertEqual(cfg.max_iterations, 20)

    def test_loads_from_toml(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[llm]\nbackend = "openai"\nmax_iterations = 5\n'
                    'anthropic_api_key = "sk-file"\n')
            path = Path(f.name)
        try:
            cfg = load_config(config_path=path)
            self.assertEqual(cfg.backend, "openai")
            self.assertEqual(cfg.max_iterations, 5)
            self.assertEqual(cfg.anthropic_api_key, "sk-file")
        finally:
            path.unlink()

    def test_env_overrides_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write('[llm]\nbackend = "openai"\nanthropic_api_key = "sk-file"\n')
            path = Path(f.name)
        try:
            with mock.patch.dict(os.environ, {"FDP_LLM_BACKEND": "anthropic",
                                              "ANTHROPIC_API_KEY": "sk-env"}):
                cfg = load_config(config_path=path)
            self.assertEqual(cfg.backend, "anthropic")
            self.assertEqual(cfg.anthropic_api_key, "sk-env")
        finally:
            path.unlink()

    def test_malformed_toml_raises_config_error(self):
        from toksearch.llm.errors import LLMConfigError
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("this is not [valid toml")
            path = Path(f.name)
        try:
            with self.assertRaises(LLMConfigError):
                load_config(config_path=path)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
