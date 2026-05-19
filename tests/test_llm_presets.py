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
"""Tests for Preset resolution."""

import unittest

from toksearch.llm.config import Config
from toksearch.llm.errors import LLMConfigError
from toksearch.llm.presets import Preset, resolve_preset, BUILTIN_PRESETS


class TestPreset(unittest.TestCase):
    def test_builtin_anthropic_exists(self):
        self.assertIn("anthropic", BUILTIN_PRESETS)
        p = BUILTIN_PRESETS["anthropic"]
        self.assertEqual(p.backend, "anthropic")
        self.assertEqual(p.api_key_env, "ANTHROPIC_API_KEY")

    def test_builtin_openai_exists(self):
        self.assertIn("openai", BUILTIN_PRESETS)
        p = BUILTIN_PRESETS["openai"]
        self.assertEqual(p.backend, "openai")
        self.assertEqual(p.api_key_env, "OPENAI_API_KEY")


class TestResolvePreset(unittest.TestCase):
    def test_resolves_builtin(self):
        p = resolve_preset("anthropic", Config())
        self.assertEqual(p.backend, "anthropic")

    def test_unknown_preset_raises(self):
        with self.assertRaises(LLMConfigError):
            resolve_preset("nonexistent", Config())

    def test_user_preset_overrides_builtin(self):
        cfg = Config(user_presets={"anthropic": {"model": "custom-model"}})
        p = resolve_preset("anthropic", cfg)
        # User preset is shallow-merged onto built-in; model is overridden.
        self.assertEqual(p.model, "custom-model")
        # Other fields fall through from built-in.
        self.assertEqual(p.backend, "anthropic")

    def test_user_only_preset(self):
        cfg = Config(user_presets={"mysite": {"backend": "anthropic",
                                              "base_url": "https://x.example",
                                              "model": "claude-sonnet-4-6",
                                              "api_key_env": "MY_KEY"}})
        p = resolve_preset("mysite", cfg)
        self.assertEqual(p.backend, "anthropic")
        self.assertEqual(p.base_url, "https://x.example")
        self.assertEqual(p.api_key_env, "MY_KEY")


class TestDiscoveredPresets(unittest.TestCase):
    def setUp(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def tearDown(self):
        from toksearch.llm.discovery import clear_discovery_cache
        clear_discovery_cache()

    def test_discovered_preset_resolves(self):
        from unittest import mock
        preset = Preset(backend="anthropic",
                         base_url="https://site.example",
                         api_key_env="SITE_KEY")
        ep = mock.MagicMock()
        ep.name = "site"
        ep.load.return_value = preset
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [ep] if group == "toksearch.llm.presets" else [],
        ):
            p = resolve_preset("site", Config())
        self.assertEqual(p.base_url, "https://site.example")

    def test_user_preset_overrides_discovered(self):
        from unittest import mock
        discovered = Preset(backend="anthropic",
                              base_url="https://discovered.example")
        ep = mock.MagicMock()
        ep.name = "site"
        ep.load.return_value = discovered
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [ep] if group == "toksearch.llm.presets" else [],
        ):
            cfg = Config(user_presets={"site": {"base_url": "https://user.example"}})
            p = resolve_preset("site", cfg)
        self.assertEqual(p.base_url, "https://user.example")
        # Other fields fall through from discovered.
        self.assertEqual(p.backend, "anthropic")

    def test_discovered_does_not_override_builtin(self):
        """Discovered presets do NOT shadow built-ins of the same name."""
        from unittest import mock
        sneaky = Preset(backend="anthropic",
                         base_url="https://hijack.example")
        ep = mock.MagicMock()
        ep.name = "anthropic"
        ep.load.return_value = sneaky
        with mock.patch(
            "toksearch.llm.discovery._entry_points",
            side_effect=lambda group: [ep] if group == "toksearch.llm.presets" else [],
        ):
            p = resolve_preset("anthropic", Config())
        # Built-in wins.
        self.assertIsNone(p.base_url)


if __name__ == "__main__":
    unittest.main()
