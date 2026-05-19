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
"""Tests for entry-point discovery.

Discovery is monkeypatched via ``mock.patch`` on
``toksearch.llm.discovery._entry_points``; tests construct fake
EntryPoint-like objects whose ``.load()`` returns the value the test wants
the discovery to surface.
"""

import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

from toksearch.llm.discovery import (
    discover_namespace_contributors,
    discover_skill_dirs,
    discover_presets,
    clear_discovery_cache,
)
from toksearch.llm.presets import Preset


def _fake_ep(name, value):
    """Return a fake EntryPoint with `.name` and `.load()`."""
    ep = mock.MagicMock()
    ep.name = name
    ep.load.return_value = value
    return ep


class _Base(unittest.TestCase):
    def setUp(self):
        clear_discovery_cache()

    def tearDown(self):
        clear_discovery_cache()


class TestDiscoverNamespace(_Base):
    def test_returns_module_with_description(self):
        mod = ModuleType("fake_pkg")
        mod.__llm_description__ = "fake description"
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", mod)]):
            result = discover_namespace_contributors()
        self.assertEqual(len(result), 1)
        name, value, desc = result[0]
        self.assertEqual(name, "fake_pkg")
        self.assertIs(value, mod)
        self.assertEqual(desc, "fake description")

    def test_missing_description_defaults_to_empty(self):
        mod = ModuleType("fake_pkg")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", mod)]):
            result = discover_namespace_contributors()
        self.assertEqual(result[0][2], "")

    def test_no_entry_points_returns_empty(self):
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[]):
            result = discover_namespace_contributors()
        self.assertEqual(result, [])


class TestDiscoverSkillDirs(_Base):
    def test_path_value(self):
        p = Path("/tmp/fake_skills")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", p)]):
            result = discover_skill_dirs()
        self.assertEqual(result, [("fake_pkg", p)])

    def test_callable_value_invoked(self):
        p = Path("/tmp/from_callable")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg", lambda: p)]):
            result = discover_skill_dirs()
        self.assertEqual(result, [("fake_pkg", p)])

    def test_string_value_coerced_to_path(self):
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("fake_pkg",
                                               "/tmp/string_skills")]):
            result = discover_skill_dirs()
        self.assertEqual(result, [("fake_pkg", Path("/tmp/string_skills"))])


class TestDiscoverPresets(_Base):
    def test_preset_value(self):
        preset = Preset(backend="anthropic", base_url="https://x.example")
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("mysite", preset)]):
            result = discover_presets()
        self.assertEqual(result, {"mysite": preset})

    def test_non_preset_value_skipped_with_warning(self):
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[_fake_ep("broken", "not a preset")]):
            with self.assertLogs("toksearch.llm.discovery", level="WARNING"):
                result = discover_presets()
        self.assertEqual(result, {})


class TestCache(_Base):
    def test_results_are_cached(self):
        mod = ModuleType("fake_pkg")
        mod.__llm_description__ = "x"
        mock_ep = _fake_ep("fake_pkg", mod)
        with mock.patch("toksearch.llm.discovery._entry_points",
                        return_value=[mock_ep]) as patched:
            discover_namespace_contributors()
            discover_namespace_contributors()
            discover_namespace_contributors()
        # Once per cache slot, not three times.
        self.assertEqual(patched.call_count, 1)


if __name__ == "__main__":
    unittest.main()
