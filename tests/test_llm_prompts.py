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
"""Tests for system-prompt assembly."""

import unittest

from toksearch.llm.tools import Skill
from toksearch.llm.prompts import build_system_prompt


class TestBuildSystemPrompt(unittest.TestCase):
    def test_contains_kernel_text(self):
        sp = build_system_prompt(skills={}, namespace_entries=[])
        self.assertIn("TokSearch", sp)
        self.assertIn("run_python", sp)

    def test_lists_namespace_entries(self):
        sp = build_system_prompt(
            skills={},
            namespace_entries=[("toksearch_d3d", "DIII-D signal classes")],
        )
        self.assertIn("toksearch_d3d", sp)
        self.assertIn("DIII-D signal classes", sp)

    def test_lists_skills_in_catalog(self):
        sp = build_system_prompt(
            skills={"foo": Skill(name="foo", description="Foo skill", body=""),
                    "bar": Skill(name="bar", description="Bar skill", body="")},
            namespace_entries=[],
        )
        self.assertIn("foo", sp)
        self.assertIn("Foo skill", sp)
        self.assertIn("bar", sp)
        self.assertIn("Bar skill", sp)

    def test_empty_catalogs_omit_sections(self):
        sp = build_system_prompt(skills={}, namespace_entries=[])
        # Should still be a non-empty kernel prompt without empty bullet lists.
        self.assertNotIn("\n-\n", sp)
        self.assertNotIn(" - :", sp)


if __name__ == "__main__":
    unittest.main()
