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
"""Tests for tools.py — ToolSpec, ToolOutput, and tool handlers.

Tool handlers take ``(args: dict, session)``.  These tests pass a duck-typed
stub with a ``.namespace`` attribute; the real ``Session`` class is tested in
``test_llm_session.py``.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

from toksearch.llm.tools import (
    ToolSpec,
    ToolOutput,
    RUN_PYTHON,
)


def _stub_session(namespace: dict | None = None):
    return SimpleNamespace(namespace=namespace if namespace is not None else {})


class TestToolSpec(unittest.TestCase):
    def test_run_python_spec_shape(self):
        self.assertEqual(RUN_PYTHON.name, "run_python")
        self.assertIn("code", RUN_PYTHON.input_schema["properties"])
        self.assertIn("thought", RUN_PYTHON.input_schema["properties"])
        self.assertEqual(set(RUN_PYTHON.input_schema["required"]),
                         {"code", "thought"})


class TestRunPython(unittest.TestCase):
    def test_simple_expression_no_output(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "x = 1 + 1", "thought": "test"}, s)
        self.assertFalse(out.is_error)
        self.assertEqual(s.namespace["x"], 2)

    def test_stdout_captured(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "print('hi')", "thought": "x"}, s)
        self.assertFalse(out.is_error)
        self.assertIn("hi", out.text)

    def test_stderr_captured(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "import sys; sys.stderr.write('warn\\n')",
                                  "thought": "x"}, s)
        self.assertFalse(out.is_error)
        self.assertIn("warn", out.text)

    def test_exception_becomes_error(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "1/0", "thought": "boom"}, s)
        self.assertTrue(out.is_error)
        self.assertIn("ZeroDivisionError", out.text)
        # Traceback contains the offending line:
        self.assertIn("Traceback", out.text)

    def test_namespace_persists_across_calls(self):
        s = _stub_session()
        RUN_PYTHON.handler({"code": "x = 42", "thought": "set"}, s)
        out = RUN_PYTHON.handler({"code": "print(x * 2)", "thought": "read"}, s)
        self.assertIn("84", out.text)

    def test_no_output_message(self):
        s = _stub_session()
        out = RUN_PYTHON.handler({"code": "x = 1", "thought": "x"}, s)
        self.assertEqual(out.text, "(no output)")

    def test_keyboard_interrupt_returns_interrupted(self):
        s = _stub_session()
        out = RUN_PYTHON.handler(
            {"code": "raise KeyboardInterrupt()", "thought": "x"}, s)
        self.assertTrue(out.is_error)
        self.assertTrue(out.interrupted)
        self.assertEqual(out.text, "(interrupted)")


class TestLookupDocs(unittest.TestCase):
    """``lookup_docs`` reads SKILL.md bodies from the Session's skill registry.

    The handler accesses ``session.skills`` (a dict[name -> Skill]) which the
    real Session builds at __init__ from ``extra_skill_dirs`` + the core
    ``toksearch/skills/`` directory.  These tests stub that mapping.
    """

    def _stub_session(self, skills):
        return SimpleNamespace(skills=skills)

    def test_unknown_skill_is_error(self):
        from toksearch.llm.tools import LOOKUP_DOCS
        s = self._stub_session({})
        out = LOOKUP_DOCS.handler({"skill_name": "missing"}, s)
        self.assertTrue(out.is_error)
        self.assertIn("missing", out.text)

    def test_known_skill_returns_body(self):
        from toksearch.llm.tools import LOOKUP_DOCS, Skill
        s = self._stub_session({"foo": Skill(name="foo",
                                             description="d",
                                             body="Hello body.")})
        out = LOOKUP_DOCS.handler({"skill_name": "foo"}, s)
        self.assertFalse(out.is_error)
        self.assertEqual(out.text, "Hello body.")

    def test_lookup_docs_spec_shape(self):
        from toksearch.llm.tools import LOOKUP_DOCS
        self.assertEqual(LOOKUP_DOCS.name, "lookup_docs")
        self.assertIn("skill_name", LOOKUP_DOCS.input_schema["properties"])
        self.assertEqual(LOOKUP_DOCS.input_schema["required"], ["skill_name"])


class TestDiscoverSkills(unittest.TestCase):
    def test_returns_empty_for_nonexistent_dirs(self):
        from toksearch.llm.tools import discover_skills
        skills = discover_skills([Path("/nonexistent")])
        self.assertEqual(skills, {})

    def test_parses_skill_with_frontmatter(self):
        from toksearch.llm.tools import discover_skills, parse_skill_md
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            sk = root / "myskill"
            sk.mkdir()
            (sk / "SKILL.md").write_text(
                "---\nname: myskill\ndescription: My skill\n---\n\n"
                "Body content here.\n")
            skills = discover_skills([root])
            self.assertIn("myskill", skills)
            self.assertEqual(skills["myskill"].description, "My skill")
            self.assertIn("Body content here", skills["myskill"].body)

    def test_skips_dirs_without_skill_md(self):
        from toksearch.llm.tools import discover_skills
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "not_a_skill").mkdir()
            self.assertEqual(discover_skills([root]), {})

    def test_parse_skill_md_no_frontmatter(self):
        from toksearch.llm.tools import parse_skill_md
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md",
                                          delete=False) as f:
            f.write("Just body, no frontmatter.\n")
            path = Path(f.name)
        try:
            fm, body = parse_skill_md(path)
            self.assertEqual(fm, {})
            self.assertIn("Just body", body)
        finally:
            path.unlink()


if __name__ == "__main__":
    unittest.main()
