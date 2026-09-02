# Copyright 2026 General Atomics
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

import os
import subprocess
import tempfile
import unittest

from toksearch.provenance.code import capture_code


def _init_repo(path):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    subprocess.run(["git", "init", "-q", path], check=True, env=env)
    with open(os.path.join(path, "f.txt"), "w") as fh:
        fh.write("one\n")
    subprocess.run(["git", "-C", path, "add", "f.txt"], check=True, env=env)
    subprocess.run(["git", "-C", path, "commit", "-qm", "init"], check=True, env=env)
    return env


class TestCaptureCode(unittest.TestCase):
    def test_returns_commit_in_a_repo(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            code = capture_code(cwd=d)
            self.assertEqual(len(code.commit), 40)

    def test_reports_clean_tree(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            self.assertFalse(capture_code(cwd=d).dirty)

    def test_reports_dirty_tree(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            with open(os.path.join(d, "f.txt"), "w") as fh:
                fh.write("two\n")
            self.assertTrue(capture_code(cwd=d).dirty)

    def test_records_repo_root(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            self.assertIsNotNone(capture_code(cwd=d).repo_root)

    def test_outside_a_repo_returns_none_commit_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            code = capture_code(cwd=d)
            self.assertIsNone(code.commit)

    def test_outside_a_repo_dirty_is_none_not_false(self):
        # None means "unknown", False means "known clean". Conflating them
        # would let a provenance record claim a clean tree it never saw.
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(capture_code(cwd=d).dirty)

    def test_records_argv(self):
        code = capture_code()
        self.assertIsInstance(code.argv, tuple)

    def test_is_json_serializable(self):
        from toksearch.provenance.hashing import canonical_json
        canonical_json(capture_code().to_dict())

    def test_to_dict_round_trips_every_field(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            code = capture_code(cwd=d)
            as_dict = code.to_dict()
            for field in ("commit", "dirty", "repo_root", "script", "argv"):
                self.assertIn(field, as_dict)

    def test_untracked_files_alone_count_as_dirty(self):
        # `git status --porcelain` lists untracked files, so a tree with only
        # untracked additions reads as dirty. That is the intended reading:
        # the run may depend on a file that is not in the recorded commit.
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            with open(os.path.join(d, "scratch.py"), "w") as fh:
                fh.write("x = 1\n")
            self.assertTrue(capture_code(cwd=d).dirty)

    def test_is_frozen(self):
        import dataclasses

        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            code = capture_code(cwd=d)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                code.commit = "nope"
