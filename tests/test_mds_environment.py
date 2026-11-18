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

import json
import os
import subprocess
import sys
import unittest

_PROBE = (
    "import json, os, toksearch.signal.mds; "
    "print(json.dumps({k: os.environ.get(k) "
    "for k in ('MDSPLUS_DIR', 'DYLD_LIBRARY_PATH')}))"
)


def _environment_after_import(**overrides):
    """Import toksearch.signal.mds in a fresh interpreter; return the two variables it may touch."""
    env = dict(os.environ)
    env.pop("MDSPLUS_DIR", None)
    env.pop("DYLD_LIBRARY_PATH", None)
    env.update(overrides)
    out = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


class TestMdsplusEnvironment(unittest.TestCase):
    def test_mdsplus_dir_set_from_prefix_when_tables_exist(self):
        tables = os.path.join(sys.prefix, "xml", "mdsdcl_commands.xml")
        expected = sys.prefix if os.path.exists(tables) else None
        self.assertEqual(_environment_after_import()["MDSPLUS_DIR"], expected)

    def test_explicit_mdsplus_dir_is_kept(self):
        result = _environment_after_import(MDSPLUS_DIR="/custom/mdsplus")
        self.assertEqual(result["MDSPLUS_DIR"], "/custom/mdsplus")

    @unittest.skipUnless(sys.platform == "darwin", "MDSplus sets it only on macOS")
    def test_no_dyld_library_path_left_for_subprocesses(self):
        self.assertIsNone(_environment_after_import()["DYLD_LIBRARY_PATH"])

    @unittest.skipUnless(sys.platform == "darwin", "MDSplus sets it only on macOS")
    def test_explicit_dyld_library_path_is_kept(self):
        result = _environment_after_import(DYLD_LIBRARY_PATH="/custom/lib")
        self.assertEqual(result["DYLD_LIBRARY_PATH"], "/custom/lib")


if __name__ == "__main__":
    unittest.main()
