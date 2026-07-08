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

"""Regression guard: importing toksearch must not eagerly import the heavy
Ray and Spark backends. See
docs/superpowers/specs/2026-07-07-lazy-backend-imports-design.md.
"""

import json
import subprocess
import sys
import unittest


HEAVY_PREFIXES = ("ray", "pyspark")


def _heavy_modules_after(snippet):
    """Run `snippet` in a clean interpreter, return the sorted list of loaded
    ray/pyspark modules."""
    code = (
        f"{snippet}\n"
        "import sys, json\n"
        "loaded = sorted(\n"
        "    m for m in sys.modules\n"
        "    if m == 'ray' or m == 'pyspark'\n"
        "    or m.startswith('ray.') or m.startswith('pyspark.')\n"
        ")\n"
        "print(json.dumps(loaded))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"subprocess failed (rc={result.returncode}):\n{result.stderr}"
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


class TestLazyBackendImports(unittest.TestCase):
    def test_import_toksearch_does_not_import_ray_or_spark(self):
        heavy = _heavy_modules_after("import toksearch")
        self.assertEqual(
            heavy, [], f"`import toksearch` eagerly loaded: {heavy}"
        )

    def test_building_pipeline_does_not_import_ray_or_spark(self):
        heavy = _heavy_modules_after(
            "from toksearch import Pipeline\np = Pipeline([1, 2, 3])"
        )
        self.assertEqual(
            heavy, [], f"Building a Pipeline eagerly loaded: {heavy}"
        )
