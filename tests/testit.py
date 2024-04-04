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

#!/usr/bin/env python

import argparse
import unittest
import sys
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--noptdata", action="store_true")
    parser.add_argument("--nod3drdb", action="store_true")

    args = parser.parse_args()

    os.environ["TOKSEARCH_INTEGRATION"] = "no" if args.mock else "yes"

    os.environ["TOKSEARCH_PTDATA_TEST"] = "no" if args.noptdata else "yes"

    os.environ["TOKSEARCH_D3DRDB_TEST"] = "no" if args.nod3drdb else "yes"

    loader = unittest.TestLoader()
    runner = unittest.TextTestRunner(verbosity=2)

    test_dir = "."
    tests = loader.discover(test_dir)

    res = runner.run(tests)
    sys.exit(not res.wasSuccessful())
