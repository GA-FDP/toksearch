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

"""Tests for per-variable cloudpickle checkpoints of a namespace."""

import json
import os
import threading
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from toksearch.llm.kernel import checkpoint as ck


def _ns(**user):
    ns = {"np": np, "preset": 1, "__builtins__": {}}
    ns.update(user)
    return ns


BASELINE = frozenset({"np", "preset", "__builtins__"})


class TestUserNames(unittest.TestCase):
    def test_excludes_baseline_dunders_and_modules(self):
        ns = _ns(x=1, os_mod=os, _private=2)
        self.assertEqual(ck.user_names(ns, BASELINE), ["_private", "x"])


class TestRoundTrip(unittest.TestCase):
    def test_values_functions_and_origins_survive(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            src = _ns(arr=np.arange(5.0), cfg={"shot": 30420},
                      f=lambda v: v + 1)
            with mock.patch.dict(os.environ, {"FDP_CORE_VERSION": "1.40.0"}):
                index = ck.save_namespace(src, BASELINE, {"arr": 3}, d)
            self.assertEqual(index["fdp_core_version"], "1.40.0")
            self.assertEqual({e["name"]: e["status"]
                              for e in index["variables"]},
                             {"arr": "saved", "cfg": "saved", "f": "saved"})
            dst = {}
            report = ck.load_namespace(dst, d)
            self.assertTrue(report["found"])
            self.assertEqual(report["restored"], ["arr", "cfg", "f"])
            self.assertEqual(report["origins"], {"arr": 3})
            np.testing.assert_array_equal(dst["arr"], np.arange(5.0))
            self.assertEqual(dst["f"](1), 2)

    def test_missing_checkpoint(self):
        with TemporaryDirectory() as tmp:
            report = ck.load_namespace({}, Path(tmp) / "namespace")
            self.assertFalse(report["found"])
            self.assertIsNone(ck.read_index(Path(tmp) / "namespace"))


class TestLimitsAndFailures(unittest.TestCase):
    def test_unpicklable_is_recorded_with_its_cell(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            index = ck.save_namespace(_ns(lock=threading.Lock(), ok=1),
                                      BASELINE, {"lock": 2}, d)
            lock = next(e for e in index["variables"] if e["name"] == "lock")
            self.assertEqual(lock["status"], "unpicklable")
            self.assertIn("lock", lock["error"])
            self.assertFalse((d / "lock.pkl").exists())
            report = ck.load_namespace({}, d)
            self.assertEqual(report["restored"], ["ok"])
            self.assertEqual(report["not_restored"],
                             [{"name": "lock", "reason": "unpicklable",
                               "cell": 2, "error": lock["error"]}])

    def test_per_variable_limit(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            index = ck.save_namespace(
                _ns(big=np.zeros(10_000), small=1), BASELINE, {}, d,
                per_var_limit=1_000)
            status = {e["name"]: e["status"] for e in index["variables"]}
            self.assertEqual(status, {"big": "too_large", "small": "saved"})

    def test_total_limit(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            index = ck.save_namespace(
                _ns(a=np.zeros(1_000), b=np.zeros(1_000)), BASELINE, {}, d,
                total_limit=10_000)
            status = {e["name"]: e["status"] for e in index["variables"]}
            self.assertEqual(status, {"a": "saved", "b": "over_total"})

    def test_failed_write_keeps_previous_checkpoint(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            ck.save_namespace(_ns(x=1), BASELINE, {}, d)
            with mock.patch.object(ck, "_write_index",
                                   side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    ck.save_namespace(_ns(x=2), BASELINE, {}, d)
            dst = {}
            ck.load_namespace(dst, d)
            self.assertEqual(dst["x"], 1)

    def test_old_copy_is_used_after_crash_between_renames(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            ck.save_namespace(_ns(x=1), BASELINE, {}, d)
            d.rename(d.with_name("namespace.old"))
            dst = {}
            self.assertTrue(ck.load_namespace(dst, d)["found"])
            self.assertEqual(dst["x"], 1)

    def test_unloadable_pickle(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            ck.save_namespace(_ns(x=1), BASELINE, {"x": 4}, d)
            (d / "x.pkl").write_bytes(b"not a pickle")
            report = ck.load_namespace({}, d)
            self.assertEqual(report["not_restored"][0]["reason"], "unloadable")
            self.assertEqual(report["not_restored"][0]["cell"], 4)


class TestDescribe(unittest.TestCase):
    def test_describe_and_not_saved(self):
        index = {"variables": [
            {"name": "df", "status": "too_large", "cell": 7},
            {"name": "conn", "status": "unpicklable", "cell": 2,
             "error": "TypeError: x"},
            {"name": "ok", "status": "saved", "cell": 1}]}
        missing = ck.not_saved(index)
        self.assertEqual([m["name"] for m in missing], ["df", "conn"])
        self.assertEqual(
            ck.describe_not_restored(missing),
            "`df` (too large to save; bound by cell 7), "
            "`conn` (could not be saved; bound by cell 2)")


if __name__ == "__main__":
    unittest.main()
