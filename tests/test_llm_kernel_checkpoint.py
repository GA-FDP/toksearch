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
import shutil
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
            self.assertEqual(sorted(p.name for p in d.glob("*.pkl")),
                             ["v0002.pkl"])
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
            (d / "v0001.pkl").write_bytes(b"not a pickle")
            report = ck.load_namespace({}, d)
            self.assertEqual(report["not_restored"][0]["reason"], "unloadable")
            self.assertEqual(report["not_restored"][0]["cell"], 4)
            self.assertEqual(report["not_restored"][0]["path"],
                             str((d / "v0001.pkl").resolve()))


class Fragile:
    """Unpickling raises while the marker file exists."""

    def __init__(self, marker, value):
        self.marker, self.value = str(marker), value

    def __setstate__(self, state):
        if os.path.exists(state["marker"]):
            raise RuntimeError("environment is broken")
        self.__dict__.update(state)


class TestCrashSafety(unittest.TestCase):
    def test_double_crash_restores_the_second_checkpoint_from_new(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            ck.save_namespace(_ns(x=1), BASELINE, {}, d)
            os.rename(d, d.with_name("namespace.old"))
            real_rename = os.rename

            def failing(src, dst, *a, **k):
                if Path(src).name == "namespace.new" and Path(dst) == d:
                    raise OSError("crash during swap")
                return real_rename(src, dst, *a, **k)

            with mock.patch.object(ck.os, "rename", side_effect=failing):
                with self.assertRaises(OSError):
                    ck.save_namespace(_ns(x=2), BASELINE, {}, d)
            self.assertTrue(d.with_name("namespace.old").exists())
            self.assertEqual(ck.find_checkpoint(d), d.with_name("namespace.new"))
            dst = {}
            self.assertTrue(ck.load_namespace(dst, d)["found"])
            self.assertEqual(dst["x"], 2)
            # The next save promotes nothing stale and ends clean.
            ck.save_namespace(_ns(x=3), BASELINE, {}, d)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["namespace"])
            dst = {}
            ck.load_namespace(dst, d)
            self.assertEqual(dst["x"], 3)

    def test_truncated_index_falls_back_to_old(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            ck.save_namespace(_ns(x=1), BASELINE, {}, d)
            old = d.with_name("namespace.old")
            keep = Path(tmp) / "keep"
            shutil.copytree(d, keep)
            ck.save_namespace(_ns(x=2), BASELINE, {}, d)
            # The state a crash leaves mid-write: .old still present and the
            # index in namespace cut short.
            shutil.copytree(keep, old)
            text = (d / "index.json").read_text()
            (d / "index.json").write_text(text[: len(text) // 2])
            self.assertEqual(ck.find_checkpoint(d), old)
            dst = {}
            ck.load_namespace(dst, d)
            self.assertEqual(dst["x"], 1)

    def test_hostile_names_round_trip_inside_the_folder(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp) / "conv"
            root.mkdir()
            d = root / "namespace"
            names = {"../x": 1, "a/b": 2, "A": 3, "a": 4}
            ck.save_namespace(_ns(**names), BASELINE, {}, d)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()),
                             ["conv"])
            self.assertEqual(sorted(p.name for p in root.iterdir()),
                             ["namespace"])
            self.assertEqual(sorted(p.name for p in d.iterdir()),
                             ["index.json", "v0001.pkl", "v0002.pkl",
                              "v0003.pkl", "v0004.pkl"])
            dst = {}
            report = ck.load_namespace(dst, d)
            self.assertEqual(report["not_restored"], [])
            self.assertEqual({k: dst[k] for k in names}, names)


class TestCarryForward(unittest.TestCase):
    def test_unloadable_variable_is_carried_and_loads_once_fixed(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            marker = Path(tmp) / "broken"
            ck.save_namespace(_ns(frag=Fragile(marker, 99), ok=1),
                              BASELINE, {"frag": 3}, d)
            marker.touch()
            dst = {}
            report = ck.load_namespace(dst, d)
            self.assertEqual(report["restored"], ["ok"])
            bad = report["not_restored"][0]
            self.assertEqual((bad["name"], bad["reason"]),
                             ("frag", "unloadable"))
            carried = {"frag": (bad["path"], bad)}
            index = ck.save_namespace(_ns(**dst), BASELINE, {"ok": 1}, d,
                                      carried=carried)
            frag = next(e for e in index["variables"] if e["name"] == "frag")
            self.assertEqual(frag["status"], "saved")
            self.assertTrue(frag["carried"])
            self.assertEqual(frag["cell"], 3)
            self.assertEqual(frag["type"], "Fragile")
            marker.unlink()
            dst2 = {}
            report = ck.load_namespace(dst2, d)
            self.assertEqual(sorted(report["restored"]), ["frag", "ok"])
            self.assertEqual(dst2["frag"].value, 99)

    def test_carried_name_rebound_by_the_user_is_not_copied(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "namespace"
            ck.save_namespace(_ns(v=1), BASELINE, {}, d)
            entry = {"name": "v", "type": "int", "cell": 1}
            index = ck.save_namespace(_ns(v=5), BASELINE, {}, d,
                                      carried={"v": (str(d / "v0001.pkl"),
                                                     entry)})
            self.assertEqual(len(index["variables"]), 1)
            self.assertNotIn("carried", index["variables"][0])


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
