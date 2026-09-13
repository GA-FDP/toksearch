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

import unittest
import numpy as np
import sys

import time
import tempfile

from toksearch.signal.signal import Signal
from toksearch.signal.mock_signal import MockSignal

from toksearch.record import Record
from toksearch.record import InvalidShotNumber
from toksearch.record import InvalidRecordField
from toksearch.record import MissingShotNumber


class TestRecord(unittest.TestCase):

    def test_from_dict_using_valid_dict(self):
        input_dict = {"shot": 1234, "blah": "abc"}
        rec = Record.from_dict(input_dict)

        self.assertEqual(rec["shot"], 1234)
        self.assertEqual(rec["blah"], "abc")

    def test_from_dict_without_shot(self):
        input_dict = {"blah": "abc"}
        self.assertRaises(MissingShotNumber, Record.from_dict, input_dict)

    def test_from_with_dict_with_key(self):
        input_dict = {"shot": 1234, "key": "dummy"}
        self.assertRaises(InvalidRecordField, Record.from_dict, input_dict)

    def test_from_with_dict_with_errors(self):
        input_dict = {"shot": 1234, "errors": "dummy"}
        self.assertRaises(InvalidRecordField, Record.from_dict, input_dict)

    def test_record_with_shot_number_as_string(self):
        self.assertRaises(InvalidShotNumber, Record, "abc")

    def test_record_with_shot_number_as_dict(self):
        self.assertRaises(InvalidShotNumber, Record, {})

    def test_set_val(self):
        rec = Record(1234)
        rec["abc"] = 1

    def test_keep(self):
        rec = Record(1)
        rec["a"] = 1
        rec["b"] = "blah"

        rec.keep(["a"])
        self.assertIn("a", rec)
        self.assertNotIn("b", rec)
        self.assertIn("shot", rec)
        self.assertIn("errors", rec)

    def test_discard(self):
        rec = Record(1)
        for key in ("a", "b", "c"):
            rec[key] = key
        rec.discard(["a", "b", "shot"])
        self.assertNotIn("a", rec)
        self.assertNotIn("b", rec)
        self.assertIn("c", rec)

        # We tried to discard shot, but it won't let us
        # (on purpose)
        self.assertIn("shot", rec)


class TestReservedProvenanceFields(unittest.TestCase):
    """version/snapshot as reserved Record fields.

    These live in a TestCase, not as module-level test_* functions: the
    suite is collected by unittest.TestLoader().discover() (see
    tests/testit.py), which is also what the conda recipe runs. Bare
    functions import cleanly and are then silently never executed.
    """

    def test_version_and_snapshot_survive_keep(self):
        """Provenance a routine keep() discards is provenance a reproduction
        cannot rely on, so these join shot and errors."""
        r = Record.from_dict({"shot": 165920, "version": 2,
                              "snapshot": "catalog_x", "peak": 1.5})
        r.keep(["peak"])
        self.assertEqual(r["version"], 2)
        self.assertEqual(r["snapshot"], "catalog_x")
        self.assertEqual(r["shot"], 165920)
        self.assertEqual(r["peak"], 1.5)

    def test_version_and_snapshot_cannot_be_deleted(self):
        """Both routes must refuse. pop() and __delitem__ are separate code
        paths and only pop() guarded anything before this."""
        r = Record.from_dict({"shot": 165920, "version": 2})
        r.pop("version")
        self.assertEqual(r["version"], 2)
        with self.assertRaises(InvalidRecordField):
            del r["version"]
        self.assertEqual(r["version"], 2)
        # Attribute deletion is the natural dual of item deletion in a class
        # whose fields ARE attributes, and it bypassed every guard until it
        # was closed.
        with self.assertRaises(InvalidRecordField):
            del r.version
        self.assertEqual(r["version"], 2)

    def test_shot_cannot_be_deleted_either(self):
        """Pre-existing hole: __delitem__ bypassed the guard pop() applies, so
        `del record["shot"]` succeeded while `record.pop("shot")` did not."""
        r = Record.from_dict({"shot": 165920})
        with self.assertRaises(InvalidRecordField):
            del r["shot"]
        with self.assertRaises(InvalidRecordField):
            del r.shot
        self.assertEqual(r["shot"], 165920)

    def test_pop_returns_the_value_it_removed(self):
        """Declared `-> Any` with no return, so it yielded None for every
        key."""
        r = Record.from_dict({"shot": 165920, "scratch": 42})
        self.assertEqual(r.pop("scratch"), 42)
        self.assertNotIn("scratch", r)
        # NOT `assertIsNone(r.pop("shot"))` alone: the old broken pop had no
        # return statement, so that passed against it too. Assert the guard's
        # real effect -- the field survives.
        self.assertIsNone(r.pop("shot"))
        self.assertEqual(r["shot"], 165920)

    def test_a_user_may_still_supply_version_and_snapshot(self):
        """Optional-known, NOT forbidden. Reserving them the way key and
        errors are reserved would reject the very shot list that seeds a
        pinned run."""
        r = Record.from_dict({"shot": 165920, "version": 2,
                              "snapshot": "catalog_x"})
        self.assertEqual(r["version"], 2)
        self.assertEqual(r["snapshot"], "catalog_x")
