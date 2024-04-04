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
