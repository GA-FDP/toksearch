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
import sys
import os
import inspect
from abc import ABCMeta, abstractmethod
import MDSplus as mds
import numpy as np

from toksearch.signal import Signal
from toksearch.signal.mock_signal import MockSignal
from toksearch.signal.signal import SignalRegistry


class TestSignalRegistry(unittest.TestCase):

    def test_register(self):
        reg = SignalRegistry()
        sig = MockSignal()
        reg.register(sig)
        self.assertIn(sig, reg)

    def test_register_multiple(self):
        reg = SignalRegistry()
        sig = MockSignal()
        sig2 = MockSignal()
        reg.register(sig)
        reg.register(sig2)
        self.assertIn(sig, reg)
        self.assertIn(sig2, reg)

    def test_is_singleton(self):
        reg = SignalRegistry()
        reg2 = SignalRegistry()
        self.assertIs(reg, reg2)

    def test_cleanup(self):
        reg = SignalRegistry()
        sig = MockSignal()
        reg.register(sig)
        reg.cleanup()
        self.assertNotIn(sig, reg)

    def test_cleanup_multiple(self):
        reg = SignalRegistry()
        sig = MockSignal()
        sig2 = MockSignal()
        reg.register(sig)
        reg.register(sig2)
        reg.cleanup()
        self.assertNotIn(sig, reg)
        self.assertNotIn(sig2, reg)

    def test_cleanup_shot(self):
        reg = SignalRegistry()
        sig = MockSignal()
        reg.register(sig)
        reg.cleanup_shot(1234)
        # Should not raise an error
        self.assertIn(sig, reg)

    def test_register_same_signal_twice(self):
        reg = SignalRegistry()
        reg.reset()
        sig = MockSignal()
        reg.register(sig)
        reg.register(sig)
        self.assertIn(sig, reg)
        self.assertEqual(len(reg.signals), 1)


class TestDimensionedSignal(unittest.TestCase):

    def signal(self, **kwargs):
        sig = MockSignal(**kwargs)
        return sig

    def test_fetch(self):
        sig = self.signal()
        shot = 1234
        res = sig.fetch(shot)
        for key, val in list(res.items()):
            self.assertGreater(len(val), 0)
        self.assertIn("units", res)

    def test_fetch_no_units(self):
        sig = self.signal(with_units=False)

    def test_fetch_with_callback(self):
        sig = self.signal().set_callback(lambda _: {"data": 1234})
        shot = 1234
        res = sig.fetch(shot)
        self.assertEqual(res["data"], 1234)

    def test_fetch_no_times(self):
        sig = self.signal(dims=None)
        shot = 1234
        res = sig.fetch(shot)
        self.assertGreater(len(res["data"]), 0)
        self.assertFalse("times" in res)
