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

from toksearch.library.ell1 import compute_ell1


class TestEll1(unittest.TestCase):

    def _ramp_signal(self):
        x = np.zeros(1000)
        x[500:] = np.arange(500) / 500
        return x

    def test_happy_path(self):
        x = self._ramp_signal()
        y = compute_ell1(x, 1)["y"]

        np.testing.assert_allclose(x, y, atol=1e-2)

    def test_multi_x(self):
        x = self._ramp_signal()
        x_nd = np.tile(x[:, None], (1, 5))
        y = compute_ell1(x_nd, 0.2)["y"]
        self.assertEqual(x_nd.shape, y.shape)

    def test_empty_x(self):
        x = self._ramp_signal()
        with self.assertRaises(Exception):
            y = compute_ell1(None, 0.2)["y"]

    def test_reasonable_yl(self):
        x = self._ramp_signal()
        x[0] = 0.1
        y = compute_ell1(x, 1, yl=0.0)["y"]

        np.testing.assert_allclose(y[0], 0, atol=1e-2)

    def test_bad_yl(self):
        x = self._ramp_signal()
        with self.assertRaises(Exception):
            y = compute_ell1(x, 0.2, yl=dict())["y"]

    def test_bad_yl(self):
        x = self._ramp_signal()
        with self.assertRaises(Exception):
            y = compute_ell1(x, 0.2, yr=dict())["y"]

    def test_reasonable_yr(self):
        x = self._ramp_signal()

        orig_xr = x[-1]
        x[-1] = orig_xr + 0.2
        y = compute_ell1(x, 1, yr=orig_xr)["y"]

        np.testing.assert_allclose(y[-1], orig_xr, atol=1e-2)

    def test_reasonable_eta(self):
        x = self._ramp_signal()
        y = compute_ell1(x, 1, eta=0.2)["y"]

        # Not sure what to test here. Just running
        # without crashing is good enough for now I guess.

    def test_reasonable_eps(self):
        x = self._ramp_signal()
        y = compute_ell1(x, 1, eps=0.2)["y"]

        # Not sure what to test here. Just running
        # without crashing is good enough for now I guess.

    def test_reasonable_maxiters(self):
        x = self._ramp_signal()
        y = compute_ell1(x, 0.5, maxiters=100)["y"]

    def test_num_threads(self):
        x = self._ramp_signal()
        y = compute_ell1(x, 0.5, nthreads=5)["y"]
