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

from __future__ import print_function
import unittest
import numpy as np

# from toksearch import ShotPipeline
# from toksearch import XarrayShotData

from toksearch import Pipeline
from toksearch import Signal
from toksearch import XarrayAligner
from toksearch.pipeline.dataset import merge_val

from toksearch.signal.mock_signal import MockSignal

# class MockSignal(Signal):
#    #TODO: Make this actually inherit from signal
#    default_d = np.arange(4)
#    default_t = np.arange(4)
#
#    def __init__(self, data, times, dims=('times',)):
#        self.data = data
#        self.times = times
#        self.dims = dims
#
#
#    def fetch(self, shot):
#        return {'data': self.data, 'times': self.times}
#
#    def fetch_as_xarray(self, shot):
#        import xarray as xr
#        signal_as_dict = self.fetch(shot)
#        d = signal_as_dict['data']
#        coords = {}
#        dims = ['times']
#        for dim in dims:
#            coords[dim] = signal_as_dict[dim]
#
#        results = xr.DataArray(d, dims=dims, coords=coords)
#        return results
#
#    def cleanup(self):
#        pass
#
#    def cleanup_shot(self, shot):
#        pass


class TestTimebaseAligner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shot = 165920

    @classmethod
    def setUp(self):
        self.pipeline = Pipeline([self.shot])
        t1 = np.arange(10)
        d1 = t1[:]
        sig1 = MockSignal(d1, t1)

        t2 = t1 + 0.4
        d2 = t2[:]
        sig2 = MockSignal(d2, t2)

        self.t1, self.t2, self.d1, self.d2 = t1, t2, d1, d2
        self.pipeline.fetch_dataset("ds", {"s1": sig1, "s2": sig2})

    def _align(self, align_with, method, extrapolate):
        # aligner = XarrayAligner(align_with,
        #                        method=method,
        #                        extrapolate=extrapolate)
        # def alignit(rec):
        #    rec['ds'] = aligner(rec['ds'])
        #    return rec

        # self.pipeline.map(alignit)
        self.pipeline.align("ds", align_with, method=method, extrapolate=extrapolate)

    def test_with_string_input_linear(self):
        self._align("s1", "linear", True)
        res = self.pipeline.compute_shot(self.shot)
        ds = res["ds"]
        self.assertTrue(np.all(ds.s1 == ds.s2))
        self.assertFalse(res.errors)

    def test_with_ndarray_input_linear(self):
        timebase = np.arange(10) + 10.0
        self._align(timebase, "linear", True)
        res = self.pipeline.compute_shot(self.shot)
        ds = res["ds"]
        self.assertTrue(np.all(ds.s1 == ds.s2))
        self.assertFalse(res.errors)

    def test_with_string_input_pad(self):
        self._align("s1", "pad", True)
        res = self.pipeline.compute_shot(self.shot)
        ds = res["ds"]
        self.assertEquals(self.d2[-2], ds["s2"].values[-1])
        self.assertFalse(res.errors)

    def test_with_ndarray_input_pad(self):
        timebase = np.arange(10) + 10.0
        self._align(timebase, "pad", True)
        res = self.pipeline.compute_shot(self.shot)
        ds = res["ds"]
        self.assertTrue(self.d2[-1], ds["s2"].values[-1])
        self.assertFalse(res.errors)

    def test_linear_with_no_dim(self):

        # This will create a field with no times dim
        @self.pipeline.map
        def update_ds(rec):
            rec["ds"] = merge_val(rec["ds"], "x", 1)
            return rec

        self._align("s1", "linear", True)
        res = self.pipeline.compute_shot(self.shot)
        self.assertFalse(res.errors)

    def test_pad_with_no_dim(self):

        # This will create a field with no times dim
        @self.pipeline.map
        def update_ds(rec):
            rec["ds"] = merge_val(rec["ds"], "x", 1)
            return rec

        self._align("s1", "pad", True)
        res = self.pipeline.compute_shot(self.shot)
        self.assertFalse(res.errors)

    def test_with_numeric_input_pad(self):
        orig_ds = self.pipeline.compute_shot(self.shot)
        self._align(2.5, "pad", True)
        res = self.pipeline.compute_shot(self.shot)
        ds = res["ds"]
        dt = ds.times.diff("times").values[0]
        self.assertEqual(dt, 2.5)

    # def test_with_list_input(self):
    #    pipeline = _create_pipeline(pyfics.FicsXarrayPipeline)
    #    pipeline.fetch('ip_ptdata',  pyfics.PtDataSignal('ip'))
    #    aligner = pyfics.Aligner(range(4000), method='linear')
    #    pipeline.modify(aligner)
    #    res = pipeline.compute_shot(self.shot)
    #    self.assertFalse(res.errors)

    # def test_with_ndarray_input(self):
    #    pipeline = _create_pipeline(pyfics.FicsXarrayPipeline)
    #    pipeline.fetch('ip_ptdata',  pyfics.PtDataSignal('ip'))
    #    aligner = pyfics.Aligner(np.arange(4000), method='linear')
    #    pipeline.modify(aligner)
    #    res = pipeline.compute_shot(self.shot)
    #    self.assertFalse(res.errors)
