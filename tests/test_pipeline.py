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
import ray

import tempfile

from toksearch.pipeline.pipeline_source import PipelineSource
from toksearch import Pipeline

from toksearch.record import Record
from toksearch.signal.mock_signal import MockSignal

from toksearch.backend.ray.raydd import RayDD, BadFunctionError


class TestPipelineSource(unittest.TestCase):

    def test_with_list_of_shots(self):
        shots = [1, 2, 3]
        source = PipelineSource(shots)

        for rec in source.records:
            self.assertIsInstance(rec, Record)

    def test_with_list_of_dicts(self):
        dicts = [
            {"shot": 1, "blah": 4},
            {"shot": 2, "blah": 8},
            {"shot": 3, "blah": 16},
        ]

        source = PipelineSource(dicts)

        for rec in source.records:
            self.assertIsInstance(rec, Record)

        self.assertEqual(source.records[0]["blah"], 4)

    def test_with_mixed_list(self):
        items = [{"shot": 1, "blah": 4}, 2, Record(3)]

        source = PipelineSource(items)

        for rec in source.records:
            self.assertIsInstance(rec, Record)

        self.assertEqual(source.records[0]["shot"], 1)
        self.assertEqual(source.records[0]["blah"], 4)

        self.assertEqual(source.records[1]["shot"], 2)
        self.assertEqual(source.records[2]["shot"], 3)


def idempotent_ray_init():
    if not sys.warnoptions:
        import warnings

        warnings.simplefilter("ignore")
    if not ray.is_initialized():
        temp_dir = tempfile.mkdtemp()
        ray.init(num_cpus=5, _temp_dir=temp_dir)


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        idempotent_ray_init()

    @classmethod
    def setUp(cls):
        cls.shots = [1, 2, 3]
        cls.shotlist = PipelineSource([1])
        cls.pipeline = Pipeline(cls.shotlist)

    def test_fetch(self):
        p = self.pipeline
        sig = MockSignal()
        p.fetch("my_sig", sig)
        record = p.compute_shot(1234)
        self.assertFalse(record.errors)
        self.assertEqual(len(record["my_sig"]["data"]), len(MockSignal.default_d))

        self.assertEqual(len(record["my_sig"]["times"]), len(MockSignal.default_t))

    def test_map(self):
        p = self.pipeline
        sig = MockSignal()
        p.fetch("my_sig", sig)

        @p.map
        def f(rec):
            rec["blah"] = "some_val"

        record = p.compute_shot(1234)

    def _create_pipeline(self):
        shots = [1, 2, 3]
        p = Pipeline(shots)
        sig = MockSignal()
        p.fetch("my_sig", sig)
        p.fetch_dataset("ds", {"s1": sig})

        @p.map
        def f(rec):
            rec["aardvark"] = rec.shot

        @p.where
        def where(rec):
            return rec.shot < 3

        return shots, p

    def test_compute_serial(self):
        shots, p = self._create_pipeline()
        recs = p.compute_serial()

        self.assertEqual(len(recs), len(shots) - 1)
        for i, rec in enumerate(recs):
            self.assertEqual(rec["aardvark"], i + 1)

    def test_continued_pipeline(self):

        shots = list(range(0, 10))
        parent = Pipeline(shots)

        @parent.map
        def blah(rec):
            rec["blah"] = rec["shot"] * 2

        child = Pipeline(parent)
        child.keep([])

        parent_results = parent.compute_serial()
        child_results = child.compute_serial()

        self.assertEqual(len(parent_results), len(child_results))

        p_r0 = parent_results[0]
        c_r0 = child_results[0]

        self.assertIn("blah", p_r0)
        self.assertNotIn("blah", c_r0)

    def test_compute_ray(self):
        shots, p = self._create_pipeline()
        recs = p.compute_ray()

        self.assertEqual(len(recs), len(shots) - 1)
        for i, rec in enumerate(recs):
            self.assertEqual(rec["aardvark"], i + 1)

    def test_ray_continued_pipeline(self):
        shots = list(range(0, 10))
        parent = Pipeline(shots)

        @parent.map
        def blah(rec):
            rec["blah"] = rec["shot"] * 2

        child = Pipeline(parent)
        results = child.compute_ray()
        num_results = len(results)
        self.assertEqual(num_results, len(shots))
        self.assertEqual(results[-1]["blah"], shots[-1] * 2)

    def test_keep_with_ray(self):

        # This test is in response to a bug caused by
        # accidentally running ray tasks twice. Calling
        # keep and throwing away a value obtained from
        # an sql query or a dictionary that was used in
        # a map function would cause a KeyError
        pipe = Pipeline([{"shot": 1, "dummy": 2}])

        @pipe.map
        def use_data(rec):
            rec["a"] = rec["dummy"] * 2

        pipe.keep([])

        @pipe.where
        def no_errors(rec):
            return not rec.errors

        results = pipe.compute_ray()

        self.assertEqual(len(results), 1)

    def test_compute_spark(self):
        shots, p = self._create_pipeline()
        recs = p.compute_spark()

        self.assertEqual(len(recs), len(shots) - 1)
        for i, rec in enumerate(recs):
            self.assertEqual(rec["aardvark"], i + 1)

    def test_spark_continued_pipeline(self):
        shots = list(range(0, 10))
        parent = Pipeline(shots)

        @parent.map
        def blah(rec):
            rec["blah"] = rec["shot"] * 2

        child = Pipeline(parent)

        @child.map
        def child_blah(rec):
            rec["child_blah"] = rec["shot"] * 3

        results = child.compute_spark()
        num_results = len(results)
        self.assertEqual(num_results, len(shots))
        self.assertEqual(results[-1]["blah"], shots[-1] * 2)
        self.assertEqual(results[-1]["child_blah"], shots[-1] * 3)

    def test_spark_continued_from_result(self):
        shots = list(range(0, 10))
        parent = Pipeline(shots)

        @parent.map
        def blah(rec):
            rec["blah"] = rec["shot"] * 2

        parent_results = parent.compute_spark()

        child = Pipeline(parent_results)

        @child.map
        def child_blah(rec):
            rec["child_blah"] = rec["shot"] * 3

        results = child.compute_spark()
        num_results = len(results)
        self.assertEqual(num_results, len(shots))
        self.assertEqual(results[-1]["blah"], shots[-1] * 2)
        self.assertEqual(results[-1]["child_blah"], shots[-1] * 3)

        self.assertNotIn("child_blah", parent_results[0])

    def test_serial_continued_pipeline(self):
        shots = list(range(0, 10))
        parent = Pipeline(shots)

        @parent.map
        def blah(rec):
            rec["blah"] = rec["shot"] * 2

        child = Pipeline(parent)
        results = child.compute_spark()
        num_results = len(results)
        self.assertEqual(num_results, len(shots))
        self.assertEqual(results[-1]["blah"], shots[-1] * 2)

    def test_batch_size_ok(self):
        batch_size = 1

        shots, p = self._create_pipeline()

        recs = p.compute_ray(batch_size=batch_size)

        self.assertEqual(len(recs), len(shots) - 1)

    def test_multiporcessing_backend(self):
        shots, p = self._create_pipeline()
        recs = p.compute_multiprocessing()
        self.assertEqual(len(recs), len(shots) - 1)
        for i, rec in enumerate(recs):
            self.assertEqual(rec["aardvark"], i + 1)


class TestRayRecordSet(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shots, cls.p = cls._create_pipeline()
        ray_init_kwargs = {"num_cpus": 5}
        cls.res = cls.p.compute_ray(ray_init_kwargs=ray_init_kwargs)

    @classmethod
    def _create_pipeline(cls):
        shots = list(range(0, 100))
        p = Pipeline(shots)
        sig = MockSignal()
        p.fetch("my_sig", sig)

        @p.map
        def f(rec):
            rec["aardvark"] = rec.shot

        @p.where
        def where(rec):
            return rec.shot < 50

        return shots, p

    def test_len(self):
        self.assertEqual(len(self.res), 50)

    def test_getitem(self):
        item = self.res[0]
        self.assertIsInstance(item, Record)
        self.assertEqual(item["shot"], 0)
        self.assertIn("aardvark", item)

    def test_iter(self):
        for i, item in enumerate(self.res):
            self.assertIsInstance(item, Record)
            self.assertIn("aardvark", item)

    def test_slice(self):
        items = self.res[4:6]
        self.assertEqual(len(items), 2)

    def test_object_ids(self):
        ids = self.res.object_ids()
        self.assertEqual(len(ids), 50)
        for item in ids:
            self.assertIsInstance(item, ray.ObjectID)

    def test_continue_from_result(self):
        shots, p = self._create_pipeline()
        results = p.compute_ray()
        child = Pipeline(results)

        @child.map
        def blah(rec):
            rec["blah"] = "blah"

        child_results = child.compute_ray()

        p_r0 = results[0]
        c_r0 = child_results[0]

        self.assertIn("blah", c_r0)
        self.assertNotIn("blah", p_r0)

        self.assertEqual(len(results), len(child_results))

    def test_with_memory_per_shot(self):
        shots, p = self._create_pipeline()

        results = p.compute_ray(memory_per_shot=int(100e6))
        self.assertEqual(len(results), 50)


def good_func(x):
    res = [el * 2 for el in x if el > 0]
    return res


def bad_func(x):
    return 123


class TestRayDD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        idempotent_ray_init()

        cls.elements = [0, 1, 2, 3, 4, 5]
        # cls.raydd = RayDD.from_iterator(cls.elements)

    def _test_map(self, func=good_func, batch_size=None, numparts=None):
        raydd = RayDD.from_iterator(
            self.elements, batch_size=batch_size, numparts=numparts
        )
        results = raydd.map(func).get()
        self.assertEqual(len(results), len(self.elements) - 1)
        self.assertEqual(results[0], 2)

    def test_map_with_defaults(self):
        self._test_map()

    def test_map_with_batch_size(self):
        self._test_map(batch_size=3)

    def test_map_with_numparts(self):
        self._test_map(numparts=2)

    def test_map_with_numparts_and_batch_size(self):
        self._test_map(numparts=2, batch_size=1)

    def test_map_with_bad_func(self):
        raydd = RayDD.from_iterator(self.elements)
        with self.assertRaises(BadFunctionError):
            results = raydd.map(bad_func).get()
