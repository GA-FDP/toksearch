# Copyright 2026 General Atomics
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

import os
import tempfile
import unittest

import pandas as pd

from toksearch import Pipeline
from toksearch.signal.mock_signal import MockSignal
from toksearch.provenance import JsonProvenance


def _add_scalar(rec):
    rec["peak"] = float(max(rec["ip"]["data"]))


def _pipeline():
    pipeline = Pipeline([1, 2, 3])
    pipeline.fetch("ip", MockSignal())
    pipeline.map(_add_scalar)
    pipeline.keep(["peak"])
    return pipeline


class TestToDataFrame(unittest.TestCase):
    def test_returns_a_dataframe(self):
        self.assertIsInstance(_pipeline().compute_serial().to_dataframe(), pd.DataFrame)

    def test_has_one_row_per_record(self):
        self.assertEqual(len(_pipeline().compute_serial().to_dataframe()), 3)

    def test_includes_the_shot_column(self):
        df = _pipeline().compute_serial().to_dataframe()
        self.assertEqual(sorted(df["shot"]), [1, 2, 3])

    def test_shot_is_the_first_column(self):
        df = _pipeline().compute_serial().to_dataframe()
        self.assertEqual(list(df.columns)[0], "shot")

    def test_includes_kept_fields(self):
        self.assertIn("peak", _pipeline().compute_serial().to_dataframe().columns)

    def test_excludes_the_errors_field_by_default(self):
        # errors survives keep(), so this exclusion is doing real work.
        self.assertNotIn("errors", _pipeline().compute_serial().to_dataframe().columns)

    def test_explicit_fields_selection(self):
        df = _pipeline().compute_serial().to_dataframe(fields=["peak"])
        self.assertEqual(list(df.columns), ["shot", "peak"])

    def test_empty_recordset_gives_an_empty_frame(self):
        pipeline = Pipeline([1])
        pipeline.where(lambda rec: False)
        self.assertEqual(len(pipeline.compute_serial().to_dataframe()), 0)

    def test_empty_recordset_still_has_a_shot_column(self):
        pipeline = Pipeline([1])
        pipeline.where(lambda rec: False)
        self.assertIn("shot", pipeline.compute_serial().to_dataframe().columns)

    def test_write_bookkeeping_field_is_excluded(self):
        pipeline = Pipeline([1, 2])
        pipeline.fetch("ip", MockSignal())
        pipeline.map(_add_scalar)
        results = pipeline.compute_serial()
        for rec in results:
            rec["_toksearch_write_dir"] = "/tmp/somewhere"
        self.assertNotIn("_toksearch_write_dir", results.to_dataframe().columns)


class TestToDataFrameWithFailures(unittest.TestCase):
    """Records go ragged when a shot fails mid-map.

    A shot whose map function raised keeps `errors` and `shot` but never gains
    the field the others have. to_dataframe must still produce a frame, with
    the failed shot showing NaN rather than dropping out or raising --
    otherwise a partial run looks like a smaller complete one.
    """

    def _ragged(self):
        def peak_but_fail_on_two(rec):
            if rec.shot == 2:
                raise ValueError("no data")
            rec["peak"] = float(max(rec["ip"]["data"]))

        pipeline = Pipeline([1, 2, 3])
        pipeline.fetch("ip", MockSignal())
        pipeline.map(peak_but_fail_on_two)
        pipeline.keep(["peak"])
        return pipeline.compute_serial()

    def test_all_shots_are_present(self):
        self.assertEqual(sorted(self._ragged().to_dataframe()["shot"]), [1, 2, 3])

    def test_the_failed_shot_is_nan_not_missing(self):
        self.assertEqual(int(self._ragged().to_dataframe()["peak"].isna().sum()), 1)

    def test_the_successful_shots_keep_their_values(self):
        self.assertEqual(int(self._ragged().to_dataframe()["peak"].notna().sum()), 2)

    def test_errors_are_still_excluded_from_the_frame(self):
        self.assertNotIn("errors", self._ragged().to_dataframe().columns)


class TestToParquet(unittest.TestCase):
    def test_writes_a_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.parquet")
            _pipeline().compute_serial().to_parquet(path)
            self.assertTrue(os.path.exists(path))

    def test_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.parquet")
            _pipeline().compute_serial().to_parquet(path)
            self.assertEqual(len(pd.read_parquet(path)), 3)

    def test_returns_the_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.parquet")
            self.assertEqual(_pipeline().compute_serial().to_parquet(path), path)

    def test_honours_field_selection(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.parquet")
            _pipeline().compute_serial().to_parquet(path, fields=["peak"])
            self.assertEqual(list(pd.read_parquet(path).columns), ["shot", "peak"])

    def test_declares_the_output_to_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            record_path = os.path.join(d, "run.json")
            out = os.path.join(d, "out.parquet")
            prov = JsonProvenance("study", path=record_path)
            results = _pipeline().compute_serial(provenance=prov)
            results.to_parquet(out)
            self.assertIn(out, [o["path"] for o in prov._outputs])

    def test_the_declared_output_is_tagged_with_its_source(self):
        with tempfile.TemporaryDirectory() as d:
            record_path = os.path.join(d, "run.json")
            out = os.path.join(d, "out.parquet")
            prov = JsonProvenance("study", path=record_path)
            results = _pipeline().compute_serial(provenance=prov)
            results.to_parquet(out)
            entry = next(o for o in prov._outputs if o["path"] == out)
            self.assertEqual(entry["source"], "to_parquet")

    def test_without_provenance_it_just_writes(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.parquet")
            _pipeline().compute_serial().to_parquet(out)
            self.assertTrue(os.path.exists(out))

    def test_a_failing_provenance_backend_does_not_lose_the_file(self):
        # The file is written before provenance is told about it. A backend
        # failure must not cost the user their output.
        import warnings

        from toksearch.provenance.base import Provenance

        class _Exploding(Provenance):
            run_id = "x"

            def on_compute_start(self, ctx): pass
            def on_compute_end(self, ctx, recordset): pass
            def output(self, *paths, **kw): raise RuntimeError("boom")
            def metrics(self, name, values): pass
            def finalize(self): pass

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "out.parquet")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = _pipeline().compute_serial(provenance=_Exploding())
                results.to_parquet(out)
            self.assertTrue(os.path.exists(out))


class TestNoToNetcdf(unittest.TestCase):
    def test_to_netcdf_is_deliberately_absent(self):
        # Driver-side concatenation is slow and is a transformation that
        # deserves its own stage. Array-shaped results go through
        # Pipeline.write instead.
        results = _pipeline().compute_serial()
        self.assertFalse(hasattr(results, "to_netcdf"))
