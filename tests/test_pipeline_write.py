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

import glob
import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import xarray as xr

from toksearch.pipeline.writers import (
    UnknownFormat,
    extension_for,
    write_object,
    writer_for,
)


class TestWriterLookup(unittest.TestCase):
    def test_explicit_format_wins(self):
        self.assertIsNotNone(writer_for(fmt="netcdf", obj=None))

    def test_infers_netcdf_from_dataset(self):
        self.assertEqual(writer_for(obj=xr.Dataset()).fmt, "netcdf")

    def test_infers_netcdf_from_dataarray(self):
        self.assertEqual(writer_for(obj=xr.DataArray([1, 2])).fmt, "netcdf")

    def test_infers_parquet_from_dataframe(self):
        self.assertEqual(writer_for(obj=pd.DataFrame({"a": [1]})).fmt, "parquet")

    def test_infers_npy_from_ndarray(self):
        self.assertEqual(writer_for(obj=np.array([1, 2])).fmt, "npy")

    def test_infers_json_from_dict(self):
        self.assertEqual(writer_for(obj={"a": 1}).fmt, "json")

    def test_a_dict_of_arrays_is_json_not_npz(self):
        # npz is registered with no types precisely because a dict of arrays
        # is indistinguishable from a plain dict. It must be asked for.
        self.assertEqual(writer_for(obj={"a": np.array([1, 2])}).fmt, "json")

    def test_unknown_type_raises_a_clear_error(self):
        with self.assertRaises(UnknownFormat):
            writer_for(obj=object())

    def test_a_bare_scalar_raises_rather_than_guessing(self):
        # The likely user mistake: write(dir, field='betan_max') where the
        # field is a float. Must fail loudly with usable guidance.
        with self.assertRaises(UnknownFormat):
            writer_for(obj=3.14)

    def test_a_pandas_series_raises_rather_than_guessing(self):
        # Ambiguous between npy and parquet -- make the caller choose.
        with self.assertRaises(UnknownFormat):
            writer_for(obj=pd.Series([1, 2]))

    def test_the_error_names_the_type_and_says_what_to_do(self):
        with self.assertRaises(UnknownFormat) as caught:
            writer_for(obj=3.14)
        message = str(caught.exception)
        self.assertIn("float", message)
        self.assertIn("fmt=", message)

    def test_unknown_format_name_raises(self):
        with self.assertRaises(UnknownFormat):
            writer_for(fmt="nonesuch", obj=None)

    def test_unknown_format_error_lists_known_formats(self):
        with self.assertRaises(UnknownFormat) as caught:
            writer_for(fmt="nonesuch", obj=None)
        self.assertIn("netcdf", str(caught.exception))

    def test_extension_for_netcdf(self):
        self.assertEqual(extension_for("netcdf"), ".nc")

    def test_extension_for_parquet(self):
        self.assertEqual(extension_for("parquet"), ".parquet")


class TestWriteObject(unittest.TestCase):
    def test_writes_a_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.nc")
            write_object(xr.Dataset({"a": ("t", [1.0, 2.0])}), path, fmt="netcdf")
            self.assertTrue(os.path.exists(path))

    def test_roundtrips_a_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.nc")
            write_object(xr.Dataset({"a": ("t", [1.0, 2.0])}), path, fmt="netcdf")
            self.assertEqual(list(xr.open_dataset(path)["a"].values), [1.0, 2.0])

    def test_writes_a_dataframe(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.parquet")
            write_object(pd.DataFrame({"a": [1]}), path, fmt="parquet")
            self.assertTrue(os.path.exists(path))

    def test_roundtrips_a_dataframe(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.parquet")
            write_object(pd.DataFrame({"a": [1, 2]}), path, fmt="parquet")
            self.assertEqual(len(pd.read_parquet(path)), 2)

    def test_writes_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            write_object({"a": 1}, path, fmt="json")
            self.assertTrue(os.path.exists(path))

    def test_writes_npz_when_asked_explicitly(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.npz")
            write_object({"a": np.array([1, 2])}, path, fmt="npz")
            self.assertTrue(os.path.exists(path))

    def test_infers_the_format_when_not_given(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.nc")
            write_object(xr.Dataset({"a": ("t", [1.0])}), path)
            self.assertTrue(os.path.exists(path))

    def test_returns_the_path(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.json")
            self.assertEqual(write_object({"a": 1}, path, fmt="json"), path)


class TestRegistration(unittest.TestCase):
    def test_a_custom_writer_can_be_registered(self):
        from toksearch.pipeline.writers import register_writer

        class _Custom:
            pass

        def _write(obj, path):
            with open(path, "w") as fh:
                fh.write("custom")

        try:
            register_writer("custom_test_fmt", ".custom", _write, (_Custom,))
            self.assertEqual(writer_for(obj=_Custom()).fmt, "custom_test_fmt")
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "x.custom")
                write_object(_Custom(), path)
                self.assertTrue(os.path.exists(path))
        finally:
            from toksearch.pipeline import writers

            writers._WRITERS.pop("custom_test_fmt", None)
            if "custom_test_fmt" in writers._ORDER:
                writers._ORDER.remove("custom_test_fmt")


class TestWrittenPathIsTheReturnedPath(unittest.TestCase):
    """write_object's return value is recorded as a provenance artifact.

    np.save/np.savez append their extension to a path that lacks one, which
    would make the recorded path point at a file that does not exist.
    """

    def test_npy_writes_exactly_where_it_says(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bare")
            returned = write_object(np.array([1, 2]), path, fmt="npy")
            self.assertTrue(os.path.exists(returned))
            self.assertEqual(os.listdir(d), ["bare"])

    def test_npz_writes_exactly_where_it_says(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bare")
            returned = write_object({"a": np.array([1, 2])}, path, fmt="npz")
            self.assertTrue(os.path.exists(returned))
            self.assertEqual(os.listdir(d), ["bare"])

    def test_npy_still_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.npy")
            write_object(np.array([1, 2, 3]), path, fmt="npy")
            self.assertEqual(list(np.load(path)), [1, 2, 3])

    def test_npz_still_roundtrips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.npz")
            write_object({"a": np.array([1, 2])}, path, fmt="npz")
            self.assertEqual(list(np.load(path)["a"]), [1, 2])

    def test_every_format_writes_to_the_returned_path(self):
        cases = [
            ("netcdf", xr.Dataset({"a": ("t", [1.0])})),
            ("parquet", pd.DataFrame({"a": [1]})),
            ("npy", np.array([1, 2])),
            ("npz", {"a": np.array([1, 2])}),
            ("json", {"a": 1}),
        ]
        with tempfile.TemporaryDirectory() as d:
            for fmt, obj in cases:
                returned = write_object(obj, os.path.join(d, fmt), fmt=fmt)
                self.assertTrue(os.path.exists(returned), f"{fmt} path mismatch")


from toksearch.pipeline.pipeline_funcs import _SafeWrite
from toksearch.record import Record


def _make_record(shot, value=1.0):
    rec = Record(shot)
    rec["ds"] = xr.Dataset({"a": ("t", [value, value])})
    return rec


class TestSafeWrite(unittest.TestCase):
    def test_writes_one_file_named_for_the_shot(self):
        with tempfile.TemporaryDirectory() as d:
            _SafeWrite(d, field="ds", fmt="netcdf")(_make_record(123))
            self.assertTrue(os.path.exists(os.path.join(d, "123.nc")))

    def test_returns_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _make_record(123)
            self.assertIs(_SafeWrite(d, field="ds", fmt="netcdf")(rec), rec)

    def test_records_the_output_path_on_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _SafeWrite(d, field="ds", fmt="netcdf")(_make_record(123))
            self.assertEqual(rec["output_path"], os.path.join(d, "123.nc"))

    def test_the_recorded_path_actually_exists(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _SafeWrite(d, field="ds", fmt="netcdf")(_make_record(123))
            self.assertTrue(os.path.exists(rec["output_path"]))

    def test_records_the_output_directory_on_the_record(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _SafeWrite(d, field="ds", fmt="netcdf")(_make_record(123))
            self.assertEqual(rec["_toksearch_write_dir"], os.path.abspath(d))

    def test_creates_the_directory_if_absent(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "nested", "deeper")
            _SafeWrite(out, field="ds", fmt="netcdf")(_make_record(1))
            self.assertTrue(os.path.exists(os.path.join(out, "1.nc")))

    def test_infers_format_from_the_object(self):
        with tempfile.TemporaryDirectory() as d:
            _SafeWrite(d, field="ds")(_make_record(123))
            self.assertTrue(os.path.exists(os.path.join(d, "123.nc")))

    def test_missing_field_sets_an_error_and_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            rec = _SafeWrite(d, field="nope", fmt="netcdf")(_make_record(123))
            self.assertIn("write", rec["errors"])

    def test_missing_field_writes_no_file(self):
        with tempfile.TemporaryDirectory() as d:
            _SafeWrite(d, field="nope", fmt="netcdf")(_make_record(123))
            self.assertEqual(os.listdir(d), [])

    def test_missing_field_still_returns_the_record(self):
        # A failed write must not break the pipeline chain: _apply_operations
        # stops on a falsy return.
        with tempfile.TemporaryDirectory() as d:
            rec = _make_record(123)
            self.assertIs(_SafeWrite(d, field="nope", fmt="netcdf")(rec), rec)

    def test_an_unwritable_object_sets_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Record(1)
            rec["scalar"] = 3.14
            out = _SafeWrite(d, field="scalar")(rec)
            self.assertIn("write", out["errors"])

    def test_multiple_fields_are_merged_into_one_dataset(self):
        with tempfile.TemporaryDirectory() as d:
            rec = Record(7)
            rec["a"] = xr.Dataset({"x": ("t", [1.0])})
            rec["b"] = xr.Dataset({"y": ("t", [2.0])})
            _SafeWrite(d, fields=["a", "b"], fmt="netcdf")(rec)
            written = xr.open_dataset(os.path.join(d, "7.nc"))
            self.assertIn("x", written)
            self.assertIn("y", written)

    def test_custom_name_callable(self):
        with tempfile.TemporaryDirectory() as d:
            op = _SafeWrite(d, field="ds", fmt="netcdf",
                            name=lambda rec: f"shot_{rec.shot}")
            op(_make_record(5))
            self.assertTrue(os.path.exists(os.path.join(d, "shot_5.nc")))

    def test_writer_func_returning_an_object(self):
        with tempfile.TemporaryDirectory() as d:
            _SafeWrite(d, func=lambda rec: rec["ds"], fmt="netcdf")(_make_record(9))
            self.assertTrue(os.path.exists(os.path.join(d, "9.nc")))

    def test_writer_func_returning_a_path_it_wrote(self):
        def writer(rec, path):
            with open(path, "w") as fh:
                fh.write("hi")
            return path

        with tempfile.TemporaryDirectory() as d:
            rec = _SafeWrite(d, func=writer)(_make_record(11))
            self.assertTrue(os.path.exists(os.path.join(d, "11")))
            self.assertEqual(rec["output_path"], os.path.join(d, "11"))

    def test_a_raising_writer_func_sets_an_error(self):
        def writer(rec):
            raise ValueError("nope")

        with tempfile.TemporaryDirectory() as d:
            rec = _SafeWrite(d, func=writer, fmt="netcdf")(_make_record(1))
            self.assertIn("write", rec["errors"])

    def test_custom_path_field(self):
        with tempfile.TemporaryDirectory() as d:
            op = _SafeWrite(d, field="ds", fmt="netcdf", path_field="nc_path")
            rec = op(_make_record(3))
            self.assertIn("nc_path", rec)

    def test_spec_reports_the_write_op(self):
        with tempfile.TemporaryDirectory() as d:
            spec = _SafeWrite(d, field="ds", fmt="netcdf").spec().to_dict()
            self.assertEqual(spec["op"], "write")
            self.assertEqual(spec["detail"]["fields"], ["ds"])
            self.assertEqual(spec["detail"]["track"], "directory")

    def test_spec_directory_is_absolute(self):
        with tempfile.TemporaryDirectory() as d:
            spec = _SafeWrite(d, field="ds", fmt="netcdf").spec().to_dict()
            self.assertEqual(spec["detail"]["directory"], os.path.abspath(d))

    def test_spec_is_canonically_serializable(self):
        from toksearch.provenance.hashing import canonical_json

        with tempfile.TemporaryDirectory() as d:
            op = _SafeWrite(d, field="ds", fmt="netcdf",
                            name=lambda rec: str(rec.shot))
            canonical_json(op.spec().to_dict())

    def test_spec_carries_no_memory_address(self):
        from toksearch.provenance.hashing import canonical_json

        with tempfile.TemporaryDirectory() as d:
            op = _SafeWrite(d, func=lambda rec: rec["ds"], fmt="netcdf")
            self.assertNotIn("0x", canonical_json(op.spec().to_dict()))

    def test_spec_keys_match_what_write_directories_reads(self):
        # RunContext.write_directories filters on op == "write" and
        # detail["track"] == "directory", then reads detail["directory"].
        from toksearch.provenance.code import capture_code
        from toksearch.provenance.context import BackendSpec, RunContext, SourceSpec

        with tempfile.TemporaryDirectory() as d:
            ctx = RunContext(
                source=SourceSpec(kind="shotlist", count=1),
                ops=(_SafeWrite(d, field="ds", fmt="netcdf").spec(),),
                signals={},
                backend=BackendSpec(kind="SerialRecordSet"),
                code=capture_code(),
            )
            self.assertEqual(ctx.write_directories(), [os.path.abspath(d)])


from toksearch import Pipeline
from toksearch.signal.mock_signal import MockSignal


class TestPipelineWrite(unittest.TestCase):
    def _pipeline(self):
        pipeline = Pipeline([1, 2, 3])
        pipeline.fetch_dataset("ds", {"ip": MockSignal()})
        return pipeline

    def test_declarative_form_writes_one_file_per_shot(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            pipeline = self._pipeline()
            pipeline.write(out, field="ds", fmt="netcdf")
            pipeline.compute_serial()
            self.assertEqual(sorted(os.listdir(out)), ["1.nc", "2.nc", "3.nc"])

    def test_declarative_form_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            pipeline = self._pipeline()
            self.assertIsNone(pipeline.write(os.path.join(d, "p"), field="ds"))

    def test_decorator_form_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            pipeline = self._pipeline()

            @pipeline.write(out, fmt="netcdf")
            def shot_file(rec):
                return rec["ds"]

            pipeline.compute_serial()
            self.assertEqual(sorted(os.listdir(out)), ["1.nc", "2.nc", "3.nc"])

    def test_decorator_returns_the_original_function(self):
        with tempfile.TemporaryDirectory() as d:
            pipeline = self._pipeline()

            @pipeline.write(os.path.join(d, "p"), fmt="netcdf")
            def shot_file(rec):
                return rec["ds"]

            self.assertEqual(shot_file.__name__, "shot_file")
            self.assertTrue(callable(shot_file))

    def test_two_argument_decorator_form(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            pipeline = self._pipeline()

            @pipeline.write(out)
            def shot_file(rec, path):
                with open(path, "w") as fh:
                    fh.write(str(rec.shot))
                return path

            pipeline.compute_serial()
            self.assertEqual(sorted(os.listdir(out)), ["1", "2", "3"])

    def test_multiprocessing_backend_writes_files(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            pipeline = self._pipeline()
            pipeline.write(out, field="ds", fmt="netcdf")
            pipeline.compute_multiprocessing(num_workers=2)
            self.assertEqual(sorted(os.listdir(out)), ["1.nc", "2.nc", "3.nc"])

    def test_written_files_are_readable(self):
        # Per-file open, not open_mfdataset: dask is not installed here.
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            pipeline = self._pipeline()
            pipeline.write(out, field="ds", fmt="netcdf")
            pipeline.compute_serial()
            written = sorted(glob.glob(os.path.join(out, "*.nc")))
            # data_vars is explicit: xarray warns that its default changes in a
            # future release, and the suite's warning count is a signal.
            merged = xr.concat(
                [xr.open_dataset(f) for f in written],
                dim="shot",
                data_vars="all",
            )
            self.assertIn("ip", merged)

    def test_non_empty_directory_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            os.makedirs(out)
            with open(os.path.join(out, "stale.nc"), "w") as fh:
                fh.write("old")
            pipeline = self._pipeline()
            with self.assertRaises(ValueError):
                pipeline.write(out, field="ds", fmt="netcdf")

    def test_the_refusal_explains_why(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            os.makedirs(out)
            with open(os.path.join(out, "stale.nc"), "w") as fh:
                fh.write("old")
            pipeline = self._pipeline()
            with self.assertRaises(ValueError) as caught:
                pipeline.write(out, field="ds", fmt="netcdf")
            self.assertIn("exist_ok", str(caught.exception))

    def test_non_empty_directory_allowed_with_exist_ok(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            os.makedirs(out)
            with open(os.path.join(out, "stale.nc"), "w") as fh:
                fh.write("old")
            pipeline = self._pipeline()
            pipeline.write(out, field="ds", fmt="netcdf", exist_ok=True)

    def test_an_empty_existing_directory_is_fine(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            os.makedirs(out)
            self._pipeline().write(out, field="ds", fmt="netcdf")

    def test_write_appears_in_the_run_context_ops(self):
        from toksearch.backend.serial import SerialRecordSet

        with tempfile.TemporaryDirectory() as d:
            pipeline = self._pipeline()
            pipeline.write(os.path.join(d, "p"), field="ds", fmt="netcdf")
            ctx = pipeline._run_context(SerialRecordSet, None)
            self.assertEqual([op.op for op in ctx.ops][-1], "write")

    def test_write_directories_needs_no_computation(self):
        # RunContext.write_directories reads the pipeline definition, so the
        # output directory is known before anything runs. That is what keeps
        # provenance from forcing materialization on lazy Ray/Spark backends.
        from toksearch.backend.serial import SerialRecordSet

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            pipeline = self._pipeline()
            pipeline.write(out, field="ds", fmt="netcdf")
            ctx = pipeline._run_context(SerialRecordSet, None)
            self.assertEqual(ctx.write_directories(), [os.path.abspath(out)])

    def test_track_file_is_recorded_in_the_spec(self):
        from toksearch.backend.serial import SerialRecordSet

        with tempfile.TemporaryDirectory() as d:
            pipeline = self._pipeline()
            pipeline.write(os.path.join(d, "p"), field="ds", fmt="netcdf",
                           track="file")
            ctx = pipeline._run_context(SerialRecordSet, None)
            self.assertEqual(ctx.ops[-1].detail["track"], "file")

    def test_track_file_is_excluded_from_write_directories(self):
        from toksearch.backend.serial import SerialRecordSet

        with tempfile.TemporaryDirectory() as d:
            pipeline = self._pipeline()
            pipeline.write(os.path.join(d, "p"), field="ds", fmt="netcdf",
                           track="file")
            ctx = pipeline._run_context(SerialRecordSet, None)
            self.assertEqual(ctx.write_directories(), [])

    def test_invalid_track_value_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            pipeline = self._pipeline()
            with self.assertRaises(ValueError):
                pipeline.write(os.path.join(d, "p"), field="ds", track="bogus")

    def test_end_to_end_with_provenance(self):
        import json

        from toksearch.provenance import JsonProvenance

        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "peaks")
            record = os.path.join(d, "run.json")
            prov = JsonProvenance("study", path=record)
            pipeline = self._pipeline()
            pipeline.write(out, field="ds", fmt="netcdf")
            pipeline.compute_serial(provenance=prov)
            prov.finalize()
            with open(record) as fh:
                payload = json.load(fh)
        self.assertEqual([o["path"] for o in payload["outputs"]],
                         [os.path.abspath(out)])
