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
