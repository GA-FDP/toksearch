import os
from pathlib import Path
import numpy as np
import xarray as xr
import shutil
import tempfile
import fsspec
import unittest
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

from toksearch.signal.zarr import ZarrSignal


class TestZarrSignal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.create_tmp_workdir()
        cls.create_tmp_zarr()

    @classmethod
    def create_tmp_workdir(self):
        # Create a temporary directory
        self.test_dir = Path(tempfile.mkdtemp())
        # Save the original working directory
        self.original_dir = Path.cwd()
        # Change to the temp directory
        os.chdir(self.test_dir)

    @classmethod
    def create_tmp_zarr(self):
        ip = xr.DataArray(
            np.random.random((1000,)), dims=["time"], attrs=dict(units="A")
        )
        time = xr.DataArray(np.linspace(0, 1.0, 1000), dims=["time"])
        dataset = xr.Dataset(dict(ip=ip, time=time))
        dataset.to_zarr("30421.zarr", group="magnetics")

    @classmethod
    def tearDownClass(cls):
        # Change back to the original working directory
        os.chdir(cls.original_dir)
        # Remove the temp directory
        shutil.rmtree(str(cls.test_dir))

    def test_fetch(self):
        ip_signal = ZarrSignal(path=self.test_dir, treepath="magnetics/ip")
        result = ip_signal.fetch(30421)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], np.ndarray)
        self.assertEqual(result["data"].shape, (1000,))
        self.assertIsInstance(result["times"], np.ndarray)
        self.assertEqual(result["times"].shape, (1000,))

    def test_fetch_as_xarray(self):
        ip_signal = ZarrSignal(path=self.test_dir, treepath="magnetics/ip")
        result = ip_signal.fetch_as_xarray(30421)
        self.assertIsInstance(result, xr.DataArray)
        self.assertEqual(result.shape, (1000,))
        self.assertEqual(result.attrs["units"], "A")

    def test_fetch_fs(self):
        # create a local async filesystem object for testing
        # could replace with s3fs.S3FileSystem(...) etc. for remote file access
        fs = AsyncFileSystemWrapper(fsspec.filesystem("file"), asynchronous=True)
        path = f"file://{self.test_dir}"
        ip_signal = ZarrSignal(path=path, treepath="magnetics/ip", fs=fs)
        result = ip_signal.fetch(30421)
        self.assertIsInstance(result, dict)
        self.assertIn("data", result)
        self.assertIsInstance(result["data"], np.ndarray)
        self.assertEqual(result["data"].shape, (1000,))
        self.assertIsInstance(result["times"], np.ndarray)
        self.assertEqual(result["times"].shape, (1000,))
