"""This module provides classes for fetching data from Zarr stores

The main class for user applications is the ZarrSignal class. This class
provides a way to fetch data from Zarr stores either on a local disk or
from a remote server. The ZarrSignal class is a subclass of the Signal class
and provides the same interface as the Signal class.

The ZarrSignal class is a wrapper around the ZarrLocalSignal and ZarrRemoteSignal
classes. The ZarrLocalSignal class is used to fetch data from a local disk and
the ZarrRemoteSignal class is used to fetch data from a remote server. The
ZarrSignal class determines which class to use based on the location argument
provided to the constructor.
"""

import os
import fsspec
import zarr
import zarr.storage
import xarray as xr
from typing import Iterable, Optional
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

from .signal import Signal


class ZarrSignal(Signal):
    def __init__(
        self,
        path: str,
        treepath: str,
        dims: Iterable[str] = ("times",),
        fetch_units: bool = True,
        fs: Optional[AsyncFileSystem] = None,
    ):
        """Create a signal object that fetches data from an MDSplus tree

        Arguments:
            path: The tdi expression to fetch data from
            treepath: The name of the signal within the store to load. e.g. `magentics/ip`
            dims: See documentation for the Signal class. Defaults to ('times',)
            fetch_units: See documentation for the Signal class. Defaults
                to True.
            fs: The `fsspec` filesystem object to use (optional). If `None`
                assume that Zarr store is a local path
        """
        super().__init__()
        self.path = path
        self.treepath = treepath
        self.fetch_units = fetch_units
        self.dims = dims
        self.fs = fs
        if fs is None:
            self.fs = AsyncFileSystemWrapper(
                fsspec.filesystem("file"), asynchronous=True
            )

    def gather(self, shot):
        """Gather the data for a shot

        Arguments:
            shot (int): The shot number to gather the data for

        Returns:
            dict: A dictionary containing the data gathered for the signal. The dictionary
                will contain a key 'data' with the data, and keys for each dimension of the
                data, with the values being the values of the dimensions. If the with_units
                attribute is True, the dictionary will also contain a key 'units' with the units
                of the data and dimensions.
        """
        shot_file = os.path.join(self.path, f"{shot}.zarr")
        if "://" not in shot_file:
            # edge case: default to using local `file://` protocol if none specified
            shot_file = f"file://{shot_file}"
        protocol, file_path = str(shot_file).split("://", maxsplit=1)
        parts = self.treepath.rsplit("/", maxsplit=1)
        signal_name = parts[-1]
        group_name = parts[-2] if len(parts) > 1 else None

        store = zarr.storage.FsspecStore(fs=self.fs, path=file_path)
        store = xr.open_zarr(store, group=group_name)
        signal = store[signal_name]
        data = signal.values

        # Add data
        result = dict(data=data)

        # Add dimensions
        units = {}
        for new_dim, dim in zip(self.dims, signal.sizes.keys()):
            if dim in store:
                result[new_dim] = store[dim].values

        # Add units
        if self.fetch_units:
            units = {}
            result["units"] = {"data": signal.attrs.get("units", "")}
            for new_dim, dim in zip(self.dims, signal.sizes.keys()):
                if dim in store:
                    units[new_dim] = store[dim].attrs.get("units", "")

        return result

    def cleanup_shot(self, shot: int):
        """Close the tree for this shot

        Arguments:
            shot (int): The shot number to close the tree for
        """
        pass

    def cleanup(self):
        """Cleanup"""
        pass
