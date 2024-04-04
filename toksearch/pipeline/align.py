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

from builtins import object
import numpy as np
import xarray as xr
import uuid


def _is_numeric(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


class XarrayAligner(object):
    def __init__(
        self,
        align_with,
        dim="times",
        method="pad",
        extrapolate=True,
        interp_kwargs=None,
    ):
        """
        Parameters:
            align_with: String, numpy-like array
            method: Default is to zero-order hold. Can also interpolate using ???
            signals: If none, align everything in the ds

        """
        self.align_with = align_with
        self.dim = dim
        self.method = method
        self.extrapolate = extrapolate
        self.interp_kwargs = interp_kwargs or {}
        if self.extrapolate:
            self.interp_kwargs["fill_value"] = "extrapolate"

        if isinstance(align_with, str):
            self._get_coords = _coords_from_string
        elif isinstance(align_with, list):
            self._get_coords = _coords_from_list
        elif isinstance(align_with, np.ndarray):
            self._get_coords = _coords_from_ndarray
        elif callable(align_with):
            self._get_coords = _coords_from_callable
        elif _is_numeric(align_with):
            self._get_coords = _coords_from_sample_period
        else:
            e = Exception(
                (
                    "Invalid align_with argument. "
                    "Must be string, list, numeric sample period, callable, or ndarray"
                )
            )
            raise (e)

    def __call__(self, ds):
        if self.method == "pad":
            return self._zoh(ds)
        else:
            return self._interpolate(ds)

    def _interpolate(self, ds):
        # Get the coords as an np.ndarray
        coords = self._get_coords(ds, self.align_with, self.dim)
        das = []
        for key in list(ds.data_vars.keys()):
            if self.dim not in ds[key].dims:
                continue

            base_kwargs = {self.dim: coords, "method": self.method}
            da = (
                ds[key]
                .dropna(self.dim)
                .interp(
                    kwargs=self.interp_kwargs,
                    **{self.dim: coords, "method": self.method}
                )
            )
            das.append(da)

        if "shot" in ds.dims:
            ds = xr.Dataset(coords={"shot": ds.shot})
        else:
            ds = xr.Dataset()
        ds = xr.merge([ds] + das)
        return ds

    def _zoh(self, ds):
        # Get the coords as an np.ndarray
        coords = self._get_coords(ds, self.align_with, self.dim)
        # First forward fill to deal with NANs left over from merging signals
        # into the dataset
        ds = ds.ffill(self.dim)

        # Now backfill to deal with values at the beginning that couldn't be
        # forward filled
        ds = ds.bfill(self.dim)

        ds = ds.reindex(**{self.dim: coords, "method": self.method})
        return ds


def _coords_from_sample_period(ds, align_with, dim):
    sample_period = align_with
    dim_vals = ds[dim].values
    min_dim = np.min(dim_vals)
    max_dim = np.max(dim_vals)
    return np.arange(min_dim, max_dim, sample_period)


def _coords_from_string(ds, align_with, dim):
    signal_name = align_with
    coords = ds[signal_name].dropna(dim)[dim].values
    return coords


def _coords_from_list(ds, align_with, dim):
    return np.array(align_with)


def _coords_from_ndarray(ds, align_with, dim):
    return align_with


def _coords_from_callable(ds, align_with, dim):
    return align_with(ds)
