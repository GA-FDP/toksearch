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

"""
Utilities for manipulating xarray datasets
"""
import xarray as xr


def merge_val(ds, name, val):
    """
    Given a dataset ds, merge val into it so
    that the dims are shared
    """
    da = xr.DataArray(val)
    ds = xr.merge([ds, da.to_dataset(name=name)])
    return ds


def keep(ds, keys):
    """Given a dataset ds, keep only the keys in the list keys"""
    keys = set(keys)
    existing_keys = set(ds.data_vars.keys())

    # Get rid of anything that doesn't exist in the ds
    keys = existing_keys.intersection(keys)
    keys_to_drop = existing_keys - keys
    return ds.drop(keys_to_drop)


def discard(ds, keys):
    """Given a dataset ds, discard the keys in the list keys"""
    return ds.drop(keys)
