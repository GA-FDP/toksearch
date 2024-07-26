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

import numpy as np
import xarray as xr
from .signal import Signal


class MockSignal(Signal):
    """Signal used for testing purposes.  Returns a simple array of data and times."""

    default_d = np.arange(4)
    default_t = np.arange(4)

    def __init__(self, data=None, times=None, **kwargs):
        super().__init__()

        self.set_dims(kwargs.get("dims", ("times",)))

        self.data = data if data is not None else self.default_d
        self.times = times if times is not None else self.default_t

    def gather(self, shot):
        results = {}

        results["data"] = self.data
        
        if self.dims:
            results["times"] = self.times

        units_dict = {}
        if self.with_units:
            units_dict["data"] = "A"
            if self.dims and "times" in self.dims:
                units_dict["times"] = "ms"

            results["units"] = units_dict

        return results


    def cleanup(self):
        pass

    def cleanup_shot(self, shot):
        pass
