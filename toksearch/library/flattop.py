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
from typing import Dict
from numpy.typing import ArrayLike
from toksearch.library.ell1 import L1Fit


class SimpleFlattopFinder:
    """
    Class to find the start and end times for flattop.

    This deals with the usual type of flattop where IP is reasonably flat.
    It doesn't really handle weird cases where there are "steps" in ramp up
    or ramp down.

    Attributes
        t_start (float): Flattop start time
        t_end (float): Flattop end time
        ip_hat (ArrayLike): The fitted version of d
        times (ArrayLike): The timebase (just a pass through of the input t)
        std (float): Standard deviation of IP during flattop
        average_slope (float): Average slope of IP during flattop
    """

    def __init__(
        self,
        t: ArrayLike,
        d: ArrayLike,
        lamb: float = 1.0,
        eps: float = 0.001,
        slope_threshold: float = 0.2,
        ip_fraction_threshold: float = 0.75,
    ):
        """
        Instantiate a finder object

        Parameters:
            t (ArrayLike): Time vector
            d (ArrayLike): Plasma current. Units don't matter because everything
                is normalized internally.

        Keyword Parameters:
            lamb (float): Number controlling how aggressive the underlying L1 trend
                filter is. A larger number removes more wiggliness, but will sometimes
                cause the inflection points to be off a bit. Defaults to 1.0.
            eps (float): You probably don't need to mess with this. Used internally to
                set small second derivatives to zero when they fall below this threshold.
            slope_threshold (float): Fraction of the largest slope in the shot that
                can occur in flattop. Defaults to 0.2.
            ip_fraction_threshold (float): Fraction of max absolute IP. Anything above this
                is considered flattop, anything below is not.
        """

        self.times = t
        self.lamb = lamb
        self.eps = eps
        self.slope_threshold = slope_threshold
        self.ip_fraction_threshold = ip_fraction_threshold

        self.start_index = None
        self.end_index = None
        self.t_start = None
        self.t_end = None

        abs_max = np.max(np.abs(d))

        ell1 = L1Fit(t, d, lamb=lamb, scale=1 / abs_max)

        max_abs_slope = np.max(np.abs(np.gradient(ell1.yhat, t)))

        def seg_criteria(
            t_seg,
            d_seg,
            ip_fraction_threshold=ip_fraction_threshold,
            slope_threshold=slope_threshold,
            max_abs_slope=max_abs_slope,
            max_abs_current=abs_max,
        ):
            slope = (d_seg[-1] - d_seg[0]) / (t_seg[-1] - t_seg[0])
            slope = slope / max_abs_slope

            seg_min = min(np.abs(d_seg[-1]), np.abs(d_seg[0])) / abs_max

            return (np.abs(slope) < slope_threshold) and (
                seg_min > ip_fraction_threshold
            )

        contiguous_segs = ell1.group_contiguous_segments(seg_criteria)

        contiguous_segs.sort(key=lambda seg: seg[1] - seg[0], reverse=True)
        biggest_seg = contiguous_segs[0]
        self.start_index, self.end_index = biggest_seg
        self.t_start, self.t_end = t[self.start_index], t[self.end_index]

        self.ip_hat = ell1.yhat

    @property
    def average_slope(self):
        delta_ip = self.ip_hat[self.end_index] - self.ip_hat[self.start_index]
        delta_t = self.t_end - self.t_start
        return delta_ip / delta_t

    @property
    def std(self):
        ip_flat = self.ip_hat[self.start_index : self.end_index]
        return np.std(ip_flat)
