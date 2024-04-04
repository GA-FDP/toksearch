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

from typing import Union, Dict, List, Tuple, Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from scipy.ndimage import find_objects

from .ell1module import _compute_ell1

DEFAULT_EPS = 1e-6
DEFAULT_SEG_EPS = 1e-3


class L1Fit:
    def __init__(
        self,
        x: ArrayLike,
        y: ArrayLike,
        lamb: float,
        yl: Union[float, ArrayLike, None] = None,
        yr: Union[float, ArrayLike, None] = None,
        w: Union[ArrayLike, None] = None,
        eps: float = DEFAULT_EPS,
        eta: float = 0.96,
        maxiters: int = 50,
        nthreads: int = 0,
        scale: float = 1.0,
    ):

        self.x = x
        self.y = y
        self.yhat = (1 / scale) * _compute_ell1(
            scale * y,
            lamb,
            yl=yl,
            yr=yr,
            w=w,
            eps=eps,
            eta=eta,
            maxiters=maxiters,
            nthreads=nthreads,
        )["y"]

    def segments(self, eps: float = DEFAULT_SEG_EPS) -> List[Tuple[int, int]]:

        y_prime = np.gradient(self.yhat, self.x)
        y_prime = y_prime / np.max(np.abs(y_prime))

        y_prime_prime = np.gradient(y_prime, self.x)
        y_prime_prime = y_prime_prime / np.max(np.abs(y_prime_prime))

        smalls = np.abs(y_prime_prime) < eps
        y_prime_prime[smalls] = 0

        peaks_pos, _ = find_peaks(y_prime_prime)
        peaks_neg, _ = find_peaks(-y_prime_prime)
        peaks = list(peaks_pos) + list(peaks_neg)
        peaks.sort()

        seg_ends = peaks + [len(self.x) - 1]

        seg_start = 0
        seg_tuples = []
        for i, seg_end in enumerate(seg_ends):
            seg_tuples.append((seg_start, seg_end))
            seg_start = seg_end

        return seg_tuples

    def group_contiguous_segments(
        self,
        criteria_func: Callable[[ArrayLike, ArrayLike], bool],
        eps: float = DEFAULT_SEG_EPS,
    ) -> List[Tuple[int, int]]:

        segments = self.segments(eps=eps)

        contiguous_segs = []

        seg_start = 0
        matches = np.zeros(len(segments), dtype=np.int64)

        for i, (start_idx, end_idx) in enumerate(segments):
            if criteria_func(
                self.x[start_idx : end_idx + 1], self.yhat[start_idx : end_idx + 1]
            ):
                matches[i] = 1

        # find_objects returns a list of tuples of slice
        matching_slices = [sl_tup[0] for sl_tup in find_objects(matches)]
        for matching_slice in matching_slices:
            left_idx = segments[matching_slice.start][0]
            right_idx = segments[matching_slice.stop - 1][1]
            contiguous_segs.append((left_idx, right_idx))

        return contiguous_segs


def compute_ell1(
    x: ArrayLike,
    lamb: float,
    yl: Union[float, ArrayLike, None] = None,
    yr: Union[float, ArrayLike, None] = None,
    w: Union[ArrayLike, None] = None,
    eps: float = 1e-6,
    eta: float = 0.96,
    maxiters: int = 50,
    nthreads: int = 0,
) -> Dict:
    """
    Apply an L1 trend filter to x

    Parameters:
        x (ArrayLike): The input array to fit
        lamb (float): Number controlling how aggressive the L1 trend
            filter is. In other words, the penalty applied to the second
            derivative. A larger number removes more wiggliness, but will sometimes
            cause the inflection points to be off a bit.
            eps (float): You probably don't need to mess with this. Used internally to
                set small second derivatives to zero when they fall below this threshold.

    Keyword Parameters:
        yl (Union[float, ArrayLike, None]): Left side boundary points.
            TODO: Better explanation
        yr (Union[float, ArrayLike, None]): Right side boundary points.
            TODO: Better explanation
        eps (float): You probably don't need to mess with this. Used internally to
            set small second derivatives to zero when they fall below this threshold.
        eta (float): Not sure what this does...Probably best not to mess with it!
        maxiters (int): Maximum number of iterations to run. Defaults to 50


    Returns:
        (Dict): Dictionary with fields:
            y (ArrayLike): The fitted version of x

            Note: More fields to come later with info on qualtiy of fit, etc...
    """
    return _compute_ell1(
        x,
        lamb,
        yl=yl,
        yr=yr,
        w=w,
        eps=eps,
        eta=eta,
        maxiters=maxiters,
        nthreads=nthreads,
    )
