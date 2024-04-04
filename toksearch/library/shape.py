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

import warnings
import numpy as np
from scipy import interpolate

from toksearch import Pipeline, MdsSignal


##############################################################################
# Utility Functions
def _interpolate_boundary(bdry_data, num_points=100):
    """Do spline interpolation of bdry_data

    Returns:
        ndarray of shape (num_points, 2), where the second dimension is
        a x,y coordinate
    """

    u = np.linspace(0, 1, 100)
    non_origin_indices = bdry_data[:, 0] > 0.01
    bdry_data = bdry_data[non_origin_indices]
    x = bdry_data[:, 0]
    y = bdry_data[:, 1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tck, _ = interpolate.splprep([x, y], s=0, per=True)

        xi, yi = interpolate.splev(u, tck)
        return np.stack([np.array(xi), np.array(yi)]).T


def _find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def _default_translate(x):
    centroid = np.mean(x, axis=0)
    x = x - centroid
    return x


def _no_errors(rec):
    return not rec.errors


def _passthrough(x):
    return x


class CalcDistances:
    def __init__(self, baseline_bdry, translate_func):
        self.baseline_bdry = baseline_bdry
        self.translate_func = translate_func

    def __call__(self, rec):
        baseline_bdry = self.baseline_bdry
        translate_func = self.translate_func

        data = rec["bdry"]["data"]
        times = rec["bdry"]["times"]

        # Loop over each time slice and grab the boundary, interpolating
        # the same number of points as the baseline bdry
        interpolated_boundaries = []
        deltas = []

        for bdry_data in np.split(data, data.shape[0], axis=0):
            interpolated_boundary = _interpolate_boundary(bdry_data[0, :, :])
            translated_boundary = translate_func(interpolated_boundary.copy())

            delta = translated_boundary[0, :] - interpolated_boundary[0, :]
            deltas.append(delta)
            interpolated_boundaries.append(translated_boundary)

        bdry_interp = np.stack(interpolated_boundaries)
        element_wise_square_dist = np.sum((bdry_interp - baseline_bdry) ** 2, axis=2)
        mean_square_dist = np.mean(element_wise_square_dist, axis=1)
        rec["dist"] = {"data": np.sqrt(mean_square_dist), "times": times}
        rec["translation"] = {"data": np.stack(deltas), "times": times}
        rec["min_dist"] = np.min(rec["dist"]["data"])
        rec["i_min_dist"] = np.argmin(rec["dist"]["data"])
        rec["t_min_dist"] = times[rec["i_min_dist"]]
        rec["closest_match"] = bdry_interp[rec["i_min_dist"], :, :]


##############################################################################


class BoundarySimilarityPipeline(Pipeline):

    def __init__(
        self,
        parent,
        baseline_shot,
        baseline_time,
        translate=False,
        efit_tree="efit01",
        location=None,
        bdry_pointname=r"\bdry",
        num_interpolation_points=200,
        batch_size=None,
    ):
        """
        Create a Pipeline object that calculates a similarity metric relative
        to a baseline a plasma boundary shape.

        Parameters:
            parent (Iterable or Pipeline or PipelineSource):
                If parent is an Iterable, then the elements of the Iterable
                must be one of three types:
                    1) A integer shot number
                    2) A dictionary containing at least the field "shot"
                       (and not the fields "key" or "errors")
                    3) A Record object.

                If the parent is another Pipeline, then the newly constructed
                Pipeline will act as a continuation of the parent.

                The parent can also be a PipelineSource, although typically this
                is handled internally.

            baseline_shot (int): The shot from which to fetch baseline data
            baseline_time (float): Time slice during the baseline shot to
                compare against. The closest match will be used.


        Keyword Parameters:
            translate (truth-like value or callable): Whether to attempt to
                make the similarity match translation invariant. If a truth-like
                value evaluates to true, then for each boundary, the minimum
                x and y of the boundary will be subrtracted from each boundary
                point. If a callable is oassed, then it must be of the form
                f(x) -> x_translated where both x and x_translated are ndarrays
                with shape (n_boundary_points, 2). Defaults to False.

            efit_tree (str): Name of the MDSplus efit tree to use. Defaults to
                efit01.

            location (str): Location of the MDSplus tree. See documentation for
                the MdsSignal class for details on usage.

            bdry_pointname (str): Name of the MDSplus pointname. Defaults to
                '\\bdry'

            num_interpolation_points (int): Number of boundary points to
                interpolate the boundaries onto. Defaults to 200.

            batch_size (int): If set to an integer, limits the number of
                shots being processed at once. If not set, all shots are
                done in a single batch. This is useful for very large
                jobs that need more memory than available on the host or
                cluster being used.

        """

        super().__init__(parent, batch_size=batch_size)

        if translate:
            if callable(translate):
                translate_func = translate
            else:
                translate_func = _default_translate
        else:
            translate_func = _passthrough

        bdry_sig = MdsSignal(
            bdry_pointname,
            efit_tree,
            location=location,
            dims=("x_y", "index", "times"),
            data_order=("times", "index", "x_y"),
        )

        baseline_record = bdry_sig.fetch(baseline_shot)
        baseline_data = baseline_record["data"]
        baseline_times = baseline_record["times"]

        self.baseline_time, self.baseline_index = _find_nearest(
            baseline_times, baseline_time
        )

        baseline_bdry = baseline_data[self.baseline_index, :, :]
        translated_baseline_bdry = _interpolate_boundary(baseline_bdry.copy())
        self.baseline_bdry = translate_func(translated_baseline_bdry)
        self.baseline_translation = translated_baseline_bdry[0, :] - baseline_bdry[0, :]

        self.fetch("bdry", bdry_sig)
        self.where(_no_errors)

        self.map(CalcDistances(self.baseline_bdry, translate_func))

        # Need to throw away most stuff otherwise it uses too much memory
        self.keep(
            [
                "dist",
                "min_dist",
                "i_min_dist",
                "t_min_dist",
                "closest_match",
                "translation",
            ]
        )
        self.where(_no_errors)
