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

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import random

import numpy as np
from toksearch import Pipeline
from toksearch import MdsSignal
import ray


def create_pipeline(shots, mds_location, efit_tree):
    random.shuffle(shots)
    pipeline = Pipeline(shots)

    pprime_sig = MdsSignal(
        r"\PPRIME",
        efit_tree,
        location=mds_location,
        dims=("norm_flux", "times"),
        data_func=lambda d: d.T,
    )

    cmpr_sig = MdsSignal(
        r"\EFIT_MFILE:CMPR2",
        efit_tree,
        location=mds_location,
        dims=("dim0", "times"),
        data_func=lambda d: d.T,
    )

    psirz_sig = MdsSignal(
        r"\PSIRZ",
        efit_tree,
        location=mds_location,
        dims=("r", "z", "times"),
        data_func=lambda d: d.T,
    )

    # First fetch pprime
    pipeline.fetch_dataset("ds", {"pprime": pprime_sig})

    # Check if there are enough time slices available in pprime
    @pipeline.where
    def enough_efit_data(rec):
        ds = rec["ds"]

        times = ds["pprime"]["times"].dropna("times")
        return len(times) >= 5

    # Now fetch the rest
    ds_sigs = {"cmpr": cmpr_sig, "psirz": psirz_sig}
    pipeline.fetch_dataset("ds", ds_sigs)

    # Align everything with the timebase of cmpr, which is a proxy
    # for whether or not there's a mfile at a time slice.
    pipeline.align("ds", "cmpr", method="nearest")

    def downsample_timebase(ds):
        times = ds["times"].dropna("times").values
        downsampled_times = times[::1]
        return downsampled_times

    # Pass the downample_timebase function as a callback to align, which
    # will then use the calculated timebase
    pipeline.align("ds", downsample_timebase, method="nearest")

    # Just for fun, we'll calculate the number of bytes in
    # dataset
    @pipeline.map
    def num_bytes(rec):
        rec["nbytes"] = rec["ds"].nbytes

    # Throw out anything with errors
    @pipeline.where
    def no_errors(rec):
        return not rec.errors

    return pipeline


if __name__ == "__main__":

    mds_location = "/mnt/beegfs/archives/mdsplus/codes/~t/~j~i/~h~g/~f~e/~d~c"

    url = os.environ.get("RAY_URL", None)

    if url:
        print("USING URL {}".format(url))
        ray.init(redis_address=url)
    else:
        print("USING LOCAL RAY")
        ray.init(object_store_memory=int(100e9))

    shots = np.arange(162163, 177022)
    pipeline = create_pipeline(shots, mds_location, "efit01")

    recs = pipeline.compute_ray(numparts=512)

    print("NUM RECS: {}".format(len(recs)))

    print("EXAMPLE DATASET")
    print(recs[0]["ds"])
    print(recs[0]["nbytes"])

    total_size = 0
    for rec in recs:
        total_size += rec["nbytes"]

    print("TOTAL SIZE {} Gbytes".format(total_size / 1e9))
