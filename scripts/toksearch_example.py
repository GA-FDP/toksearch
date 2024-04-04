#!/usr/bin/env python

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

import argparse
from toksearch import Pipeline
from toksearch import MdsSignal, PtDataSignal


def create_pipeline(shots):
    ipmhd_signal = MdsSignal(r"\ipmhd", "efit01")
    ip_signal = PtDataSignal("ip")

    pipeline = Pipeline(shots)
    pipeline.fetch("ipmhd", ipmhd_signal)
    pipeline.fetch("ip", ip_signal)

    @pipeline.map
    def calc_max_ipmhd(rec):
        rec["max_ipmhd"] = np.max(np.abs(rec["ipmhd"]["data"]))

    pipeline.keep(["max_ipmhd"])
    return pipeline


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("backend", choices=["spark", "ray"])
    args = parser.parse_args()

    backend = args.backend

    num_shots = 10000
    shots = list(range(165920, 165920 + num_shots))

    pipeline = create_pipeline(shots)

    if backend == "ray":
        results = pipeline.compute_ray()
    else:  # spark
        results = pipeline.compute_spark(numparts=1000)

    print(f"Got {len(results)} results using {backend}")
