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

from toksearch import Signal, MdsSignal, PtDataSignal


class TecefSignal(Signal):
    _calibrations = {}

    def __init__(self, channel, **kwargs):
        location = kwargs.pop("location", None)
        super().__init__("dummy", None, **kwargs)
        self.channel = channel

        self.sub_signals = {
            "ecevs": PtDataSignal(f"ecevs{channel:02}", ical=4),
            "cc1f": MdsSignal(r"\cc1f", "ece", location=location),
            "cc2f": MdsSignal(r"\cc2f", "ece", location=location),
            "cc3f": MdsSignal(r"\cc3f", "ece", location=location),
            "adosf": MdsSignal(r"\adosf", "ece", location=location),
        }

    def _get_calibration(self, shot, name):
        if name not in self._calibrations:
            self._calibrations[name] = {}

        if shot not in self._calibrations[name]:
            self._calibrations[name][shot] = self.sub_signals[name].fetch(shot)["data"]

        return self._calibrations[name][shot][self.channel - 1]

    def fetch(self, shot):
        cc1f = self._get_calibration(shot, "cc1f")
        cc2f = self._get_calibration(shot, "cc2f")
        cc3f = self._get_calibration(shot, "cc3f")
        adosf = self._get_calibration(shot, "adosf")

        ecevs = self.sub_signals["ecevs"].fetch(shot)

        _ece = ecevs["data"]
        data = _ece * (cc1f / np.sqrt(1 - cc2f * _ece**2) + cc3f * _ece**3) + adosf

        result = {"data": data}
        result["units"] = {"data": "keV"}
        if "times" in ecevs:
            result["times"] = ecevs["times"]
            result["units"]["times"] = "ms"

        self.apply_funcs(result)
        return result

    def cleanup_shot(self, shot):
        for sig in self.sub_signals.values():
            sig.cleanup_shot(shot)

    def cleanup(self):
        for sig in self.sub_signals.values():
            sig.cleanup()
