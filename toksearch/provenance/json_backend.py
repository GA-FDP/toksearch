# Copyright 2026 General Atomics
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
"""A dependency-free Provenance backend that writes JSON.

This exists so toksearch's own test suite can exercise every provenance hook
with no CMF, no DVC, and no database. It is also a usable record in its own
right for people who want provenance without running a metadata server.
"""

import json
import os
import uuid
from typing import Optional

from .base import Provenance


class JsonProvenance(Provenance):
    """Record a run to a JSON file."""

    def __init__(self, pipeline_name: str, stage: Optional[str] = None,
                 path: str = "toksearch_run.json", strict: bool = False):
        self.pipeline_name = pipeline_name
        self.stage = stage or pipeline_name
        self.path = path
        self.strict = strict
        self.run_id = uuid.uuid4().hex

        self._context = None
        self._input_identity = None
        self._outputs = []
        self._metrics = {}

    def on_compute_start(self, ctx) -> None:
        self._context = ctx.to_dict()
        self._input_identity = ctx.input_identity()

    def on_compute_end(self, ctx, recordset) -> None:
        self._context = ctx.to_dict()
        self._input_identity = ctx.input_identity()
        # From the context, not the recordset: see RunContext.write_directories.
        for path in ctx.write_directories():
            self.output(path, source="pipeline_write")

    def output(self, *paths, **custom_properties) -> None:
        for path in paths:
            self._outputs.append({"path": str(path), **custom_properties})

    def metrics(self, name: str, values: dict) -> None:
        self._metrics[name] = dict(values)

    def finalize(self) -> None:
        payload = {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "stage": self.stage,
            "input_identity": self._input_identity,
            "context": self._context,
            "outputs": self._outputs,
            "metrics": self._metrics,
        }
        directory = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(directory, exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True, default=repr)
