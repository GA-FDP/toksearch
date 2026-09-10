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

"""The Provenance backend interface."""

import warnings
from abc import ABC, abstractmethod


class Provenance(ABC):
    """Receives a RunContext and records it somewhere.

    Implementations must not raise in normal operation: losing a provenance
    record is bad, but losing a completed multi-hour compute is worse. Errors
    are converted to warnings by ``safe_call`` unless ``strict`` is set.
    """

    #: When True, hook failures propagate instead of warning. For CI.
    strict: bool = False

    #: Identifier for this run. Pipeline.compute copies it onto the returned
    #: RecordSet, which is how an in-process pipeline chain links itself:
    #: Pipeline(previous_recordset) reads it back as RunContext.parent_run.
    run_id = None

    @abstractmethod
    def on_compute_start(self, ctx) -> None:
        """Called before the backend runs, with the derived RunContext."""

    @abstractmethod
    def on_compute_end(self, ctx, recordset) -> None:
        """Called after compute returns, with the resulting RecordSet."""

    @abstractmethod
    def output(self, *paths, **custom_properties) -> None:
        """Declare output artifacts that toksearch did not write itself."""

    @abstractmethod
    def metrics(self, name: str, values: dict) -> None:
        """Record a named set of metrics for this run."""

    @abstractmethod
    def finalize(self) -> None:
        """Flush and close the record."""


def safe_call(provenance, hook: str, *args, **kwargs):
    """Invoke a provenance hook without letting it break the pipeline.

    A None provenance is a no-op, so call sites need no branching.
    """
    if provenance is None:
        return None

    try:
        return getattr(provenance, hook)(*args, **kwargs)
    except Exception as e:
        if getattr(provenance, "strict", False):
            raise
        warnings.warn(
            f"Provenance backend {type(provenance).__name__}.{hook} failed "
            f"and was ignored: {e!r}. The pipeline result is unaffected, but "
            f"this run's provenance record is incomplete.",
            RuntimeWarning,
            stacklevel=3,
        )
        return None
