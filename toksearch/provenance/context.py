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
"""The RunContext contract between toksearch and a provenance backend."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, Tuple

# code.py imports only stdlib, so this is a plain import -- there is no cycle.
from .code import CodeSpec
from .hashing import sha256_of


@dataclass(frozen=True)
class OpSpec:
    """One pipeline operation, described."""

    op: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SourceSpec:
    """Where the records came from."""

    kind: str                       # "shotlist" | "sql" | "recordset" | "unknown"
    count: Optional[int] = None
    hash: Optional[str] = None
    query: Optional[str] = None
    params: Optional[Tuple[Any, ...]] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class BackendSpec:
    """Which compute backend ran, and how it was configured."""

    kind: str
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RunContext:
    """Everything toksearch knows about one ``compute_*`` call.

    This is the entire contract with a provenance backend. A backend receives
    a RunContext and nothing else; it never touches a Pipeline, a Signal, or a
    Record.
    """

    source: SourceSpec
    ops: Tuple[OpSpec, ...]
    signals: Dict[str, dict]
    backend: BackendSpec
    code: CodeSpec
    device: Optional[str] = None
    parent_run: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source": self.source.to_dict(),
            "ops": [op.to_dict() for op in self.ops],
            "signals": self.signals,
            "backend": self.backend.to_dict(),
            "code": self.code.to_dict(),
            "device": self.device,
            "parent_run": self.parent_run,
        }

    def input_identity(self) -> str:
        """Hash of *what data this run reads* -- source plus signals.

        Deliberately excludes ops, backend, and code: two runs that read the
        same data share an input artifact even if they then do different
        things with it. That shared artifact is what connects the lineage
        graph.
        """
        return sha256_of(
            {
                "source": self.source.to_dict(),
                "signals": self.signals,
                "device": self.device,
            }
        )

    def write_directories(self) -> list:
        """Output directories declared by Pipeline.write operations.

        Read from the pipeline definition, not from the records. Ray and Spark
        RecordSets are lazy -- ``SparkRecordSet.map`` returns an un-actioned
        RDD and ``RayRecordSet.map`` returns unmaterialized ObjectRefs -- so
        iterating one to discover written paths would force materialization as
        a side effect of recording provenance. On Spark without caching, the
        user's next action would then recompute the whole pipeline. The
        directory is known statically, so nothing needs to be forced.

        Only ``track="directory"`` writes are covered. ``track="file"`` asks
        for per-shot artifacts, whose paths genuinely are per-record; a backend
        wanting those must iterate, and should say so.
        """
        return [
            op.detail["directory"]
            for op in self.ops
            if op.op == "write" and op.detail.get("track") == "directory"
        ]
