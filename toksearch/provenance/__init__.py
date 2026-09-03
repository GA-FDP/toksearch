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
"""Provenance recording for toksearch pipelines.

This subpackage is deliberately free of any provenance-*backend* dependency.
It derives a ``RunContext`` describing a pipeline run and hands it to a
``Provenance`` implementation. The CMF implementation lives in the separate
``toksearch_cmf`` package.
"""

from .hashing import canonical_json, sha256_of, callable_spec
from .context import RunContext, SourceSpec, OpSpec, BackendSpec
from .code import CodeSpec, capture_code
from .base import Provenance, safe_call
from .json_backend import JsonProvenance

__all__ = [
    "canonical_json",
    "sha256_of",
    "callable_spec",
    "RunContext",
    "SourceSpec",
    "OpSpec",
    "BackendSpec",
    "CodeSpec",
    "capture_code",
    "Provenance",
    "safe_call",
    "JsonProvenance",
]
