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
"""Deterministic serialization and hashing helpers.

Provenance identity depends on these being stable: the same logical input must
produce the same bytes on every machine and every run, or CMF cannot recognize
two runs as sharing an input artifact.
"""

import hashlib
import inspect
import json
from typing import Any, Callable, Optional


def canonical_json(obj: Any) -> str:
    """Serialize obj to a deterministic JSON string.

    Keys are sorted at every level and separators carry no incidental
    whitespace. Objects JSON cannot represent fall back to ``repr``, which
    keeps a weak-but-present record rather than raising mid-pipeline.
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )


def sha256_of(obj: Any) -> str:
    """Return the hex sha256 digest of ``canonical_json(obj)``."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def callable_spec(func: Optional[Callable]) -> Optional[dict]:
    """Describe a callable well enough to tell two versions of it apart.

    Returns None for None. Source is captured as a hash rather than as text so
    the record stays small and does not leak surrounding code.
    """
    if func is None:
        return None

    source_sha256 = None
    try:
        source_sha256 = hashlib.sha256(
            inspect.getsource(func).encode("utf-8")
        ).hexdigest()
    except (OSError, TypeError):
        # Builtins, C extensions, and callables defined in a REPL or via eval
        # have no retrievable source. That is expected, not an error.
        pass

    return {
        "name": getattr(func, "__qualname__", None) or getattr(func, "__name__", repr(func)),
        "module": getattr(func, "__module__", None),
        "source_sha256": source_sha256,
    }
