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

import functools
import hashlib
import inspect
import json
import re
from typing import Any, Callable, Optional

# repr() of an object with no custom __repr__ embeds its memory address, which
# is different on every run. Stripping it keeps the informative part of the
# repr while making it reproducible.
_MEMORY_ADDRESS = re.compile(r" at 0x[0-9a-fA-F]+")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_repr(obj: Any) -> str:
    """repr(obj) with any embedded memory address removed."""
    return _MEMORY_ADDRESS.sub("", repr(obj))


def _normalize(obj: Any) -> Any:
    """Rewrite obj into a shape json.dumps renders deterministically.

    Three hazards are handled here rather than left to ``json.dumps``:

    * **Sets** iterate in an order that varies with ``PYTHONHASHSEED``, so the
      same set serializes differently on every run. They are sorted.
    * **Non-string dict keys** make ``json.dumps(sort_keys=True)`` raise --
      either outright for unsupported types like tuples, or when comparing
      mixed key types such as ``{1: ..., "1": ...}``. ``default=`` is never
      consulted for keys, so this must be handled before dumping.
    * **Memory addresses** in repr fallbacks, via ``_stable_repr``.
    """
    if isinstance(obj, dict):
        return {
            (key if isinstance(key, str) else _stable_repr(key)): _normalize(value)
            for key, value in obj.items()
        }
    if isinstance(obj, (set, frozenset)):
        return sorted((_normalize(v) for v in obj), key=_stable_repr)
    if isinstance(obj, (list, tuple)):
        return [_normalize(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Serialize obj to a deterministic JSON string.

    Deterministic means: the same logical input produces identical bytes on
    any machine, in any process, on any run. Everything downstream depends on
    this -- two pipeline runs reading the same data are recognized as sharing
    an input artifact only because their descriptions hash identically.

    Keys are sorted at every level, separators carry no incidental whitespace,
    and ``_normalize`` removes the three sources of run-to-run variation (see
    its docstring). Values JSON cannot represent fall back to a repr with
    memory addresses stripped.

    Note: NaN and Infinity serialize as bare ``NaN``/``Infinity`` tokens,
    which is stable and round-trips through Python's own ``json.loads``, but
    is not strict RFC 8259.
    """
    return json.dumps(
        _normalize(obj),
        sort_keys=True,
        separators=(",", ":"),
        default=_stable_repr,
    )


def sha256_of(obj: Any) -> str:
    """Return the hex sha256 digest of ``canonical_json(obj)``."""
    return _sha256_text(canonical_json(obj))


def _source_sha256(func: Callable) -> Optional[str]:
    """Hash a callable's source, or None when the source is unavailable."""
    try:
        return _sha256_text(inspect.getsource(func))
    except (OSError, TypeError):
        # Builtins, C extensions, and callables defined in a REPL or via eval
        # have no retrievable source. That is expected, not an error.
        return None


def callable_spec(func: Optional[Callable]) -> Optional[dict]:
    """Describe a callable well enough to tell two versions of it apart.

    Returns None for None. Source is captured as a hash rather than as text so
    the record stays small and does not leak surrounding code.

    The fingerprint must be stable across runs. That rules out ``repr(func)``
    as a fallback name: ``functools.partial`` objects and instances of classes
    with ``__call__`` -- both ordinary shapes for a pipeline map function --
    have no ``__name__``, and their repr embeds a memory address that changes
    every run. Each is handled explicitly instead.
    """
    if func is None:
        return None

    if isinstance(func, functools.partial):
        # Identify the wrapped function, and record the bound arguments: they
        # are what distinguishes partial(f, level=5) from partial(f, level=6),
        # which is exactly the distinction provenance needs to draw.
        inner = callable_spec(func.func)
        return {
            "name": inner["name"],
            "module": inner["module"],
            "source_sha256": inner["source_sha256"],
            "partial_args": [_stable_repr(a) for a in func.args],
            "partial_keywords": {
                k: _stable_repr(v) for k, v in sorted(func.keywords.items())
            },
        }

    name = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if name is not None:
        return {
            "name": name,
            "module": getattr(func, "__module__", None),
            "source_sha256": _source_sha256(func),
        }

    # A callable object. Identify it by its class -- deterministic, where its
    # own repr is not -- and hash the class source, which is the code that
    # actually runs.
    cls = type(func)
    return {
        "name": f"{cls.__module__}.{cls.__qualname__}",
        "module": cls.__module__,
        "source_sha256": _source_sha256(cls),
    }
