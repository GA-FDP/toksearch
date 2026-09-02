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

"""Per-record output formats for Pipeline.write.

Writers are looked up either by an explicit ``fmt`` name or by the type of the
object being written. Registration is open so device packages and users can add
their own formats.
"""

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple


class UnknownFormat(Exception):
    """Raised when no writer matches the requested format or object type."""


@dataclass(frozen=True)
class Writer:
    fmt: str
    extension: str
    func: Callable[[Any, str], None]
    types: Tuple[type, ...] = ()


_WRITERS = {}
_ORDER = []


def register_writer(fmt, extension, func, types=()):
    """Register a writer. Later registrations of the same fmt replace earlier."""
    writer = Writer(fmt=fmt, extension=extension, func=func, types=tuple(types))
    _WRITERS[fmt] = writer
    if fmt not in _ORDER:
        _ORDER.append(fmt)
    return writer


def writer_for(fmt: Optional[str] = None, obj: Any = None) -> Writer:
    """Find a writer by explicit format name, else by the object's type."""
    if fmt is not None:
        try:
            return _WRITERS[fmt]
        except KeyError:
            raise UnknownFormat(
                f"No writer registered for format {fmt!r}. "
                f"Known formats: {sorted(_WRITERS)}"
            )

    for name in _ORDER:
        writer = _WRITERS[name]
        if writer.types and isinstance(obj, writer.types):
            return writer

    raise UnknownFormat(
        f"No writer knows how to write an object of type "
        f"{type(obj).__name__}. Pass fmt= explicitly, register a writer with "
        f"toksearch.pipeline.writers.register_writer, or use the decorator "
        f"form of Pipeline.write and write the file yourself."
    )


def extension_for(fmt: str) -> str:
    return writer_for(fmt=fmt).extension


def write_object(obj: Any, path: str, fmt: Optional[str] = None) -> str:
    """Write obj to path, returning path."""
    writer_for(fmt=fmt, obj=obj).func(obj, path)
    return path


def _write_netcdf(obj, path):
    obj.to_netcdf(path)


def _write_parquet(obj, path):
    obj.to_parquet(path)


def _write_npy(obj, path):
    import numpy as np

    np.save(path, obj)


def _write_npz(obj, path):
    import numpy as np

    np.savez(path, **obj)


def _write_json(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=repr)


def _register_builtins():
    import numpy as np
    import pandas as pd
    import xarray as xr

    register_writer("netcdf", ".nc", _write_netcdf, (xr.Dataset, xr.DataArray))
    register_writer("parquet", ".parquet", _write_parquet, (pd.DataFrame,))
    register_writer("npy", ".npy", _write_npy, (np.ndarray,))
    register_writer("npz", ".npz", _write_npz, ())
    register_writer("json", ".json", _write_json, (dict, list))


_register_builtins()
