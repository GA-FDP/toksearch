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

import functools
import os
import getpass
import warnings
from importlib.metadata import entry_points
from pathlib import Path

import pymssql
from fdp_schema import load_tokamak, Tokamak

USER_HOME_DIR = str(Path.home())
USERNAME = getpass.getuser()
DEFAULT_PASSWORD_FILE = os.path.join(USER_HOME_DIR, "D3DRDB.sybase_login")


def _read_sybase_login_file(filename):
    with open(filename, "r") as f:
        username, password = [line.strip() for line in f.readlines()]
    return username, password


def connect_d3drdb(**overrides):
    """Deprecated. Use `toksearch_d3d.sql.connect_d3drdb` (or the
    generic `toksearch.sql.mssql.connect_tokamak_sql('d3d', 'd3drdb',
    **overrides)`) instead."""
    warnings.warn(
        "toksearch.sql.mssql.connect_d3drdb is deprecated. Use "
        "toksearch_d3d.sql.connect_d3drdb instead (or "
        "toksearch.sql.mssql.connect_tokamak_sql for non-D3D tokamaks).",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from toksearch_d3d.sql import connect_d3drdb as _impl
    except ImportError as e:
        raise ImportError(
            "toksearch.sql.mssql.connect_d3drdb requires toksearch_d3d "
            "to be installed. Either `conda install toksearch_d3d` or "
            "migrate to toksearch.sql.mssql.connect_tokamak_sql."
        ) from e
    return _impl(**overrides)


def _resolve_credential(
    username: str | None, loc, password_file_override: str | None,
) -> tuple[str, str]:
    """Resolve (username, password) for an mssql SqlLocator.

    Reads a two-line password file (username on line 1, password on
    line 2) from `password_file_override` if given, else from
    `loc.auth.path`. Returns `(file_user, file_pass)` unless `username`
    is explicitly provided, in which case the explicit value wins.

    Raises:
      RuntimeError: no credential source resolvable.
    """
    pf = password_file_override
    if pf is None and loc.auth and loc.auth.kind == "password_file":
        pf = loc.auth.path
    if not pf:
        raise RuntimeError(
            f"No credential source for locator {loc.name!r}. "
            f"Pass `password=` explicitly, or provide a password_file "
            f"(catalog auth.path or `password_file=` kwarg)."
        )
    text = Path(os.path.expanduser(pf)).read_text()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    file_user, file_pass = lines[0], lines[1]
    return (username if username is not None else file_user), file_pass


@functools.cache
def _discover_catalogs() -> dict[str, Tokamak]:
    """Read and validate every YAML contributed via the
    `fdp_schema.catalogs` entry-point group. Cached for process lifetime;
    tests that patch `entry_points` must call `cache_clear()` in setUp."""
    out: dict[str, Tokamak] = {}
    for ep in entry_points(group="fdp_schema.catalogs"):
        tk = load_tokamak(ep.load())
        if tk.name in out:
            raise RuntimeError(
                f"Duplicate tokamak name {tk.name!r}: "
                f"{ep.value} conflicts with a previous entry point"
            )
        out[tk.name] = tk
    return out


def connect_tokamak_sql(
    tokamak: str,
    name: str = "main",
    *,
    host: str | None = None,
    port: int | None = None,
    db: str | None = None,
    username: str | None = None,
    password: str | None = None,
    password_file: str | None = None,
):
    """Connect to a tokamak's SQL database, reading defaults from the
    catalog contributed via the `fdp_schema.catalogs` entry-point group.

    Args:
      tokamak: tokamak name registered via fdp_schema.catalogs (e.g. "d3d").
      name: which SqlLocator within that tokamak. D3D ships "d3drdb".
      host, port, db: catalog overrides. `db` maps to SqlLocator.database.
      username, password, password_file: credential overrides.

    Credential resolution:
      1. Explicit `password` kwarg wins.
      2. Otherwise `password_file` (kwarg if given, else locator.auth.path).
      3. If neither resolves, raise RuntimeError.

    Driver scope: only driver=="mssql" is supported in v1.

    Raises:
      KeyError: unknown tokamak, or no SqlLocator with the given name.
      NotImplementedError: locator's driver is not "mssql".
      RuntimeError: no credential source resolvable.
    """
    catalogs = _discover_catalogs()
    if tokamak not in catalogs:
        raise KeyError(
            f"No tokamak named {tokamak!r}. Available: {sorted(catalogs)}"
        )
    tk = catalogs[tokamak]
    sqls = [l for l in tk.locators if l.kind == "sql" and l.name == name]
    if not sqls:
        avail = sorted(l.name for l in tk.locators if l.kind == "sql")
        raise KeyError(
            f"No sql locator named {name!r} on tokamak {tokamak!r}. "
            f"Available: {avail}"
        )
    loc = sqls[0]

    if loc.driver != "mssql":
        raise NotImplementedError(
            f"connect_tokamak_sql: driver={loc.driver!r} on locator "
            f"{name!r} is not supported in v1; only 'mssql' is implemented."
        )

    if loc.tdsver:
        os.environ.setdefault("TDSVER", loc.tdsver)

    eff_host = host if host is not None else loc.host
    eff_port = port if port is not None else loc.port
    eff_db   = db   if db   is not None else loc.database

    if password is None:
        username, password = _resolve_credential(username, loc, password_file)
    elif username is None:
        # Explicit password without username: preserve Phase 1's OS
        # current-user default so pymssql doesn't receive None.
        username = getpass.getuser()

    return pymssql.connect(
        eff_host, username, password, eff_db,
        port=str(eff_port) if eff_port else None,
    )
