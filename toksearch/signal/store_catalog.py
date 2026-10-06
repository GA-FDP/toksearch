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
"""Settling the published catalog a run reads from.

The versioned store keeps every version of a shot, and a catalog
records which version was latest at one moment. "Latest" is therefore a
lookup that moves, and a run that performs it more than once can read its
early shots from one catalog and its later ones from the next -- one result
set assembled from two states of the world, with no error and no record.

Resolving it once is not enough, because the workers are the problem. Under
``compute_multiprocessing``, Ray or Spark each worker builds its own resolver
at its own start time, so they can disagree from the very first fetch. The
catalog has to be settled *before any worker exists* and carried to them,
and an environment variable is the only channel that survives ``fork``,
``spawn``, a Ray worker on another host and a Spark executor alike.
"""

import json
import os

VAR = "FDP_STORE_CATALOG"
OLD_VAR = "FDP_STORE_SNAPSHOT"

#: Shard versions a saved snapshot names, as a compact JSON map
#: ``{"efit01-0": 5}``. An environment variable for the same
#: reason the catalog is one: a shard is resolved in the WORKER,
#: and no Python argument survives fork, spawn or a Ray worker on
#: another host. A few shards per run, so it stays small.
SHARDS_VAR = "FDP_STORE_SHARDS"

SCHEMA = "fdp-snapshot/1"
#: Every saved-snapshot schema this toksearch replays. /2 is /1 plus
#: ``sql_snapshots``, the SQL database snapshots the run read.
SCHEMAS = ("fdp-snapshot/1", "fdp-snapshot/2")

# Set once this process has pinned a run, so a second run can tell "the user
# told us" from "we decided this earlier". The two need different advice: the
# first is a conflict the caller can resolve, the second is a constraint they
# cannot.
#
# Why a run cannot be re-pinned: joblib/loky reuse worker processes across
# Parallel calls, and a worker's environment is fixed when it is spawned. A
# later os.environ change in the driver reaches workers that do not exist
# yet, and no others -- so a second, differently pinned run would execute on
# workers still holding the first pin. Ray and Spark executors are long-lived
# for the same reason. One catalog per process is not a limitation of this
# module; it is what the backends make true.
_PINNED_THIS_PROCESS = False


class SnapshotFileError(Exception):
    """A saved-snapshot file that cannot be replayed.

    Refused rather than partially honoured: replaying half a snapshot reads
    data the citation does not describe, which is worse than not replaying
    it at all.
    """


class RenamedVariable(Exception):
    """The pre-B7b spelling of the run-wide pin.

    Refused rather than honoured. Honouring it would keep working code
    working; ignoring it is the dangerous middle course, because the run
    would then report a catalog it had not read.
    """


class CatalogConflict(Exception):
    """Two sources named different catalogs for one run.

    Raised rather than resolved by precedence. A run cannot honour two pins,
    and silently preferring either is the failure this module exists to
    remove.
    """


def _resolve(store_root):
    """The newest catalog under `store_root`.

    Split out so tests can replace it without a store, and so the ptdata
    import stays lazy -- toksearch is device-neutral and most of it never
    touches a store.
    """
    from ptdata import StoreIndex

    # ptdata's API says `snapshot` where a user says `catalog`; the
    # rename deliberately stops at its library surface and the wire.
    return StoreIndex(store_root).current_snapshot


def pin_run(catalog=None):
    """Settle this run's catalog and export it. Returns it, or None.

    Precedence, most specific first:

    1. `catalog` -- named in code, by ``Pipeline.from_catalog``
    2. ``FDP_STORE_CATALOG`` -- named for the process, by ``fdp run
       --catalog`` or the user's own export
    3. the newest catalog, resolved now and frozen

    Code outranks the environment, which is the reverse of the rule for (3):
    there the pipeline is guessing and the environment was told.

    Raises CatalogConflict when (1) and (2) disagree. Every other failure is
    silent and returns None: a device with no store resolves nothing, an
    install without ptdata reads no store, and an origin that cannot be
    reached will raise a real error, with real context, at the first fetch.
    Nothing was asked for, so nothing is refused.
    """
    global _PINNED_THIS_PROCESS

    if os.environ.get(OLD_VAR, ""):
        raise RenamedVariable(
            "{} was renamed to {}. It names the origin's published catalog "
            "(catalog_<stamp>); a SAVED SNAPSHOT is a file you keep and pass "
            "to Pipeline.from_snapshot. Export {} instead.".format(
                OLD_VAR, VAR, VAR))

    existing = os.environ.get(VAR, "")

    if catalog:
        if existing and existing != catalog:
            if _PINNED_THIS_PROCESS:
                raise CatalogConflict(
                    "an earlier run in this process is already pinned to "
                    "{!r}, and this one asks for {!r}. A process reads from "
                    "one catalog: its worker processes are reused between "
                    "runs and keep the environment they were started with, "
                    "so a second pin would not reach them. To compare "
                    "catalogs, run one per process -- e.g. "
                    "`fdp run --catalog {} python sweep.py` once per "
                    "catalog.".format(existing, catalog, catalog)
                )
            raise CatalogConflict(
                "this run is pinned to {!r} by {} and to {!r} in code; they "
                "cannot both be honoured. Drop one -- either unset {} or "
                "remove the catalog from Pipeline.from_catalog.".format(
                    existing, VAR, catalog, VAR
                )
            )
        os.environ[VAR] = catalog
        _PINNED_THIS_PROCESS = True
        return catalog

    if existing:
        return existing

    root = os.environ.get("FDP_STORE_ROOT", "")
    if not root:
        return None

    try:
        resolved = _resolve(root)
    except Exception:
        # Deliberately broad, and correct only here: this function's contract
        # is that it cannot fail. Whatever went wrong -- no ptdata, an
        # unreachable origin, a malformed root -- is raised again with full
        # context by the first fetch that actually needs the store.
        return None

    if not resolved:
        return None

    os.environ[VAR] = resolved
    _PINNED_THIS_PROCESS = True
    return resolved


def load_snapshot(path):
    """Read and validate a saved-snapshot file. Returns the document.

    Raises SnapshotFileError with the path in the message: a replay that
    cannot say which file it could not read is a bad citizen of a workflow
    where the file is the citation.
    """
    if isinstance(path, str) and path.strip().lower().startswith("catalog_"):
        raise ValueError(
            "{!r} looks like a published catalog, not a saved-snapshot file. "
            "That is Pipeline.from_catalog(...); from_snapshot takes a "
            "path.".format(path))

    try:
        with open(path) as fh:
            doc = json.load(fh)
    except OSError as exc:
        raise SnapshotFileError(
            "cannot read saved snapshot {}: {}".format(path, exc)) from exc
    except ValueError as exc:
        raise SnapshotFileError(
            "{} is not valid JSON: {}".format(path, exc)) from exc

    schema = doc.get("schema")
    if schema not in SCHEMAS:
        raise SnapshotFileError(
            "{} declares schema {!r}; this toksearch reads {}.".format(
                path, schema, " or ".join(repr(s) for s in SCHEMAS)))
    if not doc.get("shots"):
        raise SnapshotFileError(
            "{} names no shots, so there is nothing to replay.".format(path))
    return doc


def pin_shards(shards):
    """Export the shard versions a saved snapshot names. Returns the map."""
    mapping = {e["shard"]: e["version"] for e in shards or ()}
    if mapping:
        os.environ[SHARDS_VAR] = json.dumps(mapping, sort_keys=True)
    return mapping


def pinned_shards():
    """The shard versions in force, as ``{shard: version}``. Never raises.

    Read in the worker, on the tree-opening path, so a malformed value must
    degrade to "nothing pinned" rather than take down every read.
    """
    raw = os.environ.get(SHARDS_VAR, "")
    if not raw:
        return {}
    try:
        got = json.loads(raw)
        return {str(k): int(v) for k, v in got.items()}
    except (ValueError, AttributeError, TypeError):
        return {}


def pin_sql_snapshots(mapping):
    """Export the SQL snapshots a saved snapshot names. Returns the mapping.

    ``{locator name: snapshot id}``, each exported as
    ``FDP_SQL_SNAPSHOT_<NAME>`` -- the variable
    ``toksearch.sql.snapshot.resolve`` honours, so a worker that connects
    reads the pinned snapshot rather than resolving its own.

    Raises ``toksearch.sql.snapshot.SnapshotConflict`` when the environment
    already names a different id, for the reasons ``pin_run`` raises
    CatalogConflict: a run cannot honour two pins, and once this process
    has pinned one, its workers would not see a second.
    """
    from ..sql.snapshot import SnapshotConflict, _pinned, env_var

    mapping = dict(mapping or {})
    for name, sid in mapping.items():
        var = env_var(name)
        existing = os.environ.get(var, "")
        if existing and existing != sid:
            if name in _pinned:
                raise SnapshotConflict(
                    "an earlier run or connection in this process is already "
                    "pinned to {!r}, and this one asks for {!r}. A process "
                    "reads one snapshot per database: its worker processes "
                    "keep the environment they were started with, so a "
                    "second pin would not reach them. Replay each saved "
                    "snapshot in its own process.".format(existing, sid))
            raise SnapshotConflict(
                "this run is pinned to {!r} by {} and to {!r} by its saved "
                "snapshot; they cannot both be honoured. Unset {} to replay "
                "the saved snapshot.".format(existing, var, sid, var))
    for name, sid in mapping.items():
        os.environ[env_var(name)] = sid
        _pinned.add(name)
    return mapping


def _have_duckdb():
    """Whether this process could read a SQL snapshot at all. Split out so
    tests run the same in an environment without DuckDB."""
    import importlib.util

    return importlib.util.find_spec("duckdb") is not None


def settle_sql_snapshots():
    """Settle every registered SQL snapshot locator for this run. Returns
    ``{locator name: snapshot id}`` of what is in force.

    For each ``sql_snapshot`` locator in every catalog registered through
    the ``fdp_schema.catalogs`` entry points, ``resolve`` it with no code
    pin -- the environment's id if there is one, otherwise the newest,
    exported -- so that workers which connect agree with each other and
    with the provenance record, even when the driver never connected.

    The same contract as ``pin_run``: this cannot fail. No DuckDB, no
    token, an unreachable origin, a malformed catalog, a device without
    such a locator -- each settles nothing for that locator, and the first
    ``connect`` raises with real context. Nothing was asked for, so nothing
    is refused.
    """
    if not _have_duckdb():
        return {}               # nothing in this process could read one
    try:
        from ..sql import snapshot
        from ..sql._snapshot_db import _token_for
        from ..sql.mssql import _discover_catalogs

        locators = [l for tk in _discover_catalogs().values()
                    for l in tk.locators if l.kind == "sql_snapshot"]
    except Exception:
        # Deliberately broad, as in pin_run: no toksearch.sql dependencies,
        # an unreadable catalog -- either way there is nothing to settle.
        return {}

    settled = {}
    for loc in locators:
        try:
            token = _token_for(loc)     # before any HTTP, as connect does
            settled[loc.name] = snapshot.resolve(loc, token=token)
        except Exception:
            # The cannot-fail contract, per locator: one unreachable
            # database must not keep another from being settled.
            continue
    return settled
