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
"""Settling the catalog snapshot a run reads from.

The versioned store keeps every version of a shot, and a catalog snapshot
records which version was latest at one moment. "Latest" is therefore a
lookup that moves, and a run that performs it more than once can read its
early shots from one catalog and its later ones from the next -- one result
set assembled from two states of the world, with no error and no record.

Resolving it once is not enough, because the workers are the problem. Under
``compute_multiprocessing``, Ray or Spark each worker builds its own resolver
at its own start time, so they can disagree from the very first fetch. The
snapshot has to be settled *before any worker exists* and carried to them,
and an environment variable is the only channel that survives ``fork``,
``spawn``, a Ray worker on another host and a Spark executor alike.
"""

import os

VAR = "FDP_STORE_SNAPSHOT"

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
# for the same reason. One snapshot per process is not a limitation of this
# module; it is what the backends make true.
_PINNED_THIS_PROCESS = False


class SnapshotConflict(Exception):
    """Two sources named different snapshots for one run.

    Raised rather than resolved by precedence. A run cannot honour two pins,
    and silently preferring either is the failure this module exists to
    remove.
    """


def _resolve(store_root):
    """The newest catalog snapshot under `store_root`.

    Split out so tests can replace it without a store, and so the ptdata
    import stays lazy -- toksearch is device-neutral and most of it never
    touches a store.
    """
    from ptdata import StoreIndex

    return StoreIndex(store_root).current_snapshot


def pin_run(snapshot=None):
    """Settle this run's catalog snapshot and export it. Returns it, or None.

    Precedence, most specific first:

    1. `snapshot` -- named in code, by ``Pipeline.from_snapshot``
    2. ``FDP_STORE_SNAPSHOT`` -- named for the process, by ``fdp run
       --snapshot`` or the user's own export
    3. the newest snapshot, resolved now and frozen

    Code outranks the environment, which is the reverse of the rule for (3):
    there the pipeline is guessing and the environment was told.

    Raises SnapshotConflict when (1) and (2) disagree. Every other failure is
    silent and returns None: a device with no store resolves nothing, an
    install without ptdata reads no store, and an origin that cannot be
    reached will raise a real error, with real context, at the first fetch.
    Nothing was asked for, so nothing is refused.
    """
    global _PINNED_THIS_PROCESS
    existing = os.environ.get(VAR, "")

    if snapshot:
        if existing and existing != snapshot:
            if _PINNED_THIS_PROCESS:
                raise SnapshotConflict(
                    "an earlier run in this process is already pinned to "
                    "{!r}, and this one asks for {!r}. A process reads from "
                    "one snapshot: its worker processes are reused between "
                    "runs and keep the environment they were started with, "
                    "so a second pin would not reach them. To compare "
                    "snapshots, run one per process -- e.g. "
                    "`fdp run --snapshot {} python sweep.py` once per "
                    "snapshot.".format(existing, snapshot, snapshot)
                )
            raise SnapshotConflict(
                "this run is pinned to {!r} by {} and to {!r} in code; they "
                "cannot both be honoured. Drop one -- either unset {} or "
                "remove the snapshot from Pipeline.from_snapshot.".format(
                    existing, VAR, snapshot, VAR
                )
            )
        os.environ[VAR] = snapshot
        _PINNED_THIS_PROCESS = True
        return snapshot

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
