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

import logging
import MDSplus
from MDSplus.connection import MdsIpException

from toksearch.signal.mds import MdsConnectionRegistry

_log = logging.getLogger(__name__)

# Maps a pseudo-treename to fetch(server, reqs). Reqs carrying such a treename
# need no open tree, so they batch into a single group regardless of shot.
# Empty here; device packages (e.g. toksearch_d3d) add theirs via register_device.
_TREELESS_GROUP_FETCHERS = {}

# Attempts per server, with the cached connection dropped between them.
#
# A batch that fails can leave the process's MDSplus.Connection unusable rather
# than merely unlucky. Over fdp:// the relay retires its session whenever a call
# fails -- a call that overran the relay's timeout, most of all -- and answers
# every later call on that token with 502 "unknown or expired session"; a
# dropped TCP connection to a plain mdsip server has the same shape.
# MdsConnectionRegistry hands the same Connection object to every caller in the
# process for its whole life, so nothing ever notices. That is how one slow
# getMany became 26 failures in GA-FDP/imas_composer CI run 33033220932: the
# worker that lost its session failed every remaining fetch, while the other
# worker's session carried on fine.
#
# Redialling costs one round trip on a path that has already failed, and is the
# difference between losing one group and losing the rest of the run.
_ATTEMPTS_PER_SERVER = 2


def _servers() -> list[str]:
    """Default ordered list of MDSplus servers, replaced by register_device()."""
    raise RuntimeError(
        "No MDSplus servers configured for toksearch.interfaces.req_interface; "
        "pass servers= to fetch_many_from_req, or import a device package "
        "(e.g. toksearch_d3d) that calls register_device()."
    )


def register_device(servers=None, treeless_fetchers=None):
    """Set the process-wide defaults that fetch_many_from_req falls back on.

    :param servers: callable returning the ordered server list, called once per
        fetch_many_from_req call. Replaces any earlier registration.
    :param treeless_fetchers: {pseudo-treename: fetch(server, reqs)}, added to
        those already registered.
    """
    global _servers
    if servers is not None:
        _servers = servers
    if treeless_fetchers:
        _TREELESS_GROUP_FETCHERS.update(treeless_fetchers)


def req_key(req):
    """Identity of a req: (mds_path, shot, treename). Matches Requirement.as_key()."""
    return (req.mds_path, req.shot, req.treename)


def _fetch_tree_group_via_server(server, treename, shot, reqs):
    """Fetch all reqs sharing (treename, shot) against server.

    getMany() truncates multi-dimensional string arrays (e.g. EFIT's
    NAMELISTS:KEQDSKS) to their first row, so reqs with isolated=True are
    fetched one by one with get(). The rest share one getMany() round trip.
    """
    conn = MdsConnectionRegistry().connect(server)
    conn.openTree(treename, shot)
    results = {}
    batched_reqs = []
    for req in reqs:
        if not getattr(req, "isolated", False):
            batched_reqs.append(req)
            continue
        try:
            results[req_key(req)] = conn.get(req.mds_path).data()
        except MDSplus.mdsExceptions.MdsException as e:
            results[req_key(req)] = e
    if not batched_reqs:
        return results

    many = conn.getMany()
    for req in batched_reqs:
        many.append(req.mds_path, req.mds_path)
    fetched_data = many.execute()
    for req in batched_reqs:
        try:
            results[req_key(req)] = many.get(req.mds_path).data()
        except MdsIpException:
            # This is needed to propagate the %TREE-E-NODATA exception which is not properly raised if many.get fails
            results[req_key(req)] = MDSplus.mdsExceptions.MdsException(MDSplus.Data.data(fetched_data[req.mds_path]["error"]))
    return results


def _drop_connection(server):
    """Evict the process-wide cached connection to `server`.

    Best effort on purpose: disconnect() talks to the network too, and a
    failure to close a connection that is already suspect says nothing the
    caller needs. Letting it propagate would replace the real fetch error --
    the one that says why the data is missing -- with a cleanup error.
    """
    try:
        MdsConnectionRegistry().disconnect(server)
    except Exception as e:
        _log.debug("Could not cleanly drop the connection to %s (%s)", server, e)


def _fetch_group_with_fallback(fetch, group, servers):
    """Run fetch(server, group) against each of servers in turn.

    fetch performs one batched round trip against the given server, storing any
    per-node failure in-band. Only a whole-group failure (connect, openTree or
    execute raising) trips the fallback to the next server, and if every server
    raises the group's exception is stored in-band per req.

    Each server gets _ATTEMPTS_PER_SERVER goes, with the cached connection
    dropped in between and after -- see that constant for why a failed batch
    means the connection is suspect and not just the call.
    """
    last_exc = None
    for i, server in enumerate(servers):
        for attempt in range(_ATTEMPTS_PER_SERVER):
            try:
                return fetch(server, group)
            except Exception as e:
                last_exc = e
                # Unconditionally, including on the last attempt: leaving a
                # connection that may be dead in the registry carries the
                # failure into every later group, which is the whole point.
                _drop_connection(server)
                if attempt + 1 < _ATTEMPTS_PER_SERVER:
                    _log.info(
                        "Batched fetch via %s failed (%s); dropping the cached "
                        "connection and retrying once",
                        server, e,
                    )
                else:
                    remaining = len(servers) - i - 1
                    _log.warning(
                        "Batched fetch of %d reqs (treename=%s, shot=%s, e.g. %s) "
                        "via %s failed %d times (%s); %s",
                        len(group), group[0].treename, group[0].shot,
                        ", ".join(str(r.mds_path) for r in group[:3]),
                        server, _ATTEMPTS_PER_SERVER, e,
                        f"trying {remaining} more server(s)" if remaining else "giving up",
                    )
                    _log.debug("Full group: %s", group)
    return {req_key(req): last_exc for req in group}


def fetch_many_from_req(reqs, servers=None, treeless_fetchers=None):
    """Fetch many reqs at once as batched getMany() round trips.

    Reqs whose treename has a tree-less fetcher batch into one group per
    treename; the rest batch per (treename, shot). Each group is tried against
    the servers in order.

    A req only needs mds_path, shot and treename attributes. An optional
    isolated=True attribute fetches a tree req with get() instead of in the
    getMany() batch.

    :param servers: ordered servers to try; defaults to the registered ones
        (see register_device).
    :param treeless_fetchers: extra {pseudo-treename: fetch(server, reqs)},
        overriding registered ones of the same name.

    :return: dict mapping each (mds_path, shot, treename) to its fetched value,
        or to the Exception if fetching failed.
    """
    fetchers = {**_TREELESS_GROUP_FETCHERS, **(treeless_fetchers or {})}
    treeless_groups = {}
    tree_groups = {}
    for req in reqs:
        if req.treename in fetchers:
            treeless_groups.setdefault(req.treename, []).append(req)
        else:
            tree_groups.setdefault((req.treename, req.shot), []).append(req)

    results = {}
    if not (treeless_groups or tree_groups):
        return results
    servers = list(servers) if servers is not None else list(_servers())

    for treename, group in treeless_groups.items():
        results.update(
            _fetch_group_with_fallback(fetchers[treename], group, servers)
        )
    for (treename, shot), group in tree_groups.items():
        results.update(
            _fetch_group_with_fallback(
                lambda server, g, tn=treename, sh=shot: _fetch_tree_group_via_server(
                    server, tn, sh, g
                ),
                group,
                servers,
            )
        )

    return results
