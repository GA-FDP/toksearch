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
import os
import socket
import MDSplus
from MDSplus.connection import MdsIpException

from toksearch.signal.mds import MdsConnectionRegistry

_log = logging.getLogger(__name__)

_PTDATA_TREENAME = "__ptdata__"
_FDP_THINCLIENT_SERVER = "fdp://fdp-d3d-origin.nationalresearchplatform.org:8443/mdsip"
_FALLBACK_MDS_SERVER = "atlas.gat.com"
_ATLAS_PORT = 8000

# Set in environments (e.g. GitHub CI) that cannot reach atlas, so a failed
# FDP thin-client fetch falls through to an atlas attempt that raises
# immediately instead of hanging on an unreachable connection.
_NO_ATLAS_ENV_VAR = "TOKSEARCH_D3D_NO_ATLAS"

# Tri-state cache (None = unchecked) for whether atlas is directly reachable
# from this process. Populated on first use by _atlas_reachable().
_atlas_reachable_cache = None


def _atlas_reachable(timeout=2.0):
    """Whether atlas is directly reachable from this process, cached process-wide.

    A raw TCP precheck is used instead of just trying MDSplus.Connection and
    catching failure, because MDSplus.Connection connects via a native call
    with no timeout -- on an unreachable host that can hang far longer than
    the toksearch round trip this check exists to avoid.
    """
    global _atlas_reachable_cache
    if os.environ.get(_NO_ATLAS_ENV_VAR):
        return False
    if _atlas_reachable_cache is None:
        try:
            with socket.create_connection((_FALLBACK_MDS_SERVER, _ATLAS_PORT), timeout=timeout):
                _atlas_reachable_cache = True
        except OSError:
            _atlas_reachable_cache = False
    return _atlas_reachable_cache


def _req_key(req):
    """Identity of a req: (mds_path, shot, treename). Matches Requirement.as_key()."""
    return (req.mds_path, req.shot, req.treename)


def _fetch_tree_group_via_server(server, treename, shot, reqs):
    """Fetch all reqs sharing (treename, shot) in one getMany() round trip against server."""
    conn = MdsConnectionRegistry().connect(server)
    conn.openTree(treename, shot)
    many = conn.getMany()
    for req in reqs:
        many.append(req.mds_path, req.mds_path)
    fetched_data = many.execute()
    results = {}
    for req in reqs:
        try:
            results[_req_key(req)] = many.get(req.mds_path).data()
        except MdsIpException:
            # This is needed to propagate the %TREE-E-NODATA exception which is not properly raised if many.get fails
            results[_req_key(req)] = MDSplus.mdsExceptions.MdsException(MDSplus.Data.data(fetched_data[req.mds_path]["error"]))
    return results


def _fetch_ptdata_group_via_server(server, reqs):
    """Fetch all ptdata reqs in one getMany() round trip against server.

    ptdata2() needs no open tree, so every ptdata req -- regardless of shot --
    batches into a single getMany. Each req expands to the three TDI forms the
    omas machine mappings emit (data, times, rarray) and is reassembled into the
    {data, times, rarray} dict the compose functions expect.
    """
    conn = MdsConnectionRegistry().connect(server)
    many = conn.getMany()
    for i, req in enumerate(reqs):
        many.append(f"d{i}", f'ptdata2("{req.mds_path}", {req.shot})')
        many.append(f"t{i}", f'dim_of(ptdata2("{req.mds_path}", {req.shot}), 0)')
        many.append(f"r{i}", f'pthead2("{req.mds_path}", {req.shot}), __rarray')
    many.execute()
    results = {}
    for i, req in enumerate(reqs):
        try:
            results[_req_key(req)] = {
                "data": many.get(f"d{i}").data(),
                "times": many.get(f"t{i}").data(),
                "rarray": many.get(f"r{i}").data(),
            }
        except Exception as e:
            results[_req_key(req)] = e
    return results


def _fetch_group_with_fallback(fetch, group):
    """Run fetch(server, group) against the FDP thin client, then atlas.

    fetch performs one batched round trip against the given server, storing any
    per-node failure in-band. Only a whole-group failure (connect, openTree or
    execute raising) trips the fallback: FDP first, atlas next when reachable,
    and if both raise the group's exception is stored in-band per req.
    """
    servers = [_FDP_THINCLIENT_SERVER]
    if _atlas_reachable():
        servers.append(_FALLBACK_MDS_SERVER)

    last_exc = None
    for server in servers:
        try:
            return fetch(server, group)
        except Exception as e:
            _log.warning("Batched fetch via %s failed (%s); trying next server", server, e)
            _log.warning(f"Attempted to fetch {group}")
            last_exc = e
    return {_req_key(req): last_exc for req in group}


def fetch_many_from_req(reqs):
    """Fetch many reqs at once as batched getMany() round trips.

    ptdata reqs (treename == "__ptdata__") batch together into a single
    tree-less getMany; the rest batch per (treename, shot). Each group is tried
    against the FDP thin client first, then atlas.

    A req only needs mds_path, shot and treename attributes. Despite its name
    mds_path can also be a PTDATA point name.

    :return: dict mapping each (mds_path, shot, treename) to its fetched value,
        or to the Exception if fetching failed.
    """
    ptdata_reqs = []
    tree_groups = {}
    for req in reqs:
        if req.treename == _PTDATA_TREENAME:
            ptdata_reqs.append(req)
        else:
            tree_groups.setdefault((req.treename, req.shot), []).append(req)

    results = {}
    if ptdata_reqs:
        results.update(
            _fetch_group_with_fallback(_fetch_ptdata_group_via_server, ptdata_reqs)
        )
    for (treename, shot), group in tree_groups.items():
        results.update(
            _fetch_group_with_fallback(
                lambda server, g, tn=treename, sh=shot: _fetch_tree_group_via_server(
                    server, tn, sh, g
                ),
                group,
            )
        )

    return results