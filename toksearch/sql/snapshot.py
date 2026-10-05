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
"""Reading a published snapshot of a SQL database.

A *snapshot* is an immutable directory under a `sql_snapshot` locator's
`base_url`: `manifest.json` plus one Parquet file per table. This module
settles which snapshot a process reads (and pins it the way
`toksearch.signal.store_catalog.pin_run` pins a catalog), then opens
DuckDB over the Parquet by HTTP range request and rewrites each statement
from T-SQL. Nothing is downloaded; nothing falls back to another source.

DuckDB, sqlglot and the `httpfs` extension are imported lazily: a device
package that ships a `sql_snapshot` locator declares them, and a
`toksearch` install that never calls this carries none of them.
"""

import fnmatch
import functools
import json
import os
import posixpath
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import warnings
import xml.etree.ElementTree as ET

SCHEMA = "fdp-sql-snapshot/1"
ENV_PREFIX = "FDP_SQL_SNAPSHOT_"

#: manifest `transforms.collation` -> DuckDB default_collation (None = leave)
COLLATIONS = {"nocase": "NOCASE", "binary": None}

CONDA_PACKAGES = "python-duckdb duckdb-extension-httpfs sqlglot"


class SnapshotError(Exception):
    """Anything about locating, reading or querying a snapshot."""


class SnapshotConflict(SnapshotError):
    """Code and environment name different snapshots."""


class SnapshotNotice(UserWarning):
    """Issued once per process: which snapshot is being read, and how to
    reach the live database instead. Silence with
    `warnings.filterwarnings('ignore', category=SnapshotNotice)`."""


def env_var(name):
    """The environment variable that pins locator `name` for a process."""
    return ENV_PREFIX + name.upper()


# -- HTTP ------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


def _request(url, token=None, method="GET", headers=None, body=None, hops=5):
    """One HTTP request: `(status, headers, body)`.

    Follows redirects itself, same method each hop, because urllib refuses
    to redirect a PROPFIND. The Authorization header is sent to the first
    host only -- a Pelican director puts the token in the redirect URL as
    `authz=`, and a bearer token should not be sprayed across hosts.
    4xx/5xx are returned, not raised; callers decide what a 404 means.
    """
    hdrs = dict(headers or {})
    if token:
        hdrs["Authorization"] = "Bearer " + token
    first_host = urllib.parse.urlsplit(url).netloc
    for _ in range(hops):
        req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
        try:
            with _opener.open(req, timeout=60) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (301, 302, 303, 307, 308) and exc.headers.get("Location"):
                url = urllib.parse.urljoin(url, exc.headers["Location"])
                if urllib.parse.urlsplit(url).netloc != first_host:
                    hdrs.pop("Authorization", None)
                continue
            return exc.code, dict(exc.headers), exc.read()
        except urllib.error.URLError as exc:
            raise SnapshotError("cannot reach {}: {}".format(
                _scrub(url), exc.reason)) from exc
    raise SnapshotError("too many redirects from {}".format(_scrub(url)))


def _scrub(text):
    """Tokens out of anything a user might see or log."""
    return re.sub(r"authz=[^&\s'\"]+", "authz=<redacted>", str(text))


# -- where snapshots live ----------------------------------------------------

@functools.lru_cache(maxsize=None)
def _director_endpoint(host):
    status, _, body = _request(
        "https://{}/.well-known/pelican-configuration".format(host), token=None)
    if status != 200:
        raise SnapshotError(
            "pelican federation {} did not answer its well-known document "
            "(HTTP {})".format(host, status))
    return json.loads(body)["director_endpoint"].rstrip("/")


def resolve_base(base_url):
    """A locator's base_url as something DuckDB and urllib can open.

    `https://` is used verbatim. `pelican://host[:port]/path` becomes the
    federation's director URL plus the path, by the well-known document,
    once per process. `file:///dir` becomes a local directory, which is
    how the tests work and how a locally mirrored snapshot would be read.
    """
    u = urllib.parse.urlsplit(base_url)
    if u.scheme == "https":
        return base_url.rstrip("/")
    if u.scheme == "file":
        return u.path.rstrip("/") or "/"
    if u.scheme == "pelican":
        return _director_endpoint(u.hostname) + u.path.rstrip("/")
    raise SnapshotError(
        "sql_snapshot base_url must be pelican://, https:// or file://, "
        "not {!r}".format(base_url))


def is_local(base):
    return not base.startswith("https://")


def list_ids(base, id_pattern, token):
    """Snapshot ids under `base` matching `id_pattern`, oldest first."""
    if is_local(base):
        names = [n for n in os.listdir(base)
                 if os.path.isdir(os.path.join(base, n))]
    else:
        status, _, body = _request(base + "/", token=token,
                                   method="PROPFIND", headers={"Depth": "1"})
        if status != 207:
            raise SnapshotError("cannot list {} (HTTP {})".format(base, status))
        names = []
        for href in ET.fromstring(body).iter("{DAV:}href"):
            name = posixpath.basename(href.text.rstrip("/"))
            if name:
                names.append(name)
    return sorted(n for n in set(names) if fnmatch.fnmatchcase(n, id_pattern))


def fetch_manifest(base, snapshot_id, token):
    """The snapshot's manifest, validated. A missing snapshot is an error
    naming the id and the base: it is never replaced by another."""
    if is_local(base):
        path = os.path.join(base, snapshot_id, "manifest.json")
        try:
            with open(path) as fh:
                doc = json.load(fh)
        except OSError as exc:
            raise SnapshotError("no snapshot {} under {} ({})".format(
                snapshot_id, base, exc)) from exc
    else:
        url = "{}/{}/manifest.json".format(base, snapshot_id)
        status, _, body = _request(url, token=token)
        if status == 404:
            raise SnapshotError("no snapshot {} under {}".format(snapshot_id, base))
        if status in (401, 403):
            raise SnapshotError("not authorized to read {} (HTTP {}); is the "
                                "bearer token valid?".format(_scrub(url), status))
        if status != 200:
            raise SnapshotError("cannot read {} (HTTP {})".format(_scrub(url), status))
        doc = json.loads(body)
    if doc.get("schema") != SCHEMA:
        raise SnapshotError(
            "snapshot {} declares schema {!r}; this toksearch reads {!r}".format(
                snapshot_id, doc.get("schema"), SCHEMA))
    collation = doc.get("transforms", {}).get("collation", "binary")
    if collation not in COLLATIONS:
        raise SnapshotError(
            "snapshot {} declares collation {!r}; this toksearch knows {}".format(
                snapshot_id, collation, sorted(COLLATIONS)))
    return doc
