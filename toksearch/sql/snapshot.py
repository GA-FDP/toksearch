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

#: a snapshot id ends in the UTC stamp it was taken at
STAMP_RE = re.compile(r"\d{8}T\d{6}Z\Z")

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
    to redirect a PROPFIND. Only http(s) redirects are followed, and an
    https -> http downgrade is refused. The Authorization header is sent to
    the first host only -- a Pelican director puts the token in the redirect
    URL as `authz=`, and a bearer token should not be sprayed across hosts.
    Once dropped on a cross-host hop it stays dropped for the rest of the
    chain, deliberately: a later hop back to the first host does not get it
    back. 4xx/5xx are returned, not raised; callers decide what a 404
    means. Network and read failures are SnapshotError.
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
                new = urllib.parse.urljoin(url, exc.headers["Location"])
                old_scheme = urllib.parse.urlsplit(url).scheme
                new_scheme = urllib.parse.urlsplit(new).scheme
                if new_scheme not in ("http", "https") or (
                        old_scheme == "https" and new_scheme == "http"):
                    raise SnapshotError("refusing redirect from {} to {}".format(
                        _scrub(url), _scrub(new))) from exc
                url = new
                if urllib.parse.urlsplit(url).netloc != first_host:
                    hdrs.pop("Authorization", None)
                continue
            try:
                err_body = exc.read()
            except OSError as exc2:
                raise SnapshotError("cannot read {}: {}".format(
                    _scrub(url), exc2)) from exc2
            return exc.code, dict(exc.headers), err_body
        except urllib.error.URLError as exc:
            raise SnapshotError("cannot reach {}: {}".format(
                _scrub(url), exc.reason)) from exc
        except OSError as exc:
            raise SnapshotError("cannot read {}: {}".format(
                _scrub(url), exc)) from exc
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
    try:
        return json.loads(body)["director_endpoint"].rstrip("/")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SnapshotError(
            "pelican federation {} returned a well-known document without a "
            "usable director_endpoint".format(host)) from exc


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
        return _director_endpoint(u.netloc) + u.path.rstrip("/")
    raise SnapshotError(
        "sql_snapshot base_url must be pelican://, https:// or file://, "
        "not {!r}".format(base_url))


def is_local(base):
    return not base.startswith("https://")


def list_ids(base, id_pattern, token):
    """Snapshot ids under `base` matching `id_pattern`, oldest first.

    An id ends in a UTC stamp `YYYYMMDDTHHMMSSZ`, and the order is by that
    stamp, whatever precedes it, so the last id is the latest. Anything else beside the snapshots -- a file, a
    scratch directory, a half-uploaded directory under another name -- is
    ignored, so it can never be chosen as the newest.
    """
    if is_local(base):
        names = [n for n in os.listdir(base)
                 if os.path.isdir(os.path.join(base, n))]
    else:
        status, _, body = _request(base + "/", token=token,
                                   method="PROPFIND", headers={"Depth": "1"})
        if status != 207:
            raise SnapshotError("cannot list {} (HTTP {})".format(base, status))
        try:
            hrefs = list(ET.fromstring(body).iter("{DAV:}href"))
        except ET.ParseError as exc:
            raise SnapshotError(
                "cannot parse the listing of {}: {}".format(base, exc)) from exc
        names = []
        for href in hrefs:
            if not href.text:
                continue
            name = posixpath.basename(urllib.parse.unquote(href.text).rstrip("/"))
            if name:
                names.append(name)
    return sorted((n for n in set(names)
                   if fnmatch.fnmatchcase(n, id_pattern) and STAMP_RE.search(n)),
                  key=lambda n: STAMP_RE.search(n).group())


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
        except ValueError as exc:
            raise SnapshotError("{} is not valid JSON: {}".format(path, exc)) from exc
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
        try:
            doc = json.loads(body)
        except ValueError as exc:
            raise SnapshotError("{} is not valid JSON: {}".format(
                _scrub(url), exc)) from exc
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


# -- which snapshot this process reads ----------------------------------------

# Locator names pinned by THIS process, so a second pin can be told apart
# from one the user exported. Same device as store_catalog._PINNED_THIS_PROCESS.
_pinned = set()


def catalog_pairing(name, token):
    """The snapshot the process's catalog was built against, or None.

    Reads `<FDP_STORE_ROOT>/catalog/<FDP_STORE_CATALOG>/meta.json`, key
    `sql_snapshots.<name>`. D3 writes that file; until it exists this
    returns None and the caller moves on. A file that exists but cannot
    be read is an error: the catalog said something and we could not hear
    it, which is not the same as it saying nothing.
    """
    root = os.environ.get("FDP_STORE_ROOT", "")
    catalog = os.environ.get("FDP_STORE_CATALOG", "")
    if not root or not catalog:
        return None
    base = resolve_base(root)
    if is_local(base):
        path = os.path.join(base, "catalog", catalog, "meta.json")
        try:
            with open(path) as fh:
                raw = fh.read()
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise SnapshotError("cannot read catalog metadata {}: {}".format(
                path, exc)) from exc
        where = path
    else:
        where = "{}/catalog/{}/meta.json".format(base, catalog)
        status, _, raw = _request(where, token=token)
        if status == 404:
            return None
        if status != 200:
            raise SnapshotError("cannot read catalog metadata {} (HTTP {})".format(
                _scrub(where), status))
    try:
        pairings = json.loads(raw).get("sql_snapshots", {})
        value = pairings.get(name)
    except (ValueError, AttributeError) as exc:
        raise SnapshotError("catalog metadata {} is malformed: {}".format(
            _scrub(where), exc)) from exc
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise SnapshotError(
            "catalog {} pairs {} with {!r}, which is not a snapshot id".format(
                catalog, name, value))
    return value


def resolve(locator, snapshot=None, token=None):
    """Settle the snapshot id for `locator` in this process and export it.

    Precedence, most specific first:

    1. `snapshot` -- named in code
    2. `FDP_SQL_SNAPSHOT_<NAME>` -- named for the process by `fdp run`, a
       saved-snapshot replay, or the user's own export
    3. the pairing the process's catalog records (`catalog_pairing`)
    4. the newest published under the locator's base_url

    Code outranks the environment, but a disagreement raises
    SnapshotConflict rather than picking one. One snapshot per process:
    workers keep the environment they were started with, so a second,
    different pin would not reach them (store_catalog.pin_run says why).
    (3) and (4) are errors when they name nothing readable -- never a
    fallback to another tier.
    """
    var = env_var(locator.name)
    existing = os.environ.get(var, "")

    if snapshot:
        if existing and existing != snapshot:
            if locator.name in _pinned:
                raise SnapshotConflict(
                    "an earlier connection in this process is already pinned "
                    "to {!r}, and this one asks for {!r}. A process reads one "
                    "snapshot per database: its worker processes keep the "
                    "environment they were started with, so a second pin would "
                    "not reach them. Run one snapshot per process.".format(
                        existing, snapshot))
            raise SnapshotConflict(
                "this process is pinned to {!r} by {} and asks for {!r} in "
                "code; they cannot both be honoured. Unset {} or drop the "
                "snapshot= argument.".format(existing, var, snapshot, var))
        os.environ[var] = snapshot
        _pinned.add(locator.name)
        return snapshot

    if existing:
        return existing

    base = resolve_base(locator.base_url)
    sid = catalog_pairing(locator.name, token)
    if sid is None:
        ids = list_ids(base, locator.id_pattern, token)
        if not ids:
            raise SnapshotError(
                "no snapshot matching {!r} is published under {}".format(
                    locator.id_pattern, locator.base_url))
        sid = ids[-1]
    os.environ[var] = sid
    _pinned.add(locator.name)
    return sid
