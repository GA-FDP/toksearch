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

import collections
import contextlib
import fnmatch
import hashlib
import json
import os
import posixpath
import random
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

SCHEMA = "fdp-sql-snapshot/1"
ENV_PREFIX = "FDP_SQL_SNAPSHOT_"

#: manifest `transforms.collation` -> DuckDB default_collation (None = leave)
COLLATIONS = {"nocase": "NOCASE", "binary": None}

#: a snapshot id ends in the UTC stamp it was taken at
STAMP_RE = re.compile(r"\d{8}T\d{6}Z\Z")

CONDA_PACKAGES = "python-duckdb duckdb-extension-httpfs sqlglot"

__all__ = [
    "connect", "connect_tokamak", "locator_for", "resolve", "resolve_base",
    "list_ids", "fetch_manifest", "catalog_pairing", "SnapshotConnection",
    "SnapshotCursor", "SnapshotError", "SnapshotConflict", "SnapshotNotice",
    "env_var", "token_for", "verify_files", "VerifyResult", "SCHEMA",
]


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


def token_for(locator):
    """The bearer token the locator's AuthHint names, or None for a local
    base or an `AuthHint(kind="none")` (a public server). Missing for a
    remote base is an error before any HTTP: locality is read from the
    base_url's scheme (`file://` is local), not from `resolve_base`, which
    for `pelican://` already makes a request."""
    if urllib.parse.urlsplit(locator.base_url).scheme == "file":
        return None
    auth = locator.auth
    if auth is not None and auth.kind == "none":
        return None
    if auth is None or auth.kind != "bearer_token" or not auth.env:
        raise SnapshotError(
            "sql_snapshot locator {!r} names no bearer_token env var; nothing "
            "to authenticate with".format(locator.name))
    token = os.environ.get(auth.env, "")
    if not token:
        raise SnapshotError(
            "{} is not set; run under `fdp run`, or `fdp login`".format(auth.env))
    return token


# -- HTTP ------------------------------------------------------------------

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_opener = urllib.request.build_opener(_NoRedirect)


@contextlib.contextmanager
def _open(url, token=None, method="GET", headers=None, body=None, hops=5,
          timeout=60):
    """One HTTP request, following redirects: `(status, headers, response,
    url)` with the response open for reading (an HTTPError for 4xx/5xx).

    Follows redirects itself, same method each hop, because urllib refuses
    to redirect a PROPFIND. Only http(s) redirects are followed, and an
    https -> http downgrade is refused. The Authorization header is sent to
    the first host only -- a Pelican director puts the token in the redirect
    URL as `authz=`, and a bearer token should not be sprayed across hosts.
    Once dropped on a cross-host hop it stays dropped for the rest of the
    chain, deliberately: a later hop back to the first host does not get it
    back. 4xx/5xx are returned, not raised; callers decide what a 404
    means. Network failures are SnapshotError; read failures are the
    caller's to turn into one (`_request` and `_stream` do).
    """
    hdrs = dict(headers or {})
    # osg-htc.org rejects urllib's default User-Agent with 403.
    if not any(k.lower() == "user-agent" for k in hdrs):
        try:
            from toksearch import __version__
            hdrs["User-Agent"] = "toksearch-sql-snapshot/{}".format(__version__)
        except Exception:
            hdrs["User-Agent"] = "toksearch-sql-snapshot"
    if token:
        hdrs["Authorization"] = "Bearer " + token
    first_host = urllib.parse.urlsplit(url).netloc
    for _ in range(hops):
        req = urllib.request.Request(url, data=body, method=method, headers=hdrs)
        try:
            resp = _opener.open(req, timeout=timeout)
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
            with exc:           # an error response holds a socket too
                yield exc.code, dict(exc.headers), exc, url
            return
        except urllib.error.URLError as exc:
            raise SnapshotError("cannot reach {}: {}".format(
                _scrub(url), exc.reason)) from exc
        except OSError as exc:
            raise SnapshotError("cannot read {}: {}".format(
                _scrub(url), exc)) from exc
        with resp:
            yield resp.status, dict(resp.headers), resp, url
        return
    raise SnapshotError("too many redirects from {}".format(_scrub(url)))


def _request(url, token=None, method="GET", headers=None, body=None, hops=5,
             timeout=60):
    """One HTTP request: `(status, headers, body)`. Redirects, auth and
    errors as `_open`; a failure reading the body is SnapshotError.
    `timeout` is per socket operation, each hop."""
    with _open(url, token=token, method=method, headers=headers, body=body,
               hops=hops, timeout=timeout) as (status, hdrs, resp, final):
        try:
            data = resp.read()
        except OSError as exc:
            raise SnapshotError("cannot read {}: {}".format(
                _scrub(final), exc)) from exc
    return status, hdrs, data


class _HttpStatus(SnapshotError):
    """A GET that `_stream` was asked to read answered something else."""

    def __init__(self, message, status):
        super().__init__(message)
        self.status = status


def _stream(url, token=None, chunk=1 << 20, timeout=60):
    """GET `url` and yield its body in `chunk`-byte pieces, so a 1 GB file
    is hashed without being held. Redirects and auth as `_request`. Any
    status but 200 is `_HttpStatus` (a SnapshotError carrying `.status`);
    a failure mid-body is SnapshotError."""
    with _open(url, token=token, timeout=timeout) as (status, _, resp, final):
        if status != 200:
            if status in (401, 403):
                msg = "not authorized to read {} (HTTP {}); is the bearer token valid?"
            else:
                msg = "cannot read {} (HTTP {})"
            raise _HttpStatus(msg.format(_scrub(final), status), status)
        while True:
            try:
                piece = resp.read(chunk)
            except OSError as exc:
                raise SnapshotError("cannot read {}: {}".format(
                    _scrub(final), exc)) from exc
            if not piece:
                return
            yield piece


def _scrub(text):
    """Tokens out of anything a user might see or log."""
    return re.sub(r"authz=[^&\s'\"]+", "authz=<redacted>", str(text))


# -- where snapshots live ----------------------------------------------------

_DIRECTORS = {}


def _director_endpoint(host, timeout=60):
    """The federation's director URL, once per process per host. Cached by
    host alone: how long one lookup was willing to wait does not change
    the answer."""
    if host in _DIRECTORS:
        return _DIRECTORS[host]
    status, _, body = _request(
        "https://{}/.well-known/pelican-configuration".format(host), token=None,
        timeout=timeout)
    if status != 200:
        raise SnapshotError(
            "pelican federation {} did not answer its well-known document "
            "(HTTP {})".format(host, status))
    try:
        endpoint = json.loads(body)["director_endpoint"].rstrip("/")
    except (ValueError, KeyError, TypeError, AttributeError) as exc:
        raise SnapshotError(
            "pelican federation {} returned a well-known document without a "
            "usable director_endpoint".format(host)) from exc
    _DIRECTORS[host] = endpoint
    return endpoint


_director_endpoint.cache_clear = _DIRECTORS.clear


def resolve_base(base_url, timeout=60):
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
    if u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost"):
        return base_url.rstrip("/")      # the test server only; never a real origin
    if u.scheme == "pelican":
        return _director_endpoint(u.netloc, timeout=timeout) + u.path.rstrip("/")
    raise SnapshotError(
        "sql_snapshot base_url must be pelican://, https:// or file://, "
        "not {!r}".format(base_url))


def is_local(base):
    return not base.startswith(("https://", "http://"))


def list_ids(base, id_pattern, token, timeout=60):
    """Snapshot ids under `base` matching `id_pattern`, oldest first.

    An id ends in a UTC stamp `YYYYMMDDTHHMMSSZ`, and the order is by that
    stamp, whatever precedes it, so the last id is the latest (ties broken
    by the whole id, so the order is deterministic). Anything else beside
    the snapshots -- a file, a scratch directory, a half-uploaded directory
    under another name -- is ignored, so it can never be chosen as the
    newest.
    """
    if is_local(base):
        names = [n for n in os.listdir(base)
                 if os.path.isdir(os.path.join(base, n))]
    else:
        status, _, body = _request(base + "/", token=token,
                                   method="PROPFIND", headers={"Depth": "1"},
                                   timeout=timeout)
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
                  key=lambda n: (STAMP_RE.search(n).group(), n))


def _client_version():
    """toksearch's version for error messages; imported lazily because this
    module may be imported while the toksearch package is initialising."""
    try:
        import toksearch
        return "version " + str(toksearch.__version__)
    except (ImportError, AttributeError):
        return "version unknown"


def fetch_manifest(base, snapshot_id, token, timeout=60):
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
        status, _, body = _request(url, token=token, timeout=timeout)
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
            "snapshot {} declares schema {!r}; this toksearch ({}) reads {!r}".format(
                snapshot_id, doc.get("schema"), _client_version(), SCHEMA))
    collation = doc.get("transforms", {}).get("collation", "binary")
    if collation not in COLLATIONS:
        raise SnapshotError(
            "snapshot {} declares collation {!r}; this toksearch ({}) knows {}".format(
                snapshot_id, collation, _client_version(), sorted(COLLATIONS)))
    return doc


# -- which snapshot this process reads ----------------------------------------

# Locator names pinned by THIS process, so a second pin can be told apart
# from one the user exported. Same device as store_catalog._PINNED_THIS_PROCESS.
_pinned = set()


def catalog_pairing(name, token):
    """The snapshot the process's catalog was built against, or None.

    Reads `<FDP_STORE_ROOT>/catalog/<FDP_STORE_CATALOG>/meta.json`, key
    `sql_snapshots.<name>`; None when the file does not exist. A record of
    what the catalog was built beside: `resolve` does not consult it. A file that exists but cannot
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


def settled_conflict(name, existing, wanted):
    """The message for asking a process for a second snapshot of `name`
    once it has settled one -- by a connect, or by Pipeline.compute()
    settling the newest. Shared with store_catalog.pin_sql_snapshots."""
    var = env_var(name)
    return (
        "{name} is already settled to {existing!r} in this process (by an "
        "earlier connect or Pipeline.compute(), which settles the newest when "
        "nothing is pinned); workers started since keep it, so {wanted!r} "
        "cannot be honoured here. To read {wanted!r}: start a fresh process "
        "and either pass snapshot={wanted!r} before the first "
        "connect/compute(), export {var}={wanted!r}, or replay with "
        "Pipeline.from_snapshot(...).".format(
            name=name, existing=existing, wanted=wanted, var=var))


def resolve(locator, snapshot=None, token=None, timeout=60):
    """Settle the snapshot id for `locator` in this process and export it.

    Precedence, most specific first:

    1. `snapshot` -- named in code
    2. `FDP_SQL_SNAPSHOT_<NAME>` -- named for the process by `fdp run`, a
       saved-snapshot replay, or the user's own export
    3. the newest published under the locator's base_url

    Code outranks the environment, but a disagreement raises
    SnapshotConflict rather than picking one. One snapshot per process:
    workers keep the environment they were started with, so a second,
    different pin would not reach them (store_catalog.pin_run says why).
    (3) is an error when nothing readable is published -- never a fallback
    to another source.

    The pairing a published catalog records (`catalog_pairing`) is not
    consulted. Catalogs are republished rarely, so making it the default
    would hold every unpinned user on an old snapshot until the next
    catalog, and it buys no reproducibility: that comes from the run
    recording the id it read, in its saved snapshot and its provenance. The
    pairing stays a record of what a catalog was built beside, not a lookup.
    """
    var = env_var(locator.name)
    existing = os.environ.get(var, "")

    if snapshot:
        if existing and existing != snapshot:
            if locator.name in _pinned:
                raise SnapshotConflict(settled_conflict(locator.name, existing, snapshot))
            raise SnapshotConflict(
                "this process is pinned to {!r} by {} and asks for {!r} in "
                "code; they cannot both be honoured. Unset {} or drop the "
                "snapshot= argument.".format(existing, var, snapshot, var))
        os.environ[var] = snapshot
        _pinned.add(locator.name)
        return snapshot

    if existing:
        return existing

    if token is None:
        token = token_for(locator)      # before resolve_base: no HTTP without it
    base = resolve_base(locator.base_url, timeout=timeout)
    ids = list_ids(base, locator.id_pattern, token, timeout=timeout)
    if not ids:
        raise SnapshotError(
            "no snapshot matching {!r} is published under {}".format(
                locator.id_pattern, locator.base_url))
    sid = ids[-1]
    os.environ[var] = sid
    _pinned.add(locator.name)
    return sid


# -- checking the bytes ------------------------------------------------------

#: `verify_files`' answer. `failures` lists `(path, expected_sha256,
#: actual_sha256)`, path as the manifest names it, actual None when the
#: file is missing. `checked < total` means a sample: the rest is unchecked.
VerifyResult = collections.namedtuple("VerifyResult", "checked total failures")


def verify_files(locator, snapshot_id, sample=None):
    """Hash every Parquet file of a published snapshot against its manifest.

    The byte check a range read cannot make: each file the manifest lists
    is streamed in full, over the same path the client reads (the same
    base, token and redirects), through SHA-256, and its size and digest
    compared with the manifest's `bytes` and `sha256`. A `file://` base
    hashes the local files.

    `sample=N` (N >= 1) checks the files of N tables, chosen
    deterministically from the snapshot id, so a repeated sampled check
    reads the same tables.

    Returns VerifyResult(checked, total, failures), counting files. A
    mismatch is reported, not raised. So is a file that cannot be read --
    missing (actual None) or failing mid-stream (actual "<error: ...>") --
    so that one bad file does not lose the failures already collected. A
    manifest that cannot be found or read is SnapshotError, never a pass.
    """
    if sample is not None and sample < 1:
        raise ValueError("sample must be None or >= 1, not {!r}".format(sample))
    token = token_for(locator)            # before resolve_base: no HTTP without it
    base = resolve_base(locator.base_url)
    manifest = fetch_manifest(base, snapshot_id, token)

    tables = sorted(manifest.get("tables", []), key=lambda t: t["name"])
    total = sum(len(t.get("files", [])) for t in tables)
    if sample is not None and sample < len(tables):
        tables = random.Random(snapshot_id).sample(tables, sample)

    checked, failures = 0, []
    for table in tables:
        for entry in table.get("files", []):
            path = entry["path"]
            digest, size = hashlib.sha256(), 0
            try:
                for piece in _file_chunks(base, snapshot_id, path, token):
                    digest.update(piece)
                    size += len(piece)
                actual = digest.hexdigest()
            except (FileNotFoundError, _Missing):
                actual = None
            except (SnapshotError, OSError) as exc:
                actual = "<error: {}>".format(_scrub(exc))
            checked += 1
            if actual is None or actual != entry.get("sha256") or (
                    "bytes" in entry and size != entry["bytes"]):
                failures.append((path, entry.get("sha256"), actual))
    return VerifyResult(checked, total, failures)


class _Missing(Exception):
    pass


def _file_chunks(base, snapshot_id, path, token, chunk=1 << 20):
    rel = posixpath.normpath(path)
    if rel.startswith(("/", "..")) or rel == ".":
        raise SnapshotError(
            "manifest path {!r} is outside snapshot {}".format(path, snapshot_id))
    if is_local(base):
        with open(os.path.join(base, snapshot_id, path), "rb") as fh:
            while True:
                piece = fh.read(chunk)
                if not piece:
                    return
                yield piece
    else:
        try:
            yield from _stream("{}/{}/{}".format(base, snapshot_id, path),
                               token=token, chunk=chunk)
        except _HttpStatus as exc:
            if exc.status == 404:
                raise _Missing(path) from exc
            raise


# -- by tokamak and name ---------------------------------------------------------

def locator_for(tokamak, name):
    """The `sql_snapshot` locator called `name` on `tokamak`, from the
    catalogs registered via the fdp_schema.catalogs entry-point group."""
    from .mssql import _discover_catalogs
    catalogs = _discover_catalogs()
    if tokamak not in catalogs:
        raise KeyError("No tokamak named {!r}. Available: {}".format(
            tokamak, sorted(catalogs)))
    locs = [l for l in catalogs[tokamak].locators
            if l.kind == "sql_snapshot" and l.name == name]
    if not locs:
        avail = sorted(l.name for l in catalogs[tokamak].locators
                       if l.kind == "sql_snapshot")
        raise KeyError("No sql_snapshot locator named {!r} on tokamak {!r}. "
                       "Available: {}".format(name, tokamak, avail))
    return locs[0]


def connect_tokamak(tokamak, name, snapshot=None):
    """`connect` for the locator `locator_for(tokamak, name)` finds."""
    from ._snapshot_db import connect as _connect   # lazy: import cycle
    return _connect(locator_for(tokamak, name), snapshot=snapshot)


# -- the connection, re-exported lazily ------------------------------------
# _snapshot_db imports from this module at import time, so this module must
# not import it eagerly (PEP 562 module __getattr__ instead).

_DB_EXPORTS = ("connect", "SnapshotConnection", "SnapshotCursor",
               "_import_duckdb", "_open_duckdb", "_noticed")


def __dir__():
    return sorted(set(globals()) | set(_DB_EXPORTS))


def __getattr__(name):
    if name in _DB_EXPORTS:
        from . import _snapshot_db
        return getattr(_snapshot_db, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
