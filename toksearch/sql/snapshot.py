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
    if u.scheme == "http" and u.hostname in ("127.0.0.1", "localhost"):
        return base_url.rstrip("/")      # the test server only; never a real origin
    if u.scheme == "pelican":
        return _director_endpoint(u.netloc) + u.path.rstrip("/")
    raise SnapshotError(
        "sql_snapshot base_url must be pelican://, https:// or file://, "
        "not {!r}".format(base_url))


def is_local(base):
    return not base.startswith(("https://", "http://"))


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


# -- the connection ------------------------------------------------------------

_noticed = set()


def _import_duckdb():
    try:
        import duckdb
        import sqlglot  # noqa: F401
    except ImportError as exc:
        raise SnapshotError(
            "reading a SQL snapshot needs DuckDB and sqlglot ({}). Install "
            "them: `conda install {}`. A device package that ships a "
            "sql_snapshot locator, such as toksearch_d3d, declares them.".format(
                exc, CONDA_PACKAGES)) from exc
    return duckdb


def _token_for(locator, base):
    """The bearer token the locator's AuthHint names, or None for a local
    base. Missing for a remote base is an error before any HTTP."""
    if is_local(base):
        return None
    auth = locator.auth
    if auth is None or auth.kind != "bearer_token" or not auth.env:
        raise SnapshotError(
            "sql_snapshot locator {!r} names no bearer_token env var; nothing "
            "to authenticate with".format(locator.name))
    token = os.environ.get(auth.env, "")
    if not token:
        raise SnapshotError(
            "{} is not set; run under `fdp run`, or `fdp login`".format(auth.env))
    return token


def _open_duckdb(duckdb, token):
    """An in-memory DuckDB with httpfs loaded from the conda prefix.

    Extensions are never fetched from DuckDB's own server: an FDP
    environment gets its binaries from conda-forge. That package installs
    the extension under <prefix>/duckdb/extensions and does not sign it,
    so both the directory and allow_unsigned_extensions are needed. The
    trust is conda-forge's build, the same trust every other package in
    the environment has.
    """
    ext_dir = os.path.join(sys.prefix, "duckdb", "extensions")
    config = {
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        "allow_unsigned_extensions": True,
    }
    if os.path.isdir(ext_dir):
        config["extension_directory"] = ext_dir
    con = duckdb.connect(config=config)
    try:
        con.execute("LOAD httpfs")
    except duckdb.Error as exc:
        raise SnapshotError(
            "DuckDB's httpfs extension is not installed ({}). Install "
            "duckdb-extension-httpfs from conda-forge; it is not fetched "
            "from the network at first use.".format(
                str(exc).splitlines()[0])) from exc
    if token:
        con.execute(_create_secret(token))
    con.execute("SET enable_object_cache = true")
    return con


def _create_secret(token):
    # CREATE SECRET does not take bound parameters (DuckDB 1.5), so the
    # token is quoted as a SQL literal.
    return "CREATE SECRET fdp_snapshot (TYPE http, BEARER_TOKEN '{}')".format(
        token.replace("'", "''"))


def _define_views(con, base, sid, manifest):
    con.execute("CREATE SCHEMA dbo")
    for table in manifest["tables"]:
        if is_local(base):
            urls = [os.path.join(base, sid, f["path"]) for f in table["files"]]
        else:
            urls = ["{}/{}/{}".format(base, sid, f["path"]) for f in table["files"]]
        literal = "[" + ", ".join("'" + u.replace("'", "''") + "'" for u in urls) + "]"
        con.execute('CREATE VIEW dbo."{}" AS SELECT * FROM read_parquet({})'.format(
            table["name"].replace('"', '""'), literal))
    con.execute("SET search_path = 'dbo'")
    collation = COLLATIONS[manifest.get("transforms", {}).get("collation", "binary")]
    if collation:
        con.execute("PRAGMA default_collation = '{}'".format(collation))


def _notice_once(name, sid, base_url):
    if name in _noticed:
        return
    _noticed.add(name)
    warnings.warn(
        "{}: reading snapshot {} from {}. For the live database pass "
        "live=True. Silence this with warnings.filterwarnings('ignore', "
        "category=SnapshotNotice).".format(name, sid, base_url),
        SnapshotNotice, stacklevel=4)


def connect(locator, snapshot=None):
    """A DB-API-shaped connection to one snapshot of `locator`'s database.

    Resolves the snapshot (`resolve`), fetches and validates its manifest,
    opens DuckDB over the manifest's Parquet files and returns a
    SnapshotConnection. Bytes are never checked against the manifest's
    hashes here -- a range read cannot hash a file; `fdp snapshot verify`
    does that.
    """
    duckdb = _import_duckdb()
    base = resolve_base(locator.base_url)
    token = _token_for(locator, base)
    pinned_before = locator.name in _pinned
    sid = resolve(locator, snapshot=snapshot, token=token)
    try:
        manifest = fetch_manifest(base, sid, token)
    except SnapshotError:
        # A pin this call made for a snapshot that turned out not to exist
        # must not survive it; one the user exported is theirs to keep.
        if not pinned_before and locator.name in _pinned:
            _pinned.discard(locator.name)
            os.environ.pop(env_var(locator.name), None)
        raise
    con = _open_duckdb(duckdb, token)
    _define_views(con, base, sid, manifest)
    _notice_once(locator.name, sid, locator.base_url)
    return SnapshotConnection(con, sid, manifest, locator, base)


class SnapshotConnection:
    """What `connect` returns. The shape pd.read_sql and `with` expect."""

    def __init__(self, con, sid, manifest, locator, base):
        self._con = con
        self.snapshot = sid
        self.manifest = manifest
        self._locator = locator
        self._base = base
        self._nocase = manifest.get("transforms", {}).get("collation") == "nocase"
        self._excluded = {e["table"].lower(): e.get("reason", "")
                          for e in manifest.get("excluded", [])}
        auth = locator.auth
        self._token_seen = os.environ.get(auth.env, "") if auth and auth.env else None

    @property
    def duckdb(self):
        """The native DuckDB connection, for callers who want it."""
        return self._con

    def cursor(self):
        # A DuckDB cursor is a duplicate connection: it shares the database
        # (views, secrets, global settings such as default_collation) but
        # not session settings, and search_path is one of those.
        cur = self._con.cursor()
        cur.execute("SET search_path = 'dbo'")
        return SnapshotCursor(self, cur)

    def close(self):
        self._con.close()

    def commit(self):
        pass

    def rollback(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- used by the cursor --

    def _refresh_token(self):
        """Re-read the token once; True if it changed and the secret was replaced."""
        auth = self._locator.auth
        if is_local(self._base) or auth is None or not auth.env:
            return False
        new = os.environ.get(auth.env, "")
        if not new or new == self._token_seen:
            return False
        self._token_seen = new
        self._con.execute("DROP SECRET IF EXISTS fdp_snapshot")
        self._con.execute(_create_secret(new))
        return True

    def _explain(self, exc, note):
        """A DuckDB error as a SnapshotError a user can act on."""
        msg = _scrub(str(exc)).strip()
        m = re.search(r"Table with name (\w+) does not exist", msg)
        if m and m.group(1).lower() in self._excluded:
            msg += ("\n{} is excluded from snapshots of this database "
                    "(reason recorded in the manifest: {!r}).".format(
                        m.group(1).upper(), self._excluded[m.group(1).lower()]))
        if note:
            msg += "\n" + note
        return SnapshotError("[snapshot {}] {}".format(self.snapshot, msg))


class SnapshotCursor:
    def __init__(self, conn, cur):
        self._conn = conn
        self._cur = cur

    def execute(self, sql, params=None):
        from . import _tsql
        import duckdb
        rewritten, style, note = _tsql.rewrite(sql, nocase=self._conn._nocase)
        if style == "named" and not isinstance(params, dict):
            raise SnapshotError("%(name)s placeholders need a mapping of parameters")
        if style == "qmark" and params is not None and isinstance(params, dict):
            raise SnapshotError("%s placeholders need a sequence of parameters")
        for attempt in (1, 2):
            try:
                if params is None:
                    self._cur.execute(rewritten)
                else:
                    self._cur.execute(rewritten, params)
                return self
            except duckdb.Error as exc:
                text = str(exc)
                if attempt == 1 and ("401" in text or "403" in text) and self._conn._refresh_token():
                    continue
                if "401" in text or "403" in text:
                    raise SnapshotError(
                        "[snapshot {}] not authorized (HTTP 401/403) and the bearer "
                        "token has not changed; run `fdp login`. {}".format(
                            self._conn.snapshot, _scrub(text).splitlines()[0])) from exc
                raise self._conn._explain(exc, note) from exc

    def fetchone(self):
        return self._cur.fetchone()

    def fetchmany(self, size=1):
        return self._cur.fetchmany(size)

    def fetchall(self):
        return self._cur.fetchall()

    @property
    def description(self):
        return self._cur.description

    @property
    def rowcount(self):
        return self._cur.rowcount

    def __iter__(self):
        row = self._cur.fetchone()
        while row is not None:
            yield row
            row = self._cur.fetchone()

    def close(self):
        self._cur.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
