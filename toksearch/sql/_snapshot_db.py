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
"""The connection half of `toksearch.sql.snapshot`: DuckDB over a resolved
snapshot's Parquet, and the DB-API-shaped objects callers hold.

`toksearch.sql.snapshot` re-exports `connect`, `SnapshotConnection` and
`SnapshotCursor` (and defines `SnapshotNotice`). This module imports from
`snapshot` at import time; `snapshot` reaches back here only lazily, through
its module `__getattr__`, so either module may be imported first.
"""

import os
import re
import sys
import threading
import urllib.parse
import warnings

from .snapshot import (
    COLLATIONS,
    CONDA_PACKAGES,
    SnapshotError,
    SnapshotNotice,
    _pinned,
    _scrub,
    env_var,
    fetch_manifest,
    is_local,
    resolve,
    resolve_base,
)


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


def _token_for(locator):
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


def _open_duckdb(duckdb, token, base=None):
    """An in-memory DuckDB with httpfs loaded from the conda prefix.

    Extensions are never fetched from DuckDB's own server: an FDP
    environment gets its binaries from conda-forge. That package installs
    the extension under <prefix>/duckdb/extensions and does not sign it,
    so both the directory and allow_unsigned_extensions are needed. The
    trust is conda-forge's build, the same trust every other package in
    the environment has. The directory is set even when it does not
    exist, so unsigned loading never reaches ~/.duckdb instead.

    `token`, when given, is installed as an http secret scoped to `base`.
    """
    ext_dir = os.path.join(sys.prefix, "duckdb", "extensions")
    config = {
        "autoinstall_known_extensions": False,
        "autoload_known_extensions": False,
        "allow_unsigned_extensions": True,
        "extension_directory": ext_dir,
    }
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
        _set_secret(duckdb, con, token, base)
    con.execute("SET enable_object_cache = true")
    return con


def _sql_literal(text):
    return "'" + text.replace("'", "''") + "'"


def _create_secret(token, base):
    # CREATE SECRET does not take bound parameters (DuckDB 1.5), so the
    # token is quoted as a SQL literal.
    # OR REPLACE: a rotation swaps the secret in one statement, so no other
    # cursor ever runs between a DROP and a CREATE with no secret at all.
    # SCOPE: the token goes only to URLs under the snapshot base, not to any
    # https URL a query names. SCOPE is a plain string prefix, hence the
    # trailing slash (".../d3drdb" would also cover ".../d3drdb_other").
    # Probed 2026-10-05 against the real director: a secret scoped to
    # https://osdf-director.osg-htc.org/fdp-d3d/metadata/d3drdb/ survives the
    # director's redirect to a cache (count(*) of _spike/SHOTS_TYPE.parquet =
    # 167631), and one scoped elsewhere gets 403 -- the scope is matched on
    # the requested URL, not the redirect target.
    return ("CREATE OR REPLACE SECRET fdp_snapshot (TYPE http, BEARER_TOKEN {}, "
            "SCOPE {})".format(_sql_literal(token), _sql_literal(base.rstrip("/") + "/")))


def _redact(text, token):
    """`_scrub`, and the raw token too when there is one."""
    text = _scrub(text)
    if token:
        text = text.replace(token, "<redacted>")
    return text


def _set_secret(duckdb, con, token, base):
    """Install the bearer token as DuckDB's http secret, scoped to `base`.
    A failure may echo the statement, token and all, so the error is
    redacted and raised without its cause."""
    try:
        con.execute(_create_secret(token, base))
    except duckdb.Error as exc:
        raise SnapshotError("cannot install the bearer token in DuckDB: {}".format(
            _redact(str(exc), token))) from None


def _view_sql(base, sid, table):
    """CREATE VIEW for one manifest table over its Parquet files."""
    if is_local(base):
        urls = [os.path.join(base, sid, f["path"]) for f in table["files"]]
    else:
        urls = ["{}/{}/{}".format(base, sid, f["path"]) for f in table["files"]]
    literal = "[" + ", ".join("'" + u.replace("'", "''") + "'" for u in urls) + "]"
    return 'CREATE VIEW dbo."{}" AS SELECT * FROM read_parquet({})'.format(
        table["name"].replace('"', '""'), literal)


def _define_views(con, base, sid, manifest):
    """The schema and settings only. Each table's view is created when a
    statement first names it (SnapshotConnection._ensure_tables): creating
    a view binds it, which reads the Parquet footer, so doing all of them
    here would cost two requests per table before the first query."""
    con.execute("CREATE SCHEMA dbo")
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
    token = _token_for(locator)          # before resolve_base: no HTTP without it
    base = resolve_base(locator.base_url)
    pinned_before = locator.name in _pinned
    exported_before = os.environ.get(env_var(locator.name))
    sid = resolve(locator, snapshot=snapshot, token=token)
    try:
        manifest = fetch_manifest(base, sid, token)
    except SnapshotError:
        # A pin this call made for a snapshot that turned out not to exist
        # must not survive it; a value already in the environment -- the
        # user's export, even one naming the same id -- is theirs to keep.
        if not pinned_before:
            _pinned.discard(locator.name)
        if exported_before is None:
            os.environ.pop(env_var(locator.name), None)
        raise
    con = _open_duckdb(duckdb, token, base)
    _define_views(con, base, sid, manifest)
    _notice_once(locator.name, sid, locator.base_url)
    return SnapshotConnection(con, sid, manifest, locator, base, token)


class SnapshotConnection:
    """What `connect` returns. The shape pd.read_sql and `with` expect."""

    def __init__(self, con, sid, manifest, locator, base, token=None):
        self._con = con
        self._sid = sid
        self.snapshot = sid
        self.manifest = manifest
        self._locator = locator
        self._base = base
        self._nocase = manifest.get("transforms", {}).get("collation") == "nocase"
        self._excluded = {e["table"].lower(): e.get("reason", "")
                          for e in manifest.get("excluded", [])}
        self._token_seen = token             # the token in DuckDB's secret, if any
        self._tables = {t["name"].lower(): t for t in manifest["tables"]}
        self._views = set()                  # lower-cased names created so far
        self._lock = threading.Lock()    # guards the views and the token

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

    def _refresh_token(self, started_with):
        """After a 401/403 on a statement begun with token `started_with`:
        True if a retry may succeed, because another cursor has already
        replaced the token since, or because the environment now holds a
        different one, which this call installs. False if nothing changed.
        """
        auth = self._locator.auth
        if (is_local(self._base) or auth is None or auth.kind != "bearer_token"
                or not auth.env):
            return False
        with self._lock:
            if self._token_seen != started_with:
                return True
            new = os.environ.get(auth.env, "")
            if not new or new == self._token_seen:
                return False
            import duckdb
            _set_secret(duckdb, self._con, new, self._base)
            self._token_seen = new
            return True

    def _missing_table(self, text):
        """The manifest table a DuckDB table-not-found error names, or None."""
        m = re.search(r"Table with name (\w+) does not exist", text)
        if not m or m.group(1).lower() not in self._tables:
            return None
        return m.group(1).lower()

    def _bind(self, name):
        """Create manifest table `name`'s view unless it exists. Its errors
        are the caller's to handle, like any statement's."""
        with self._lock:
            if name not in self._views:
                self._con.execute(_view_sql(self._base, self._sid, self._tables[name]))
                self._views.add(name)

    def _ensure_tables(self, sql):
        """Create the views `sql` names that do not exist yet.

        Table names come from parsing the already-rewritten statement in
        the DuckDB dialect; the schema part is ignored and case does not
        matter, and CTE names are not tables. Names that are not manifest
        tables are left alone, so DuckDB reports them (with the
        excluded-table hint). This is the fast path only: if sqlglot cannot
        parse the statement nothing is created here, and `execute` binds
        each table DuckDB reports missing (`_missing_table`, `_bind`) and retries.
        """
        import sqlglot
        from sqlglot import exp
        try:
            trees = [t for t in sqlglot.parse(sql, read="duckdb") if t is not None]
        except sqlglot.errors.SqlglotError:
            return
        names = set()
        for tree in trees:
            ctes = {c.alias.lower() for c in tree.find_all(exp.CTE)}
            names |= {t.name.lower() for t in tree.find_all(exp.Table)} - ctes
        with self._lock:
            for name in sorted(names & set(self._tables) - self._views):
                self._con.execute(_view_sql(self._base, self._sid, self._tables[name]))
                self._views.add(name)

    def _explain(self, exc, note):
        """A DuckDB error as a SnapshotError a user can act on."""
        msg = _redact(str(exc), self._token_seen).strip()
        m = re.search(r"Table with name (\w+) does not exist", msg)
        if m and m.group(1).lower() in self._excluded:
            msg += ("\n{} is excluded from snapshots of this database "
                    "(reason recorded in the manifest: {!r}).".format(
                        m.group(1).upper(), self._excluded[m.group(1).lower()]))
        if note:
            msg += "\n" + note
        return SnapshotError("[snapshot {}] DuckDB {}: {}".format(
            self.snapshot, type(exc).__name__, msg))


#: an HTTP auth refusal in a DuckDB error, as httpfs words it
_AUTH_FAILURE = re.compile(r"HTTP (401|403)\b")


class SnapshotCursor:
    def __init__(self, conn, cur):
        self._conn = conn
        self._cur = cur

    def execute(self, sql, params=None):
        from . import _tsql
        import duckdb
        rewritten, style, note = _tsql.rewrite(sql, nocase=self._conn._nocase)
        if style == "named" and not isinstance(params, dict):
            raise ValueError("%(name)s placeholders need a mapping of parameters")
        if style == "qmark" and params is not None and isinstance(params, dict):
            raise ValueError("%s placeholders need a sequence of parameters")
        conn = self._conn
        started_with = conn._token_seen
        reauthorized = False
        bound = 0
        pending_bind = None    # a table DuckDB reported missing, bound in the try
        while True:
            try:
                if pending_bind is not None:
                    conn._bind(pending_bind)
                    pending_bind = None
                conn._ensure_tables(rewritten)
                if params is None:
                    self._cur.execute(rewritten)
                else:
                    self._cur.execute(rewritten, params)
                return self
            except duckdb.Error as exc:
                text = str(exc)
                auth_failure = _AUTH_FAILURE.search(text)
                if auth_failure and not reauthorized and conn._refresh_token(started_with):
                    reauthorized = True        # one re-read of the token, one retry
                    continue
                missing = (None if auth_failure
                           or not isinstance(exc, duckdb.CatalogException)
                           else conn._missing_table(text))
                if missing is not None and bound < len(conn._tables):
                    pending_bind = missing
                    bound += 1
                    continue
                if auth_failure:
                    why = ("even with the token re-read from the environment"
                           if reauthorized else "and the bearer token has not changed")
                    raise SnapshotError(
                        "[snapshot {}] not authorized (HTTP 401/403) {}; run "
                        "`fdp login`. {}".format(
                            conn.snapshot, why,
                            _redact(text, conn._token_seen).splitlines()[0])) from None
                raise self._conn._explain(exc, note) from None

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
