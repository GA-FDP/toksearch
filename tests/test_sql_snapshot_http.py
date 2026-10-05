"""The connection against an origin-like server: auth, ranges, listing.

Every assertion here has a control that must be seen to fail; a control
that passes means the test is checking nothing.
"""

import contextlib
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
import warnings
import xml.etree.ElementTree as ET
from unittest import mock

try:
    import duckdb  # noqa: F401
    import sqlglot  # noqa: F401
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False

from fdp_schema import SqlSnapshotLocator, AuthHint

import sql_fixture
from toksearch.sql import snapshot

TOKEN = "fixture-token-0123456789"


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@unittest.skipUnless(HAVE_DUCKDB, "duckdb/sqlglot not installed (conda-forge: python-duckdb duckdb-extension-httpfs sqlglot)")
class TestOverHttp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        sql_fixture.build(cls.tmp.name, "d3drdb_20260901T000000Z")
        sql_fixture.build(cls.tmp.name, "d3drdb_20261005T120000Z")
        os.makedirs(os.path.join(cls.tmp.name, "_spike"))
        cls.server = sql_fixture.Server(cls.tmp.name, TOKEN).__enter__()
        cls.loc = SqlSnapshotLocator(
            name="httpdb", base_url=cls.server.url, id_pattern="d3drdb_*",
            auth=AuthHint(kind="bearer_token", env="FIXTURE_BEARER"))

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)
        cls.tmp.cleanup()

    def setUp(self):
        snapshot._pinned.clear()
        snapshot._noticed.clear()
        self.server.reset()
        self._env = env(FDP_SQL_SNAPSHOT_HTTPDB=None, FDP_STORE_ROOT=None,
                        FDP_STORE_CATALOG=None, FIXTURE_BEARER=TOKEN)
        self._env.__enter__()
        warnings.simplefilter("ignore", snapshot.SnapshotNotice)

    def tearDown(self):
        self.server.token = TOKEN    # a test that rotated it and failed must not fail the rest
        self._env.__exit__(None, None, None)
        warnings.resetwarnings()

    def test_listing_picks_the_newest_and_ignores_spike(self):
        with snapshot.connect(self.loc) as conn:
            self.assertEqual(conn.snapshot, "d3drdb_20261005T120000Z")
        self.assertIn("PROPFIND", self.server.stats["methods"])

    def test_a_query_reads_a_slice_by_range_request(self):
        with snapshot.connect(self.loc) as conn:
            self.server.reset()
            cur = conn.cursor()
            cur.execute("SELECT shot FROM shots WHERE shot = 3")
            self.assertEqual(cur.fetchone(), (3,))
        total = os.path.getsize(os.path.join(self.tmp.name, "d3drdb_20261005T120000Z", "SHOTS.parquet"))
        self.assertGreater(self.server.stats["requests"], 0)
        self.assertLess(self.server.stats["bytes"], total,
                        "a one-row query transferred the whole file")

    def test_control_no_token_fails_before_any_http(self):
        with env(FIXTURE_BEARER=None):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.connect(self.loc)
        self.assertIn("FIXTURE_BEARER", str(cm.exception))
        self.assertEqual(self.server.stats["requests"], 0)

    def test_control_wrong_token_is_refused(self):
        with env(FIXTURE_BEARER="wrong"):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.connect(self.loc)
        self.assertIn("403", str(cm.exception))

    def test_a_token_rotated_mid_session_is_picked_up_once(self):
        with snapshot.connect(self.loc) as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM shots")
            self.assertEqual(cur.fetchone()[0], 3)
            # The server now expects a new token; the environment has it.
            self.server.token = "rotated-token"
            with env(FIXTURE_BEARER="rotated-token"):
                cur.execute("SELECT count(*) FROM runs")
                self.assertEqual(cur.fetchone()[0], 2)
        self.server.token = TOKEN

    def test_control_a_stale_token_names_fdp_login(self):
        with snapshot.connect(self.loc) as conn:
            cur = conn.cursor()
            self.server.token = "rotated-token"       # environment NOT updated
            try:
                with self.assertRaises(snapshot.SnapshotError) as cm:
                    cur.execute("SELECT count(*) FROM runs")
                self.assertIn("fdp login", str(cm.exception))
            finally:
                self.server.token = TOKEN

    def test_unknown_id_is_an_error_not_a_substitution(self):
        with self.assertRaises(snapshot.SnapshotError) as cm:
            snapshot.connect(self.loc, snapshot="d3drdb_19990101T000000Z")
        self.assertIn("d3drdb_19990101T000000Z", str(cm.exception))
        self.assertNotIn("FDP_SQL_SNAPSHOT_HTTPDB", os.environ)

    def test_connecting_fetches_no_parquet_and_a_query_only_its_tables(self):
        # d3drdb has ~60 tables: binding them all at connect time would be
        # ~120 requests before the first query. Views are made on first use.
        with snapshot.connect(self.loc) as conn:
            self.assertEqual(
                [p for p in self.server.stats["paths"] if p.endswith(".parquet")], [],
                "connect() read Parquet before any query")
            self.server.reset()
            self.assertEqual(self.server.stats["requests"], 0)
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM shots")
            self.assertEqual(cur.fetchone()[0], 3)
        sid = "d3drdb_20261005T120000Z"
        paths = self.server.stats["paths"]
        self.assertIn("/{}/SHOTS.parquet".format(sid), paths)
        self.assertNotIn("/{}/SHOTS_TYPE.parquet".format(sid), paths)
        self.assertNotIn("/{}/RUNS.parquet".format(sid), paths)

    def test_the_token_is_redacted_from_query_errors(self):
        with snapshot.connect(self.loc) as conn:
            self.assertIn(TOKEN, str(duckdb.Error("x " + TOKEN)))   # control
            err = conn._explain(duckdb.Error("boom, sent " + TOKEN), None)
            self.assertNotIn(TOKEN, str(err))
            self.assertIn("<redacted>", str(err))
            # the 401/403 path, with the token unchanged so there is no retry
            cur = conn.cursor()
            failing = mock.Mock()
            failing.execute.side_effect = duckdb.HTTPException(
                "HTTP 403 Forbidden for Bearer " + TOKEN)
            cur._cur = failing
            with self.assertRaises(snapshot.SnapshotError) as cm:
                cur.execute("SELECT 1")
            self.assertIn("fdp login", str(cm.exception))
            self.assertNotIn(TOKEN, str(cm.exception))

    def test_concurrent_403s_after_one_rotation_both_succeed(self):
        # Both cursors get a 403 before either re-reads the token (the
        # barrier holds the first refresher until the second has failed
        # too). The second must retry with the token the first installed,
        # not report "token has not changed".
        with snapshot.connect(self.loc) as conn:
            curs = [conn.cursor(), conn.cursor()]
            curs[0].execute("SELECT count(*) FROM shots JOIN runs ON shots.run = runs.run")
            curs[0].fetchall()
            real = conn._refresh_token
            barrier = threading.Barrier(2, timeout=10)

            def refresh(*a):
                barrier.wait()
                return real(*a)

            conn._refresh_token = refresh
            self.server.token = "rotated-token"
            results, errors = {}, {}

            def run(i, table):
                try:
                    curs[i].execute("SELECT count(*) FROM " + table)
                    results[i] = curs[i].fetchone()[0]
                except Exception as exc:          # reported below
                    errors[i] = exc

            with env(FIXTURE_BEARER="rotated-token"):
                self.server.reset()
                threads = [threading.Thread(target=run, args=(0, "shots")),
                           threading.Thread(target=run, args=(1, "runs"))]
                for t in threads:
                    t.start()
                for t in threads:
                    t.join(30)
            self.assertEqual(errors, {})
            self.assertEqual(results, {0: 3, 1: 2})
            self.assertEqual(conn._token_seen, "rotated-token")

    def test_a_locator_without_auth_reads_a_public_server(self):
        with sql_fixture.Server(self.tmp.name, TOKEN, require_auth=False) as pub:
            loc = SqlSnapshotLocator(name="pubdb", base_url=pub.url, id_pattern="d3drdb_*",
                                     auth=AuthHint(kind="none", env="FIXTURE_BEARER"))
            try:
                with env(FIXTURE_BEARER="must-not-be-used"):
                    with snapshot.connect(loc) as conn:
                        self.assertIsNone(conn._token_seen)
                        self.assertEqual(conn.duckdb.execute(
                            "SELECT count(*) FROM duckdb_secrets()").fetchone()[0], 0)
                        cur = conn.cursor()
                        cur.execute("SELECT count(*) FROM shots")
                        self.assertEqual(cur.fetchone()[0], 3)
            finally:
                os.environ.pop("FDP_SQL_SNAPSHOT_PUBDB", None)

    def test_a_view_bound_on_duckdbs_error_gets_the_same_auth_handling(self):
        # sqlglot is made unable to parse, so nothing is bound up front and
        # RUNS is bound from DuckDB's table-not-found error. That bind reads
        # the footer, which the server now refuses; the refusal must come
        # out as the usual SnapshotError naming fdp login, not raw DuckDB.
        import sqlglot
        real = sqlglot.parse

        def parse(sql, read=None, **kw):
            if read == "duckdb":
                raise sqlglot.errors.ParseError("forced")
            return real(sql, read=read, **kw)

        with snapshot.connect(self.loc) as conn:
            cur = conn.cursor()
            self.server.token = "rotated-token"       # environment NOT updated
            with mock.patch.object(sqlglot, "parse", parse):
                try:
                    cur.execute("SELECT count(*) FROM runs")
                except Exception as exc:              # examined below
                    err = exc
                else:
                    self.fail("the query succeeded against a rotated token")
            self.assertIsInstance(err, snapshot.SnapshotError)
            self.assertNotIsInstance(err, duckdb.Error)
            self.assertIn("fdp login", str(err))
            self.assertNotIn("runs", conn._views)

    def test_environment_pin_is_honoured_over_http(self):
        with env(FDP_SQL_SNAPSHOT_HTTPDB="d3drdb_20260901T000000Z"):
            with snapshot.connect(self.loc) as conn:
                self.assertEqual(conn.snapshot, "d3drdb_20260901T000000Z")
        self.assertNotIn("PROPFIND", self.server.stats["methods"])


@unittest.skipUnless(HAVE_DUCKDB, "duckdb/sqlglot not installed (conda-forge: python-duckdb duckdb-extension-httpfs sqlglot)")
class TestFixtureServer(unittest.TestCase):
    """The fixture behaves like the origin where the client could tell."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        sql_fixture.build(cls.tmp.name, "d3drdb_20261005T120000Z")
        cls.path = "/d3drdb_20261005T120000Z/RUNS.parquet"
        with open(os.path.join(cls.tmp.name, cls.path.lstrip("/")), "rb") as fh:
            cls.data = fh.read()
        cls.server = sql_fixture.Server(cls.tmp.name, TOKEN).__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)
        cls.tmp.cleanup()

    def req(self, path, **headers):
        headers.setdefault("Authorization", "Bearer " + TOKEN)
        method = headers.pop("method", "GET")
        r = urllib.request.Request(self.server.url + path, method=method, headers=headers)
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, dict(resp.headers), resp.read()
        except urllib.error.HTTPError as exc:
            return exc.code, dict(exc.headers), exc.read()

    def test_suffix_range(self):
        status, hdrs, body = self.req(self.path, Range="bytes=-8")
        self.assertEqual(status, 206)
        self.assertEqual(body, self.data[-8:])
        n = len(self.data)
        self.assertEqual(hdrs["Content-Range"], "bytes {}-{}/{}".format(n - 8, n - 1, n))

    def test_range_past_eof_is_416(self):
        status, hdrs, _ = self.req(self.path, Range="bytes={}-".format(len(self.data)))
        self.assertEqual(status, 416)
        self.assertEqual(hdrs["Content-Range"], "bytes */{}".format(len(self.data)))

    def test_propfind_on_a_file_lists_just_it(self):
        status, _, body = self.req(self.path, method="PROPFIND", Depth="1")
        self.assertEqual(status, 207)
        hrefs = [h.text for h in ET.fromstring(body).iter("{DAV:}href")]
        self.assertEqual(hrefs, [self.path])

    def test_the_token_in_a_url_is_not_accepted(self):
        status, _, _ = self.req(self.path + "?authz=" + TOKEN, Authorization="")
        self.assertEqual(status, 403)


if __name__ == "__main__":
    unittest.main()
