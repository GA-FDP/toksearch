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
        self._env = env(FDP_SQL_SNAPSHOT_HTTPDB=None, FDP_SQL_SNAPSHOT_PUBDB=None,
                        FDP_SQL_SNAPSHOT_SCOPED=None, FDP_SQL_SNAPSHOT_PELDB=None,
                        FDP_STORE_ROOT=None,
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

    def test_no_token_for_a_pelican_base_fails_before_the_well_known_lookup(self):
        loc = SqlSnapshotLocator(
            name="peldb", base_url="pelican://fed.example/fdp-d3d/metadata/d3drdb",
            id_pattern="d3drdb_*", auth=AuthHint(kind="bearer_token", env="FIXTURE_BEARER"))
        snapshot._director_endpoint.cache_clear()
        with env(FIXTURE_BEARER=None), \
             mock.patch.object(snapshot, "_request") as req:
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.connect(loc)
        self.assertIn("FIXTURE_BEARER", str(cm.exception))
        req.assert_not_called()

    def test_the_token_goes_only_to_urls_under_the_base(self):
        # The base is <server>/inner. A query naming <server>/inner_outside.parquet
        # -- a string prefix of the base, but not under it -- must not get the
        # token; the in-scope read in the same session must (the control).
        with tempfile.TemporaryDirectory() as root:
            sql_fixture.build(os.path.join(root, "inner"), "d3drdb_20261005T120000Z")
            src = os.path.join(root, "inner", "d3drdb_20261005T120000Z", "RUNS.parquet")
            with open(src, "rb") as a, open(os.path.join(root, "inner_outside.parquet"), "wb") as b:
                b.write(a.read())
            with sql_fixture.Server(root, TOKEN, require_auth=False) as srv:
                loc = SqlSnapshotLocator(
                    name="scoped", base_url=srv.url + "/inner", id_pattern="d3drdb_*",
                    auth=AuthHint(kind="bearer_token", env="FIXTURE_BEARER"))
                with snapshot.connect(loc) as conn:
                    cur = conn.cursor()
                    srv.reset()
                    cur.execute("SELECT count(*) FROM runs")
                    self.assertEqual(cur.fetchone()[0], 2)
                    inside = list(zip(srv.stats["paths"], srv.stats["bearer"]))
                    srv.reset()
                    cur.execute("SELECT count(*) FROM '{}/inner_outside.parquet'".format(srv.url))
                    self.assertEqual(cur.fetchone()[0], 2)
                    outside = list(zip(srv.stats["paths"], srv.stats["bearer"]))
        self.assertTrue(inside and all(b for _, b in inside), inside)
        self.assertTrue(outside, "the out-of-scope file was never requested")
        self.assertFalse(any(b for _, b in outside), outside)

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
        self.assertIsNone(os.environ.get("FDP_SQL_SNAPSHOT_HTTPDB"))

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


@unittest.skipUnless(HAVE_DUCKDB, "duckdb/sqlglot not installed (the fixture is written with duckdb)")
class TestVerifyFiles(unittest.TestCase):
    """verify_files hashes every byte over the same HTTP path the client
    reads. The tampered snapshot is the control: an intact pass means
    nothing unless a changed byte is seen to fail."""

    INTACT = "d3drdb_20260901T000000Z"
    TAMPERED = "d3drdb_20261005T120000Z"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        sql_fixture.build(cls.tmp.name, cls.INTACT)
        sql_fixture.build(cls.tmp.name, cls.TAMPERED)
        with open(os.path.join(cls.tmp.name, cls.TAMPERED, "RUNS.parquet"), "ab") as fh:
            fh.write(b"\0")
        cls.server = sql_fixture.Server(cls.tmp.name, TOKEN).__enter__()
        cls.loc = SqlSnapshotLocator(
            name="verifydb", base_url=cls.server.url, id_pattern="d3drdb_*",
            auth=AuthHint(kind="bearer_token", env="FIXTURE_BEARER"))

    @classmethod
    def tearDownClass(cls):
        cls.server.__exit__(None, None, None)
        cls.tmp.cleanup()

    def setUp(self):
        self.server.reset()
        self._env = env(FIXTURE_BEARER=TOKEN)
        self._env.__enter__()
        self.addCleanup(self._env.__exit__, None, None, None)

    def test_an_intact_snapshot_has_no_failures(self):
        got = snapshot.verify_files(self.loc, self.INTACT)
        self.assertEqual(got.failures, [])
        self.assertEqual(got.checked, 3)
        self.assertEqual(got.checked, got.total)
        # Over HTTP, every file read in full, with the bearer token.
        for name in ("SHOTS", "SHOTS_TYPE", "RUNS"):
            self.assertIn("/{}/{}.parquet".format(self.INTACT, name),
                          self.server.stats["paths"])
        self.assertTrue(all(self.server.stats["bearer"]))

    def test_a_byte_appended_is_one_failure_naming_the_file(self):
        got = snapshot.verify_files(self.loc, self.TAMPERED)
        self.assertEqual(got.checked, got.total)
        self.assertEqual(len(got.failures), 1)
        path, expected, actual = got.failures[0]
        self.assertEqual(path, "RUNS.parquet")
        self.assertNotEqual(expected, actual)
        with open(os.path.join(self.tmp.name, self.TAMPERED, "RUNS.parquet"), "rb") as fh:
            import hashlib
            self.assertEqual(actual, hashlib.sha256(fh.read()).hexdigest())

    def test_a_sample_checks_fewer_than_all(self):
        got = snapshot.verify_files(self.loc, self.INTACT, sample=1)
        self.assertEqual(got.checked, 1)
        self.assertLess(got.checked, got.total)
        self.assertEqual(got.failures, [])

    def test_a_sample_is_deterministic(self):
        snapshot.verify_files(self.loc, self.INTACT, sample=1)
        first = [p for p in self.server.stats["paths"] if p.endswith(".parquet")]
        self.server.reset()
        snapshot.verify_files(self.loc, self.INTACT, sample=1)
        second = [p for p in self.server.stats["paths"] if p.endswith(".parquet")]
        self.assertEqual(len(first), 1)
        self.assertEqual(first, second)

    def test_a_sample_below_one_is_refused(self):
        for bad in (0, -1):
            with self.assertRaises(ValueError):
                snapshot.verify_files(self.loc, self.INTACT, sample=bad)

    def test_a_read_error_on_one_file_is_a_failure_and_the_rest_are_checked(self):
        real = snapshot._stream

        def flaky(url, **kw):
            if url.endswith("/SHOTS.parquet"):
                raise snapshot.SnapshotError("cannot read {} (HTTP 500)".format(url))
            return real(url, **kw)
        with mock.patch.object(snapshot, "_stream", side_effect=flaky):
            got = snapshot.verify_files(self.loc, self.TAMPERED)
        self.assertEqual(got.checked, 3)
        self.assertEqual(sorted(f[0] for f in got.failures),
                         ["RUNS.parquet", "SHOTS.parquet"])
        shots = [f for f in got.failures if f[0] == "SHOTS.parquet"][0]
        self.assertTrue(shots[2].startswith("<error: "), shots[2])
        self.assertIn("HTTP 500", shots[2])

    def test_verifying_keeps_the_long_timeout(self):
        seen = []
        real = snapshot._open

        def spy(url, **kw):
            seen.append(kw.get("timeout"))
            return real(url, **kw)
        with mock.patch.object(snapshot, "_open", side_effect=spy):
            snapshot.verify_files(self.loc, self.INTACT)
        self.assertTrue(seen)
        self.assertEqual(set(seen), {60})

    def test_no_token_fails_before_any_http(self):
        with env(FIXTURE_BEARER=None):
            with self.assertRaises(snapshot.SnapshotError):
                snapshot.verify_files(self.loc, self.INTACT)
        self.assertEqual(self.server.stats["requests"], 0)

    def test_a_missing_snapshot_is_an_error_not_a_pass(self):
        with self.assertRaises(snapshot.SnapshotError) as cm:
            snapshot.verify_files(self.loc, "d3drdb_20200101T000000Z")
        self.assertIn("d3drdb_20200101T000000Z", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
