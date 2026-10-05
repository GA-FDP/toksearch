"""The connection, over a snapshot on local disk (file://): no network.

What callers hold today is a pymssql connection they hand to pd.read_sql
or use with `with`; this is the same shape. T-SQL goes in, DuckDB runs it.
"""

import contextlib
import os
import tempfile
import unittest
import warnings
from unittest import mock

try:
    import duckdb  # noqa: F401
    import sqlglot  # noqa: F401
    HAVE_DUCKDB = True
except ImportError:
    HAVE_DUCKDB = False

from fdp_schema import SqlSnapshotLocator

import sql_fixture
from toksearch.sql import snapshot


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
class TestConnection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        sql_fixture.build(cls.tmp.name, "d3drdb_20260901T000000Z")
        sql_fixture.build(cls.tmp.name, "d3drdb_20261005T120000Z")
        os.makedirs(os.path.join(cls.tmp.name, "_spike"))
        cls.loc = SqlSnapshotLocator(name="d3drdb", base_url="file://" + cls.tmp.name,
                                     id_pattern="d3drdb_*")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        snapshot._pinned.clear()
        snapshot._noticed.clear()
        self._env = env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_STORE_ROOT=None, FDP_STORE_CATALOG=None)
        self._env.__enter__()

    def tearDown(self):
        self._env.__exit__(None, None, None)

    def connect(self, **kw):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", snapshot.SnapshotNotice)
            return snapshot.connect(self.loc, **kw)

    def test_newest_by_default_and_the_id_is_exposed(self):
        with self.connect() as conn:
            self.assertEqual(conn.snapshot, "d3drdb_20261005T120000Z")
            self.assertEqual(conn.manifest["id"], "d3drdb_20261005T120000Z")
            self.assertEqual(os.environ["FDP_SQL_SNAPSHOT_D3DRDB"], "d3drdb_20261005T120000Z")

    def test_tsql_runs_unchanged(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT TOP 2 s.shot FROM shots s JOIN shots_type t ON s.shot = t.shot "
                        "WHERE t.shot_type = 'plasma' ORDER BY s.shot DESC")
            self.assertEqual([r[0] for r in cur.fetchall()], [2, 1])

    def test_dbo_prefix_and_brackets(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT [SHOT] FROM dbo.SHOTS WHERE [RUN] = 'run1'")
            self.assertEqual(len(cur.fetchall()), 2)

    def test_pymssql_placeholders(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT shot FROM shots WHERE shot = %s", (3,))
            self.assertEqual(cur.fetchone(), (3,))
            cur.execute("SELECT shot FROM shots WHERE run = %(run)s ORDER BY shot", {"run": "run1"})
            self.assertEqual([r[0] for r in cur.fetchall()], [1, 2])

    def test_case_insensitive_like_sql_server(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM shots_type WHERE shot_type = 'PLASMA'")
            self.assertEqual(cur.fetchone()[0], 2)
            cur.execute("SELECT count(*) FROM shots WHERE brief LIKE 'elm%'")
            self.assertEqual(cur.fetchone()[0], 2)
            # a join on a text key, cases differing between the tables
            cur.execute("SELECT count(*) FROM shots s JOIN runs r ON s.run = r.run WHERE s.shot = 3")
            self.assertEqual(cur.fetchone()[0], 1)

    def test_control_without_nocase_the_same_query_returns_zero(self):
        # Proves the collation pragma is what makes the previous test pass.
        with tempfile.TemporaryDirectory() as d:
            sql_fixture.build(d, "d3drdb_20260101T000000Z", collation="binary")
            loc = SqlSnapshotLocator(name="binary_case", base_url="file://" + d, id_pattern="d3drdb_*")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", snapshot.SnapshotNotice)
                with snapshot.connect(loc) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT count(*) FROM shots_type WHERE shot_type = 'PLASMA'")
                    self.assertEqual(cur.fetchone()[0], 0)
        os.environ.pop("FDP_SQL_SNAPSHOT_BINARY_CASE", None)

    def test_description_rowcount_and_iteration(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT shot, run FROM shots ORDER BY shot")
            self.assertEqual([d[0] for d in cur.description], ["SHOT", "RUN"])
            rows = list(cur)
            self.assertEqual(len(rows), 3)

    def test_pandas_read_sql(self):
        import pandas as pd
        with self.connect() as conn:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # pandas' DBAPI2 warning
                df = pd.read_sql("SELECT TOP 10 shot, entered FROM shots ORDER BY shot", conn)
        self.assertEqual(list(df["SHOT"]), [1, 2, 3])

    def test_explicit_snapshot(self):
        with self.connect(snapshot="d3drdb_20260901T000000Z") as conn:
            self.assertEqual(conn.snapshot, "d3drdb_20260901T000000Z")

    def test_missing_snapshot_is_an_error_not_the_newest(self):
        with self.assertRaises(snapshot.SnapshotError) as cm:
            self.connect(snapshot="d3drdb_NOPE")
        self.assertIn("d3drdb_NOPE", str(cm.exception))
        # and the failed pin did not stick
        self.assertNotIn("FDP_SQL_SNAPSHOT_D3DRDB", os.environ)

    def test_excluded_table_is_explained(self):
        with self.connect() as conn:
            cur = conn.cursor()
            with self.assertRaises(snapshot.SnapshotError) as cm:
                cur.execute("SELECT * FROM personnel")
            msg = str(cm.exception)
            self.assertIn("PERSONNEL", msg)
            self.assertIn("excluded", msg)
            self.assertIn("people", msg)

    def test_a_parse_failure_reports_both_opinions(self):
        with self.connect() as conn:
            cur = conn.cursor()
            with self.assertRaises(snapshot.SnapshotError) as cm:
                cur.execute("SELECT FROM WHERE (((")
            self.assertIn("sqlglot", str(cm.exception))
            self.assertIn("d3drdb_20261005T120000Z", str(cm.exception))

    def test_the_notice_fires_once_per_process(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            snapshot.connect(self.loc).close()
            snapshot.connect(self.loc).close()
        notices = [w for w in caught if issubclass(w.category, snapshot.SnapshotNotice)]
        self.assertEqual(len(notices), 1)
        self.assertIn("d3drdb_20261005T120000Z", str(notices[0].message))
        self.assertIn("live=True", str(notices[0].message))

    def test_httpfs_missing_is_an_error_not_a_download(self):
        # An environment whose extension directory lacks httpfs: autoinstall
        # is off, so LOAD fails here instead of fetching from DuckDB's server.
        with tempfile.TemporaryDirectory() as prefix:
            os.makedirs(os.path.join(prefix, "duckdb", "extensions"))
            with mock.patch.object(snapshot.sys, "prefix", prefix):
                with self.assertRaises(snapshot.SnapshotError) as cm:
                    snapshot._open_duckdb(duckdb, None)
        self.assertIn("duckdb-extension-httpfs", str(cm.exception))

    def test_a_query_naming_two_tables_binds_both(self):
        with self.connect() as conn:
            self.assertEqual(conn._views, set())
            cur = conn.cursor()
            cur.execute("SELECT count(*) FROM dbo.shots s JOIN [SHOTS_TYPE] t ON s.shot = t.shot")
            self.assertEqual(cur.fetchone()[0], 3)
            self.assertEqual(conn._views, {"shots", "shots_type"})

    def test_a_second_cursor_sees_views_the_first_created(self):
        with self.connect() as conn:
            first, second = conn.cursor(), conn.cursor()
            first.execute("SELECT count(*) FROM runs")
            self.assertEqual(first.fetchone()[0], 2)
            # bypass the binding step: the view must already be visible
            self.assertEqual(second._cur.execute("SELECT count(*) FROM runs").fetchone()[0], 2)

    def test_a_pass_through_statement_still_binds_its_table(self):
        # T-SQL cannot parse a list comprehension, so it passes through
        # untranslated; the DuckDB dialect can, and finds `shots`.
        sql = "SELECT [x FOR x IN [shot]] FROM shots ORDER BY shot"
        import sqlglot
        with self.assertRaises(sqlglot.errors.SqlglotError):
            sqlglot.parse(sql, read="tsql")                # control
        sqlglot.parse(sql, read="duckdb")                  # and this one can
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute(sql)
            self.assertEqual([r[0] for r in cur.fetchall()], [[1], [2], [3]])
            self.assertEqual(conn._views, {"shots"})

    def test_sql_no_parser_can_read_binds_every_table(self):
        # No natural statement found that both sqlglot dialects reject and
        # DuckDB accepts, so the DuckDB-dialect parse is made to fail.
        import sqlglot
        real = sqlglot.parse

        def parse(sql, read=None, **kw):
            if read == "duckdb":
                raise sqlglot.errors.ParseError("forced")
            return real(sql, read=read, **kw)

        with self.connect() as conn:
            cur = conn.cursor()
            with mock.patch.object(sqlglot, "parse", parse):
                cur.execute("SELECT count(*) FROM runs")
            self.assertEqual(cur.fetchone()[0], 2)
            self.assertEqual(conn._views, {"shots", "shots_type", "runs"})

    def test_commit_and_rollback_are_harmless(self):
        with self.connect() as conn:
            conn.commit()
            conn.rollback()
            self.assertIsNotNone(conn.duckdb)


@unittest.skipUnless(HAVE_DUCKDB, "duckdb/sqlglot not installed (conda-forge: python-duckdb duckdb-extension-httpfs sqlglot)")
class TestSecretRedaction(unittest.TestCase):
    TOKEN = "s3cret-token-value-0123"

    def test_a_failed_create_secret_does_not_echo_the_token(self):
        real = duckdb.connect(config={"autoinstall_known_extensions": False})

        class Con:
            def execute(self, sql):
                if sql.startswith("CREATE SECRET"):
                    raise duckdb.ParserException("syntax error at or near: " + sql)
                if sql == "LOAD httpfs":
                    return None
                return real.execute(sql)

        fake = mock.Mock(Error=duckdb.Error)
        fake.connect.return_value = Con()
        with self.assertRaises(snapshot.SnapshotError) as cm:
            snapshot._open_duckdb(fake, self.TOKEN)
        self.assertNotIn(self.TOKEN, str(cm.exception))
        self.assertIn("<redacted>", str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)    # nor in the chained one
        self.assertTrue(cm.exception.__suppress_context__)


class TestMissingDependencies(unittest.TestCase):
    def test_import_error_names_the_packages(self):
        with mock.patch.dict("sys.modules", {"duckdb": None}):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot._import_duckdb()
        self.assertIn("python-duckdb", str(cm.exception))
        self.assertIn("toksearch_d3d", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
