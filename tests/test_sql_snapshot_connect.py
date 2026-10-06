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
        self._env = env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_SQL_SNAPSHOT_BINARY_CASE=None,
                        FDP_STORE_ROOT=None, FDP_STORE_CATALOG=None)
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
            # SQL Server names a column by the query's spelling, not the
            # stored one (the fixture's Parquet columns are upper case).
            self.assertEqual([d[0] for d in cur.description], ["shot", "run"])
            rows = list(cur)
            self.assertEqual(len(rows), 3)

    def test_pandas_read_sql(self):
        import pandas as pd
        with self.connect() as conn:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # pandas' DBAPI2 warning
                df = pd.read_sql("SELECT TOP 10 shot, entered FROM shots ORDER BY shot", conn)
        self.assertEqual(list(df["shot"]), [1, 2, 3])
        self.assertEqual(list(df.columns), ["shot", "entered"])

    def test_result_columns_take_the_querys_spelling(self):
        import pandas as pd
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT shot, run FROM shots")
            self.assertEqual([d[0] for d in cur.description], ["shot", "run"])
            cur.execute("SELECT s.Shot FROM shots s")
            self.assertEqual([d[0] for d in cur.description], ["Shot"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")   # pandas' DBAPI2 warning
                df = pd.read_sql("SELECT TOP 2 shot FROM shots ORDER BY shot", conn)
            self.assertEqual(list(df["shot"]), [1, 2])

    def test_star_keeps_the_stored_names(self):
        # SQL Server too: `*` has no written spelling to follow.
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM shots")
            names = [d[0] for d in cur.description]
            self.assertIn("SHOT", names)
            self.assertIn("RUN", names)

    def test_explicit_snapshot(self):
        with self.connect(snapshot="d3drdb_20260901T000000Z") as conn:
            self.assertEqual(conn.snapshot, "d3drdb_20260901T000000Z")

    def test_missing_snapshot_is_an_error_not_the_newest(self):
        with self.assertRaises(snapshot.SnapshotError) as cm:
            self.connect(snapshot="d3drdb_NOPE")
        self.assertIn("d3drdb_NOPE", str(cm.exception))
        # and the failed pin did not stick
        self.assertIsNone(os.environ.get("FDP_SQL_SNAPSHOT_D3DRDB"))

    def test_a_failed_snapshot_leaves_an_exported_pin_alone(self):
        # The user exported the id and the code names the same one; the
        # manifest is missing. The export is theirs: it stays.
        sid = "d3drdb_20250101T000000Z"
        with env(FDP_SQL_SNAPSHOT_D3DRDB=sid):
            with self.assertRaises(snapshot.SnapshotError):
                self.connect(snapshot=sid)
            self.assertEqual(os.environ.get("FDP_SQL_SNAPSHOT_D3DRDB"), sid)
        self.assertNotIn("d3drdb", snapshot._pinned)

    def test_excluded_table_is_explained(self):
        with self.connect() as conn:
            cur = conn.cursor()
            with self.assertRaises(snapshot.SnapshotError) as cm:
                cur.execute("SELECT * FROM personnel")
            msg = str(cm.exception)
            self.assertIn("PERSONNEL", msg)
            self.assertIn("excluded", msg)
            self.assertIn("people", msg)
            self.assertIn("DuckDB CatalogException", msg)

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
            with mock.patch("sys.prefix", prefix):
                with self.assertRaises(snapshot.SnapshotError) as cm:
                    snapshot._open_duckdb(duckdb, None)
        self.assertIn("duckdb-extension-httpfs", str(cm.exception))

    def test_the_extension_directory_is_set_even_when_absent(self):
        # Unset, DuckDB would use ~/.duckdb -- with unsigned loading allowed.
        fake = mock.Mock(Error=duckdb.Error)
        with tempfile.TemporaryDirectory() as prefix, mock.patch("sys.prefix", prefix):
            snapshot._open_duckdb(fake, None)
        config = fake.connect.call_args.kwargs["config"]
        self.assertEqual(config["extension_directory"],
                         os.path.join(prefix, "duckdb", "extensions"))
        self.assertFalse(os.path.isdir(config["extension_directory"]))   # control

    def test_a_placeholder_mismatch_is_a_value_error(self):
        with self.connect() as conn:
            cur = conn.cursor()
            with self.assertRaises(ValueError):
                cur.execute("SELECT shot FROM shots WHERE run = %(run)s", ("run1",))
            with self.assertRaises(ValueError):
                cur.execute("SELECT shot FROM shots WHERE shot = %s", {"shot": 3})

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

    def test_sql_no_parser_can_read_binds_only_what_it_names(self):
        # No natural statement found that both sqlglot dialects reject and
        # DuckDB accepts, so the DuckDB-dialect parse is made to fail. The
        # views then come from DuckDB's own table-not-found errors, one at
        # a time, and SHOTS_TYPE -- not named -- is never bound.
        import sqlglot
        real = sqlglot.parse

        def parse(sql, read=None, **kw):
            if read == "duckdb":
                raise sqlglot.errors.ParseError("forced")
            return real(sql, read=read, **kw)

        with self.connect() as conn:
            cur = conn.cursor()
            with mock.patch.object(sqlglot, "parse", parse):
                cur.execute("SELECT count(*) FROM runs r JOIN shots s ON s.run = r.run")
            self.assertEqual(cur.fetchone()[0], 3)
            self.assertEqual(conn._views, {"runs", "shots"})

    def test_a_cte_named_like_a_table_binds_nothing(self):
        with self.connect() as conn:
            cur = conn.cursor()
            cur.execute("WITH runs AS (SELECT 1 AS x) SELECT x FROM runs")
            self.assertEqual(cur.fetchall(), [(1,)])
            self.assertEqual(conn._views, set())

    def test_an_error_mentioning_403_is_not_an_auth_failure(self):
        with self.connect() as conn:
            cur = conn.cursor()
            failing = mock.Mock()
            failing.execute.side_effect = duckdb.InvalidInputException(
                "value 4031 is out of range; 403 rows")
            cur._cur = failing
            with self.assertRaises(snapshot.SnapshotError) as cm:
                cur.execute("SELECT 1")
            self.assertNotIn("fdp login", str(cm.exception))
            self.assertIn("DuckDB InvalidInputException", str(cm.exception))

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
                if "SECRET" in sql:
                    raise duckdb.ParserException("syntax error at or near: " + sql)
                if sql == "LOAD httpfs":
                    return None
                return real.execute(sql)

        fake = mock.Mock(Error=duckdb.Error)
        fake.connect.return_value = Con()
        with self.assertRaises(snapshot.SnapshotError) as cm:
            snapshot._open_duckdb(fake, self.TOKEN, "https://h/b")
        self.assertNotIn(self.TOKEN, str(cm.exception))
        self.assertIn("<redacted>", str(cm.exception))
        self.assertIsNone(cm.exception.__cause__)    # nor in the chained one
        self.assertTrue(cm.exception.__suppress_context__)


class TestPublicSurface(unittest.TestCase):
    def test_lazy_exports_are_visible(self):
        import pydoc
        self.assertIn("connect", dir(snapshot))
        self.assertIn("SnapshotConnection", dir(snapshot))
        self.assertIn("connect", snapshot.__all__)
        self.assertIn("connect(locator, snapshot=None)", pydoc.render_doc(snapshot, renderer=pydoc.plaintext))


class TestMissingDependencies(unittest.TestCase):
    def test_import_error_names_the_packages(self):
        with mock.patch.dict("sys.modules", {"duckdb": None}):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot._import_duckdb()
        self.assertIn("python-duckdb", str(cm.exception))
        self.assertIn("toksearch_d3d", str(cm.exception))


@unittest.skipUnless(HAVE_DUCKDB, "duckdb/sqlglot not installed (the fixture is written with duckdb)")
class TestVerifyLocalFiles(unittest.TestCase):
    """A file:// base hashes the local files; the appended byte is the control."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.sid = "d3drdb_20261005T120000Z"
        sql_fixture.build(self._tmp.name, self.sid)
        self.loc = SqlSnapshotLocator(name="d3drdb", base_url="file://" + self._tmp.name,
                                      id_pattern="d3drdb_*")

    def test_intact(self):
        got = snapshot.verify_files(self.loc, self.sid)
        self.assertEqual((got.checked, got.total, got.failures), (3, 3, []))

    def test_a_byte_appended_is_caught(self):
        with open(os.path.join(self._tmp.name, self.sid, "SHOTS.parquet"), "ab") as fh:
            fh.write(b"x")
        got = snapshot.verify_files(self.loc, self.sid)
        self.assertEqual([f[0] for f in got.failures], ["SHOTS.parquet"])

    def test_a_manifest_path_escaping_the_snapshot_is_refused(self):
        import json
        mpath = os.path.join(self._tmp.name, self.sid, "manifest.json")
        with open(mpath) as fh:
            doc = json.load(fh)
        doc["tables"][0]["files"][0]["path"] = "../elsewhere.parquet"
        with open(mpath, "w") as fh:
            json.dump(doc, fh)
        got = snapshot.verify_files(self.loc, self.sid)
        bad = [f for f in got.failures if f[0] == "../elsewhere.parquet"]
        self.assertEqual(len(bad), 1)
        self.assertTrue(bad[0][2].startswith("<error: "), bad[0][2])

    def test_a_missing_file_is_a_failure_not_a_crash(self):
        os.remove(os.path.join(self._tmp.name, self.sid, "RUNS.parquet"))
        got = snapshot.verify_files(self.loc, self.sid)
        self.assertEqual(got.checked, 3)
        self.assertEqual(len(got.failures), 1)
        path, expected, actual = got.failures[0]
        self.assertEqual(path, "RUNS.parquet")
        self.assertIsNone(actual)


if __name__ == "__main__":
    unittest.main()
