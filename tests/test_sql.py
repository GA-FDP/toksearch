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

import unittest
import os

from toksearch.sql.mssql import connect_d3drdb
from toksearch import Pipeline
from toksearch.record import MissingShotNumber
from toksearch.pipeline.record_pipeline import MissingColumnName


# if os.getenv('TOKSEARCH_D3DRDB_TEST', 'no') == 'yes':
if False:

    class TestConnectD3drdb(unittest.TestCase):
        def test_simple_query(self):
            with connect_d3drdb() as conn:
                cursor = conn.cursor()
                cursor.execute("select t_ip_flat from summaries where shot=165920")
                res = cursor.fetchone()[0]

            self.assertEqual(res, 1180.5)

    class TestPipelineFromSql(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.conn = connect_d3drdb()

        def test_pipeline_from_query_with_mssql(self):
            query = "select shot, t_ip_flat from summaries where shot=%d"
            shot = 165920
            pipe = Pipeline.from_sql(self.conn, query, shot)

            results = pipe.compute_serial()

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["shot"], shot)
            self.assertEqual(results[0]["t_ip_flat"], 1180.5)

        def test_pipeline_from_query_with_anonymous_fields(self):
            query = "select shot, 7 from summaries where shot=%d"
            shot = 165920

            with self.assertRaises(MissingColumnName):
                pipe = Pipeline.from_sql(self.conn, query, shot)


from unittest import mock


class TestResolveCredential(unittest.TestCase):
    """_resolve_credential: pure function reading a two-line password file.

    Phase 1 connect_d3drdb resolved credentials by reading a file with
    username on line 1 and password on line 2. This carries the behavior
    forward into the catalog-aware path.
    """

    def _locator(self, *, auth_kind="password_file", auth_path="~/.test.login"):
        from fdp_schema import SqlLocator, AuthHint
        return SqlLocator(
            name="testdb",
            driver="mssql",
            host="h",
            port=8001,
            database="testdb",
            auth=AuthHint(kind=auth_kind, path=auth_path) if auth_kind else None,
        )

    def test_reads_password_file_from_locator_auth_path(self):
        from toksearch.sql.mssql import _resolve_credential
        loc = self._locator(auth_path="/tmp/test.login")
        with mock.patch(
            "pathlib.Path.read_text",
            return_value="theuser\nthepass\n",
        ):
            user, password = _resolve_credential(None, loc, None)
        self.assertEqual((user, password), ("theuser", "thepass"))

    def test_password_file_override_wins_over_locator_auth_path(self):
        from toksearch.sql.mssql import _resolve_credential
        loc = self._locator(auth_path="/never/read.login")
        with mock.patch(
            "pathlib.Path.read_text",
            return_value="overuser\noverpass\n",
        ) as read_text:
            user, password = _resolve_credential(None, loc, "/tmp/o.login")
        # Verify the override path was opened, not the locator's.
        called_path = str(read_text.call_args.args[0]) if read_text.call_args.args else ""
        self.assertEqual((user, password), ("overuser", "overpass"))

    def test_explicit_username_wins_over_file_user(self):
        from toksearch.sql.mssql import _resolve_credential
        loc = self._locator()
        with mock.patch(
            "pathlib.Path.read_text",
            return_value="fromfile\nfilepass\n",
        ):
            user, password = _resolve_credential("kwarg_user", loc, None)
        self.assertEqual(user, "kwarg_user")
        self.assertEqual(password, "filepass")

    def test_tolerates_blank_lines_in_password_file(self):
        from toksearch.sql.mssql import _resolve_credential
        loc = self._locator()
        with mock.patch(
            "pathlib.Path.read_text",
            return_value="\nuser1\n\npass2\n",
        ):
            user, password = _resolve_credential(None, loc, None)
        self.assertEqual((user, password), ("user1", "pass2"))

    def test_expands_user_tilde_in_path(self):
        from toksearch.sql.mssql import _resolve_credential
        import os
        loc = self._locator(auth_path="~/.d3d.login")
        captured = {}
        def fake_read_text(self):
            captured["path"] = str(self)
            return "u\np\n"
        with mock.patch("pathlib.Path.read_text", new=fake_read_text):
            _resolve_credential(None, loc, None)
        self.assertNotIn("~", captured["path"])
        self.assertIn(os.path.expanduser("~"), captured["path"])

    def test_raises_runtimeerror_when_no_credential_source(self):
        from toksearch.sql.mssql import _resolve_credential
        loc = self._locator(auth_kind=None)
        with self.assertRaisesRegex(RuntimeError, "credential source"):
            _resolve_credential(None, loc, None)


class TestDiscoverCatalogs(unittest.TestCase):
    """_discover_catalogs reads the fdp_schema.catalogs entry-point group
    and parses each contributed YAML. Cached for process lifetime."""

    def setUp(self):
        from toksearch.sql.mssql import _discover_catalogs
        _discover_catalogs.cache_clear()

    def _ep(self, name: str, yaml: str):
        ep = mock.MagicMock()
        ep.name = name
        ep.value = f"mock:{name}"
        src = mock.MagicMock()
        src.read_text.return_value = yaml
        ep.load.return_value = src
        return ep

    def test_returns_dict_keyed_by_tokamak_name(self):
        from toksearch.sql.mssql import _discover_catalogs
        eps = [self._ep("d3d", "schema_version: 1\nname: d3d\n")]
        with mock.patch("toksearch.sql.mssql.entry_points", return_value=eps):
            result = _discover_catalogs()
        self.assertEqual(set(result.keys()), {"d3d"})
        self.assertEqual(result["d3d"].name, "d3d")

    def test_loads_full_locator_data(self):
        from toksearch.sql.mssql import _discover_catalogs
        eps = [self._ep("d3d", """
schema_version: 1
name: d3d
locators:
  - kind: sql
    name: d3drdb
    driver: mssql
    host: d3drdb.gat.com
    port: 8001
    database: d3drdb
""")]
        with mock.patch("toksearch.sql.mssql.entry_points", return_value=eps):
            result = _discover_catalogs()
        loc = result["d3d"].locators[0]
        self.assertEqual(loc.kind, "sql")
        self.assertEqual(loc.host, "d3drdb.gat.com")

    def test_duplicate_tokamak_name_raises(self):
        from toksearch.sql.mssql import _discover_catalogs
        eps = [
            self._ep("a", "schema_version: 1\nname: x\n"),
            self._ep("b", "schema_version: 1\nname: x\n"),
        ]
        with mock.patch("toksearch.sql.mssql.entry_points", return_value=eps):
            with self.assertRaisesRegex(RuntimeError, "Duplicate tokamak name"):
                _discover_catalogs()

    def test_result_is_cached(self):
        from toksearch.sql.mssql import _discover_catalogs
        eps = [self._ep("d3d", "schema_version: 1\nname: d3d\n")]
        with mock.patch(
            "toksearch.sql.mssql.entry_points", return_value=eps
        ) as ep_mock:
            _discover_catalogs()
            _discover_catalogs()
            _discover_catalogs()
        self.assertEqual(ep_mock.call_count, 1)


class TestConnectTokamakSql(unittest.TestCase):
    """connect_tokamak_sql: catalog-aware mssql connect helper."""

    YAML = """
schema_version: 1
name: testtok
locators:
  - kind: sql
    name: testdb
    driver: mssql
    host: testdb.example.com
    port: 8001
    database: testdb
    tdsver: "7.0"
    auth: { kind: password_file, path: ~/.testdb.login }
"""

    def setUp(self):
        from toksearch.sql.mssql import _discover_catalogs
        _discover_catalogs.cache_clear()

    def _patch_entry_points(self):
        ep = mock.MagicMock()
        ep.name = "testtok"
        ep.value = "mock:testtok"
        src = mock.MagicMock()
        src.read_text.return_value = self.YAML
        ep.load.return_value = src
        return mock.patch(
            "toksearch.sql.mssql.entry_points", return_value=[ep]
        )

    def test_calls_pymssql_with_catalog_values(self):
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch(
                "toksearch.sql.mssql._resolve_credential",
                return_value=("u", "p"),
            ):
                with mock.patch("pymssql.connect") as connect:
                    connect_tokamak_sql("testtok", "testdb")
        connect.assert_called_once_with(
            "testdb.example.com", "u", "p", "testdb", port="8001"
        )

    def test_kwarg_overrides_catalog_host(self):
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch(
                "toksearch.sql.mssql._resolve_credential",
                return_value=("u", "p"),
            ):
                with mock.patch("pymssql.connect") as connect:
                    connect_tokamak_sql("testtok", "testdb", host="other.example.com")
        connect.assert_called_once_with(
            "other.example.com", "u", "p", "testdb", port="8001"
        )

    def test_kwarg_db_overrides_catalog_database(self):
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch(
                "toksearch.sql.mssql._resolve_credential",
                return_value=("u", "p"),
            ):
                with mock.patch("pymssql.connect") as connect:
                    connect_tokamak_sql("testtok", "testdb", db="code_rundb")
        # db kwarg → 4th positional arg (database)
        connect.assert_called_once_with(
            "testdb.example.com", "u", "p", "code_rundb", port="8001"
        )

    def test_kwarg_port_overrides_catalog_port(self):
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch(
                "toksearch.sql.mssql._resolve_credential",
                return_value=("u", "p"),
            ):
                with mock.patch("pymssql.connect") as connect:
                    connect_tokamak_sql("testtok", "testdb", port=9999)
        connect.assert_called_once_with(
            "testdb.example.com", "u", "p", "testdb", port="9999"
        )

    def test_tdsver_setdefault_does_not_override(self):
        import os
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch.dict(os.environ, {"TDSVER": "8.0"}, clear=True):
                with mock.patch(
                    "toksearch.sql.mssql._resolve_credential",
                    return_value=("u", "p"),
                ):
                    with mock.patch("pymssql.connect"):
                        connect_tokamak_sql("testtok", "testdb")
                # Pre-existing TDSVER kept.
                self.assertEqual(os.environ["TDSVER"], "8.0")

    def test_tdsver_set_when_not_in_env(self):
        import os
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch.dict(os.environ, {}, clear=True):
                with mock.patch(
                    "toksearch.sql.mssql._resolve_credential",
                    return_value=("u", "p"),
                ):
                    with mock.patch("pymssql.connect"):
                        connect_tokamak_sql("testtok", "testdb")
                self.assertEqual(os.environ.get("TDSVER"), "7.0")

    def test_explicit_password_skips_credential_file_read(self):
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch(
                "toksearch.sql.mssql._resolve_credential"
            ) as resolve:
                with mock.patch("pymssql.connect") as connect:
                    connect_tokamak_sql(
                        "testtok", "testdb",
                        username="bob", password="explicit",
                    )
        resolve.assert_not_called()
        connect.assert_called_once_with(
            "testdb.example.com", "bob", "explicit", "testdb", port="8001"
        )

    def test_explicit_password_without_username_defaults_to_os_user(self):
        """Phase 1 connect_d3drdb signature was `username=USERNAME, password=None`.
        When a caller now passes only `password=`, preserve the OS-current-user
        default so pymssql doesn't receive None."""
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with mock.patch("getpass.getuser", return_value="osuser"):
                with mock.patch("pymssql.connect") as connect:
                    connect_tokamak_sql("testtok", "testdb", password="x")
        connect.assert_called_once_with(
            "testdb.example.com", "osuser", "x", "testdb", port="8001"
        )

    def test_unknown_tokamak_raises_keyerror_with_available_list(self):
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with self.assertRaises(KeyError) as ctx:
                connect_tokamak_sql("nonexistent", "testdb")
        self.assertIn("Available", str(ctx.exception))
        self.assertIn("testtok", str(ctx.exception))

    def test_unknown_locator_name_raises_keyerror_with_available_list(self):
        from toksearch.sql.mssql import connect_tokamak_sql
        with self._patch_entry_points():
            with self.assertRaises(KeyError) as ctx:
                connect_tokamak_sql("testtok", "nonexistent")
        self.assertIn("Available", str(ctx.exception))
        self.assertIn("testdb", str(ctx.exception))

    def test_unsupported_driver_raises_notimplementederror(self):
        yaml = """
schema_version: 1
name: testtok
locators:
  - kind: sql
    name: pgdb
    driver: postgres
    host: pg.example.com
    database: pgdb
"""
        from toksearch.sql.mssql import connect_tokamak_sql
        ep = mock.MagicMock()
        ep.name = "testtok"
        ep.value = "mock:testtok"
        src = mock.MagicMock()
        src.read_text.return_value = yaml
        ep.load.return_value = src
        with mock.patch(
            "toksearch.sql.mssql.entry_points", return_value=[ep]
        ):
            with self.assertRaisesRegex(NotImplementedError, "postgres"):
                connect_tokamak_sql("testtok", "pgdb")
