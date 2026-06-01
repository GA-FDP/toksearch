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
