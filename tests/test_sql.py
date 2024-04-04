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
