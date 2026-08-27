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

import os
import unittest
from dataclasses import dataclass
from unittest.mock import patch, MagicMock

from toksearch.interfaces import req_interface as ri


@dataclass
class FakeReq:
    mds_path: str
    shot: int
    treename: str = "ELECTRONS"

    def as_key(self):
        return (self.mds_path, self.shot, self.treename)


class TestAtlasReachable(unittest.TestCase):
    def setUp(self):
        ri._atlas_reachable_cache = None
        self._env_backup = os.environ.pop(ri._NO_ATLAS_ENV_VAR, None)

    def tearDown(self):
        ri._atlas_reachable_cache = None
        if self._env_backup is not None:
            os.environ[ri._NO_ATLAS_ENV_VAR] = self._env_backup

    def test_no_atlas_env_var_short_circuits(self):
        os.environ[ri._NO_ATLAS_ENV_VAR] = "1"
        with patch.object(ri.socket, "create_connection") as mock_connect:
            self.assertFalse(ri._atlas_reachable())
        mock_connect.assert_not_called()

    def test_reachable_when_socket_connects(self):
        with patch.object(ri.socket, "create_connection") as mock_connect:
            mock_connect.return_value.__enter__ = MagicMock()
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            self.assertTrue(ri._atlas_reachable())
        mock_connect.assert_called_once_with(
            (ri._FALLBACK_MDS_SERVER, ri._ATLAS_PORT), timeout=2.0
        )

    def test_unreachable_when_socket_raises(self):
        with patch.object(ri.socket, "create_connection", side_effect=OSError):
            self.assertFalse(ri._atlas_reachable())

    def test_result_is_cached(self):
        with patch.object(ri.socket, "create_connection") as mock_connect:
            mock_connect.return_value.__enter__ = MagicMock()
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            ri._atlas_reachable()
            ri._atlas_reachable()
        mock_connect.assert_called_once()


class TestFetchManyFromReq(unittest.TestCase):
    def setUp(self):
        ri._atlas_reachable_cache = None

    def tearDown(self):
        ri._atlas_reachable_cache = None

    def test_ptdata_reqs_batched_in_one_ptdata_group(self):
        reqs = [FakeReq("BT", 12345, "__ptdata__"), FakeReq("IP", 200000, "__ptdata__")]
        with patch.object(ri, "_atlas_reachable", return_value=True), \
             patch.object(ri, "_fetch_ptdata_group_via_server") as mock_ptdata, \
             patch.object(ri, "_fetch_tree_group_via_server") as mock_tree:
            mock_ptdata.side_effect = lambda server, group: {
                r.as_key(): f"{server}-ptdata" for r in group
            }
            result = ri.fetch_many_from_req(reqs)

        mock_ptdata.assert_called_once_with(ri._FDP_THINCLIENT_SERVER, reqs)
        mock_tree.assert_not_called()
        self.assertEqual(result[reqs[0].as_key()], f"{ri._FDP_THINCLIENT_SERVER}-ptdata")

    def test_tree_reqs_batched_per_group_fdp_first(self):
        reqs = [
            FakeReq("BT", 12345, "TREE_A"),
            FakeReq("IP", 12345, "TREE_A"),
            FakeReq("NE", 12345, "TREE_B"),
        ]
        with patch.object(ri, "_atlas_reachable", return_value=True), \
             patch.object(ri, "_fetch_tree_group_via_server") as mock_tree, \
             patch.object(ri, "_fetch_ptdata_group_via_server") as mock_ptdata:
            mock_tree.side_effect = lambda server, treename, shot, group: {
                r.as_key(): f"{server}-{treename}-{shot}" for r in group
            }
            result = ri.fetch_many_from_req(reqs)

        self.assertEqual(mock_tree.call_count, 2)
        mock_tree.assert_any_call(ri._FDP_THINCLIENT_SERVER, "TREE_A", 12345, [reqs[0], reqs[1]])
        mock_tree.assert_any_call(ri._FDP_THINCLIENT_SERVER, "TREE_B", 12345, [reqs[2]])
        mock_ptdata.assert_not_called()
        self.assertEqual(result[reqs[0].as_key()], f"{ri._FDP_THINCLIENT_SERVER}-TREE_A-12345")

    def test_ptdata_and_tree_groups_both_dispatched(self):
        reqs = [FakeReq("BT", 12345, "__ptdata__"), FakeReq("NE", 12345, "TREE_A")]
        with patch.object(ri, "_atlas_reachable", return_value=True), \
             patch.object(ri, "_fetch_ptdata_group_via_server") as mock_ptdata, \
             patch.object(ri, "_fetch_tree_group_via_server") as mock_tree:
            mock_ptdata.side_effect = lambda server, group: {r.as_key(): "pt" for r in group}
            mock_tree.side_effect = lambda server, tn, sh, group: {r.as_key(): "tr" for r in group}
            result = ri.fetch_many_from_req(reqs)

        mock_ptdata.assert_called_once()
        mock_tree.assert_called_once()
        self.assertEqual(result[reqs[0].as_key()], "pt")
        self.assertEqual(result[reqs[1].as_key()], "tr")

    def test_fdp_failure_falls_back_to_atlas(self):
        reqs = [FakeReq("BT", 12345, "TREE_A")]

        def _side_effect(server, treename, shot, group):
            if server == ri._FDP_THINCLIENT_SERVER:
                raise RuntimeError("fdp down")
            return {r.as_key(): "atlas-batch" for r in group}

        with patch.object(ri, "_atlas_reachable", return_value=True), \
             patch.object(ri, "_fetch_tree_group_via_server", side_effect=_side_effect) as mock_tree:
            result = ri.fetch_many_from_req(reqs)

        self.assertEqual(mock_tree.call_count, 2)
        mock_tree.assert_any_call(ri._FDP_THINCLIENT_SERVER, "TREE_A", 12345, reqs)
        mock_tree.assert_any_call(ri._FALLBACK_MDS_SERVER, "TREE_A", 12345, reqs)
        self.assertEqual(result[reqs[0].as_key()], "atlas-batch")

    def test_atlas_skipped_when_unreachable(self):
        reqs = [FakeReq("BT", 12345, "TREE_A")]
        boom = RuntimeError("fdp down")
        with patch.object(ri, "_atlas_reachable", return_value=False), \
             patch.object(ri, "_fetch_tree_group_via_server", side_effect=boom) as mock_tree:
            result = ri.fetch_many_from_req(reqs)

        mock_tree.assert_called_once_with(ri._FDP_THINCLIENT_SERVER, "TREE_A", 12345, reqs)
        self.assertIs(result[reqs[0].as_key()], boom)

    def test_both_servers_fail_stores_exception_in_band(self):
        reqs = [FakeReq("BT", 12345, "TREE_A"), FakeReq("IP", 12345, "TREE_A")]
        boom = RuntimeError("all down")
        with patch.object(ri, "_atlas_reachable", return_value=True), \
             patch.object(ri, "_fetch_tree_group_via_server", side_effect=boom):
            result = ri.fetch_many_from_req(reqs)

        self.assertIs(result[reqs[0].as_key()], boom)
        self.assertIs(result[reqs[1].as_key()], boom)


class TestFetchTreeGroupViaServer(unittest.TestCase):
    def test_batches_group_via_get_many(self):
        reqs = [FakeReq("BT", 12345, "TREE_A"), FakeReq("IP", 12345, "TREE_A")]

        mock_conn = MagicMock()
        mock_many = MagicMock()
        mock_conn.getMany.return_value = mock_many

        def _get(name):
            data = {"BT": MagicMock(), "IP": MagicMock()}
            data["BT"].data.return_value = 1.0
            data["IP"].data.return_value = 2.0
            return data[name]

        mock_many.get.side_effect = _get

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_tree_group_via_server(ri._FDP_THINCLIENT_SERVER, "TREE_A", 12345, reqs)

        mock_registry_cls.return_value.connect.assert_called_once_with(ri._FDP_THINCLIENT_SERVER)
        mock_conn.openTree.assert_called_once_with("TREE_A", 12345)
        self.assertEqual(mock_many.append.call_count, 2)
        mock_many.execute.assert_called_once()
        self.assertEqual(result[reqs[0].as_key()], 1.0)
        self.assertEqual(result[reqs[1].as_key()], 2.0)

    def test_per_key_failure_is_stored_in_band(self):
        reqs = [FakeReq("BT", 12345, "TREE_A")]

        mock_conn = MagicMock()
        mock_many = MagicMock()
        mock_conn.getMany.return_value = mock_many
        mock_many.get.side_effect = ValueError("%TREE-E-NODATA")

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_tree_group_via_server(ri._FALLBACK_MDS_SERVER, "TREE_A", 12345, reqs)

        self.assertIsInstance(result[reqs[0].as_key()], ValueError)


class TestFetchPtdataGroupViaServer(unittest.TestCase):
    def test_batches_ptdata_without_opening_a_tree(self):
        reqs = [FakeReq("BT", 12345, "__ptdata__"), FakeReq("IP", 200000, "__ptdata__")]

        mock_conn = MagicMock()
        mock_many = MagicMock()
        mock_conn.getMany.return_value = mock_many

        values = {
            "d0": 1.0, "t0": 10.0, "r0": [1, 2, 3, 4, 5],
            "d1": 2.0, "t1": 20.0, "r1": [6, 7, 8, 9, 10],
        }

        def _get(name):
            node = MagicMock()
            node.data.return_value = values[name]
            return node

        mock_many.get.side_effect = _get

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_ptdata_group_via_server(ri._FDP_THINCLIENT_SERVER, reqs)

        mock_registry_cls.return_value.connect.assert_called_once_with(ri._FDP_THINCLIENT_SERVER)
        mock_conn.openTree.assert_not_called()
        # Three TDI forms per req, all in one getMany.
        self.assertEqual(mock_many.append.call_count, 6)
        mock_many.append.assert_any_call("d0", 'ptdata2("BT", 12345)')
        mock_many.append.assert_any_call("t0", 'dim_of(ptdata2("BT", 12345), 0)')
        mock_many.append.assert_any_call("r0", 'pthead2("BT", 12345), __rarray')
        mock_many.execute.assert_called_once()
        self.assertEqual(
            result[reqs[0].as_key()],
            {"data": 1.0, "times": 10.0, "rarray": [1, 2, 3, 4, 5]},
        )
        self.assertEqual(
            result[reqs[1].as_key()],
            {"data": 2.0, "times": 20.0, "rarray": [6, 7, 8, 9, 10]},
        )

    def test_per_req_failure_is_stored_in_band(self):
        reqs = [FakeReq("NOSUCHPOINT", 12345, "__ptdata__")]

        mock_conn = MagicMock()
        mock_many = MagicMock()
        mock_conn.getMany.return_value = mock_many
        mock_many.get.side_effect = ValueError("%TREE-E-NODATA")

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_ptdata_group_via_server(ri._FALLBACK_MDS_SERVER, reqs)

        self.assertIsInstance(result[reqs[0].as_key()], ValueError)


if __name__ == "__main__":
    unittest.main()
