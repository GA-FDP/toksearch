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
from dataclasses import dataclass
from unittest.mock import patch, call, MagicMock

import MDSplus
from MDSplus.connection import MdsIpException

from toksearch.interfaces import req_interface as ri


@dataclass
class FakeReq:
    mds_path: str
    shot: int
    treename: str = "ELECTRONS"
    isolated: bool = False

    def as_key(self):
        return (self.mds_path, self.shot, self.treename)


_SERVER_A = "server-a"
_SERVER_B = "server-b"
_TREELESS = "__treeless__"


class TestFetchManyFromReq(unittest.TestCase):
    """Grouping and server-fallback logic, with both group fetchers stubbed out."""

    def setUp(self):
        self.mock_servers = self.enterContext(
            patch.object(ri, "_servers", return_value=[_SERVER_A, _SERVER_B])
        )
        self.mock_treeless = MagicMock()
        self.enterContext(
            patch.dict(ri._TREELESS_GROUP_FETCHERS, {_TREELESS: self.mock_treeless})
        )
        self.mock_tree = self.enterContext(
            patch.object(ri, "_fetch_tree_group_via_server")
        )
        self.mock_disconnect = self.enterContext(
            patch.object(ri, "MdsConnectionRegistry")
        ).return_value.disconnect

    def test_treeless_reqs_batched_in_one_group_across_shots(self):
        reqs = [FakeReq("BT", 12345, _TREELESS), FakeReq("IP", 200000, _TREELESS)]
        self.mock_treeless.side_effect = lambda server, group: {
            r.as_key(): f"{server}-treeless" for r in group
        }

        result = ri.fetch_many_from_req(reqs)

        self.mock_treeless.assert_called_once_with(_SERVER_A, reqs)
        self.mock_tree.assert_not_called()
        self.assertEqual(result[reqs[0].as_key()], f"{_SERVER_A}-treeless")

    def test_tree_reqs_batched_per_group_first_server_first(self):
        reqs = [
            FakeReq("BT", 12345, "TREE_A"),
            FakeReq("IP", 12345, "TREE_A"),
            FakeReq("NE", 12345, "TREE_B"),
        ]
        self.mock_tree.side_effect = lambda server, treename, shot, group: {
            r.as_key(): f"{server}-{treename}-{shot}" for r in group
        }

        result = ri.fetch_many_from_req(reqs)

        self.assertEqual(
            self.mock_tree.call_args_list,
            [
                call(_SERVER_A, "TREE_A", 12345, [reqs[0], reqs[1]]),
                call(_SERVER_A, "TREE_B", 12345, [reqs[2]]),
            ],
        )
        self.mock_treeless.assert_not_called()
        self.assertEqual(result[reqs[0].as_key()], f"{_SERVER_A}-TREE_A-12345")

    def test_first_server_failure_falls_back_to_next(self):
        reqs = [FakeReq("BT", 12345, "TREE_A")]

        # One entry per expected attempt: the first server is down for both of
        # its attempts, the next server returns data. Sequenced rather than
        # dispatched on the server argument so the raise happens inside mock
        # rather than in a test-local function, which debuggers flag as
        # user-unhandled.
        self.mock_tree.side_effect = [
            RuntimeError("server-a down"),
            RuntimeError("server-a down"),
            {reqs[0].as_key(): "server-b-batch"},
        ]

        # assertLogs both captures the expected "trying next server" warning --
        # keeping it out of the test output, where it reads as a failure -- and
        # asserts the fallback was actually reported.
        with self.assertLogs(ri._log, "WARNING") as logs:
            result = ri.fetch_many_from_req(reqs)

        self.assertIn("server-a down", "\n".join(logs.output))
        self.assertEqual(
            self.mock_tree.call_args_list,
            [
                call(_SERVER_A, "TREE_A", 12345, reqs),
                call(_SERVER_A, "TREE_A", 12345, reqs),
                call(_SERVER_B, "TREE_A", 12345, reqs),
            ],
        )
        self.assertEqual(result[reqs[0].as_key()], "server-b-batch")

    def test_single_server_failure_stores_exception_in_band(self):
        reqs = [FakeReq("BT", 12345, "TREE_A")]
        boom = RuntimeError("server-a down")
        self.mock_servers.return_value = [_SERVER_A]
        self.mock_tree.side_effect = boom

        with self.assertLogs(ri._log, "WARNING"):
            result = ri.fetch_many_from_req(reqs)

        self.assertEqual(self.mock_tree.call_count, ri._ATTEMPTS_PER_SERVER)
        self.mock_tree.assert_called_with(_SERVER_A, "TREE_A", 12345, reqs)
        self.assertIs(result[reqs[0].as_key()], boom)

    def test_all_servers_fail_stores_exception_in_band(self):
        reqs = [FakeReq("BT", 12345, "TREE_A"), FakeReq("IP", 12345, "TREE_A")]
        boom = RuntimeError("all down")
        self.mock_tree.side_effect = boom

        with self.assertLogs(ri._log, "WARNING"):
            result = ri.fetch_many_from_req(reqs)

        self.assertEqual(self.mock_tree.call_count, 2 * ri._ATTEMPTS_PER_SERVER)
        self.assertIs(result[reqs[0].as_key()], boom)
        self.assertIs(result[reqs[1].as_key()], boom)

    def test_a_failed_batch_is_retried_on_a_fresh_connection(self):
        """The bug this exists for: over fdp:// a failed batch can leave the
        relay session retired, and MdsConnectionRegistry caches that dead
        MDSplus.Connection for the life of the process. Without dropping it,
        every later group fails identically -- 26 test failures from one slow
        call in GA-FDP/imas_composer CI run 33033220932.
        """
        reqs = [FakeReq("BT", 12345, _TREELESS)]
        self.mock_treeless.side_effect = [
            RuntimeError("%MDSPLUS-E-ERROR, Error"),
            {reqs[0].as_key(): "second-try"},
        ]

        with self.assertLogs(ri._log, "INFO") as logs:
            result = ri.fetch_many_from_req(reqs)

        # A retry that succeeds is not worth a warning.
        self.assertEqual([r.levelname for r in logs.records], ["INFO"])
        # Retried against the same server, not handed straight to the next.
        self.assertEqual(
            self.mock_treeless.call_args_list,
            [call(_SERVER_A, reqs), call(_SERVER_A, reqs)],
        )
        self.mock_disconnect.assert_called_once_with(_SERVER_A)
        self.assertEqual(result[reqs[0].as_key()], "second-try")

    def test_a_poisoned_connection_is_dropped_even_when_the_retry_fails(self):
        """Leaving it cached would carry the dead session into the next group,
        which is precisely the cascade being fixed."""
        reqs = [FakeReq("BT", 12345, _TREELESS)]
        boom = RuntimeError("%MDSPLUS-E-ERROR, Error")
        self.mock_servers.return_value = [_SERVER_A]
        self.mock_treeless.side_effect = boom

        with self.assertLogs(ri._log, "WARNING"):
            result = ri.fetch_many_from_req(reqs)

        self.assertEqual(
            self.mock_disconnect.call_args_list,
            [call(_SERVER_A)] * ri._ATTEMPTS_PER_SERVER,
        )
        self.assertIs(result[reqs[0].as_key()], boom)

    def test_a_succeeding_batch_keeps_its_connection(self):
        """The retry must not cost a redial on the happy path -- that is the
        workload the cached connection exists for."""
        reqs = [FakeReq("BT", 12345, _TREELESS)]
        self.mock_treeless.side_effect = lambda server, group: {
            r.as_key(): "ok" for r in group
        }

        ri.fetch_many_from_req(reqs)

        self.mock_treeless.assert_called_once()
        self.mock_disconnect.assert_not_called()

    def test_dropping_a_connection_never_masks_the_real_failure(self):
        """disconnect() talks to the network too. If it raises, the caller must
        still see why the fetch failed, not why the cleanup did."""
        reqs = [FakeReq("BT", 12345, "TREE_A")]
        boom = RuntimeError("the real failure")
        self.mock_servers.return_value = [_SERVER_A]
        self.mock_tree.side_effect = boom
        self.mock_disconnect.side_effect = OSError("cleanup failed")

        with self.assertLogs(ri._log, "WARNING"):
            result = ri.fetch_many_from_req(reqs)

        self.assertIs(result[reqs[0].as_key()], boom)


class TestOverridesAndLogging(unittest.TestCase):
    setUp = TestFetchManyFromReq.setUp

    def test_exhausting_a_server_logs_one_warning_without_the_group(self):
        reqs = [FakeReq(f"P{i}", 12345, "TREE_A") for i in range(50)]
        self.mock_servers.return_value = [_SERVER_A]
        self.mock_tree.side_effect = RuntimeError("down")

        with self.assertLogs(ri._log, "WARNING") as logs:
            ri.fetch_many_from_req(reqs)

        self.assertEqual(len(logs.records), 1)
        msg = logs.records[0].getMessage()
        self.assertIn("50 reqs", msg)
        self.assertIn("giving up", msg)
        self.assertNotIn("P49", msg)

    def test_servers_argument_overrides_registered_servers(self):
        reqs = [FakeReq("BT", 12345, "TREE_A")]
        self.mock_tree.side_effect = lambda server, tn, sh, group: {
            r.as_key(): server for r in group
        }

        result = ri.fetch_many_from_req(reqs, servers=["explicit"])

        self.assertEqual(result[reqs[0].as_key()], "explicit")
        self.mock_servers.assert_not_called()

    def test_treeless_fetchers_argument_adds_a_group_fetcher(self):
        reqs = [FakeReq("BT", 12345, "__mine__"), FakeReq("IP", 1, "__mine__")]
        mine = MagicMock(side_effect=lambda server, group: {
            r.as_key(): "mine" for r in group
        })

        result = ri.fetch_many_from_req(reqs, treeless_fetchers={"__mine__": mine})

        mine.assert_called_once_with(_SERVER_A, reqs)
        self.mock_tree.assert_not_called()
        self.assertEqual(set(result.values()), {"mine"})


class TestRegisterDevice(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(ri, "_servers", ri._servers))
        self.enterContext(patch.dict(ri._TREELESS_GROUP_FETCHERS, clear=True))

    def test_registered_servers_and_fetchers_become_the_defaults(self):
        fetcher = MagicMock(side_effect=lambda server, group: {
            r.as_key(): server for r in group
        })
        ri.register_device(
            servers=lambda: ["registered"], treeless_fetchers={_TREELESS: fetcher}
        )

        reqs = [FakeReq("BT", 12345, _TREELESS)]
        result = ri.fetch_many_from_req(reqs)

        self.assertEqual(result[reqs[0].as_key()], "registered")


class TestNoServersConfigured(unittest.TestCase):
    def test_fetch_raises_without_device_package(self):
        # No device package is imported in toksearch's own tests, so _servers
        # is still the generic default.
        with self.assertRaisesRegex(RuntimeError, "No MDSplus servers configured"):
            ri.fetch_many_from_req([FakeReq("BT", 12345, "TREE_A")])

    def test_no_reqs_needs_no_servers(self):
        self.assertEqual(ri.fetch_many_from_req([]), {})


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
            result = ri._fetch_tree_group_via_server(
                _SERVER_A, "TREE_A", 12345, reqs
            )

        mock_registry_cls.return_value.connect.assert_called_once_with(
            _SERVER_A
        )
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
        # A node with no data surfaces as an MdsIpException out of many.get(),
        # with the real error text only available in the execute() payload.
        mock_many.get.side_effect = MdsIpException("get failed")
        mock_many.execute.return_value = {"BT": {"error": "%TREE-E-NODATA"}}

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_tree_group_via_server(
                _SERVER_B, "TREE_A", 12345, reqs
            )

        error = result[reqs[0].as_key()]
        self.assertIsInstance(error, MDSplus.mdsExceptions.MdsException)
        self.assertEqual(str(error), "%TREE-E-NODATA")

    def test_isolated_req_fetched_with_get_outside_get_many(self):
        batched = FakeReq("BT", 12345, "TREE_A")
        isolated = FakeReq("NAMELISTS:KEQDSKS", 12345, "TREE_A", isolated=True)

        mock_conn = MagicMock()
        mock_many = MagicMock()
        mock_conn.getMany.return_value = mock_many
        mock_many.get.return_value.data.return_value = 1.0
        mock_conn.get.return_value.data.return_value = ["row 0", "row 1"]

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_tree_group_via_server(
                _SERVER_A, "TREE_A", 12345, [batched, isolated]
            )

        mock_conn.get.assert_called_once_with("NAMELISTS:KEQDSKS")
        mock_many.append.assert_called_once_with("BT", "BT")
        self.assertEqual(result[batched.as_key()], 1.0)
        self.assertEqual(result[isolated.as_key()], ["row 0", "row 1"])

    def test_only_isolated_reqs_skip_get_many(self):
        isolated = FakeReq("NAMELISTS:KEQDSKS", 12345, "TREE_A", isolated=True)

        mock_conn = MagicMock()
        mock_conn.get.return_value.data.return_value = ["row 0", "row 1"]

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_tree_group_via_server(
                _SERVER_A, "TREE_A", 12345, [isolated]
            )

        mock_conn.getMany.assert_not_called()
        self.assertEqual(result[isolated.as_key()], ["row 0", "row 1"])

    def test_isolated_req_failure_is_stored_in_band(self):
        isolated = FakeReq("NAMELISTS:KEQDSKS", 12345, "TREE_A", isolated=True)

        mock_conn = MagicMock()
        mock_conn.get.side_effect = MDSplus.mdsExceptions.TreeNODATA()

        with patch.object(ri, "MdsConnectionRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.connect.return_value = mock_conn
            result = ri._fetch_tree_group_via_server(
                _SERVER_A, "TREE_A", 12345, [isolated]
            )

        self.assertIsInstance(
            result[isolated.as_key()], MDSplus.mdsExceptions.TreeNODATA
        )


if __name__ == "__main__":
    unittest.main()
