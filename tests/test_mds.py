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
import sys
import pickle
import os
import tempfile
import numpy as np
from abc import ABC, abstractmethod
import timeit
import time
import MDSplus as mds
import socket
import subprocess


from toksearch.signal.mds import (
    _BatchedGatherFailed,
    MdsConnectionRegistry,
    MdsTreeRegistry,
    MdsTreePath,
    MdsLocalSignal,
    MdsRemoteSignal,
    MdsSignal,
)

from toksearch.utilities.utilities import set_env, unset_env

this_dir = os.path.dirname(__file__)
trees_dir = os.path.join(this_dir, "trees")

mds_connection_type = mds.Connection
mds_tree_type = mds.Tree

DEFAULT_SHOT = 165920
DEFAULT_TREE = "efit01"
DEFAULT_TREEPATH = trees_dir
DEFAULT_EXPRESSION = r"\ipmhd"


class MdsIpCache:
    host = "localhost"
    _port = None

    @classmethod
    def start_server(cls, treename=DEFAULT_TREE):
        if cls._port is not None:
            return cls._port

        try:
            # Determine the script's directory to locate the mdsip.hosts file
            script_dir = os.path.dirname(os.path.abspath(__file__))
            hosts_file_path = os.path.join(script_dir, "etc", "mdsip.hosts")

            # Find a free port
            cls._port = cls.find_free_port()

            # Add the tdi directory to the MDS_PATH
            # The assumption is that we're running in a conda
            # environment, so we can find the python executable
            # and the tdi directory from that
            python_exe = sys.executable
            bin_dir = os.path.abspath(os.path.dirname(python_exe))
            env_dir = os.path.dirname(bin_dir)
            tdi_dir = os.path.join(env_dir, "tdi")
            print(f"{tdi_dir=}")

            MDS_PATH = os.environ.get("MDS_PATH", "")
            MDS_PATH = f"{MDS_PATH};{tdi_dir};"
            MDS_PATH = tdi_dir
            print(f"{MDS_PATH=}")

            subprocess_env = os.environ.copy()
            subprocess_env["MDS_PATH"] = MDS_PATH

            # Add the tree path to the environment
            subprocess_env[f"{treename}_path"] = trees_dir
            print(f"{trees_dir=}")

            # Start the mdsip server on a free port
            cls.mdsip_process = subprocess.Popen(
                ["mdsip", "-s", "-p", str(cls._port), "-h", hosts_file_path],
                #stdout=subprocess.PIPE,
                #stderr=subprocess.PIPE,
                stdout=sys.stdout,
                stderr=sys.stderr,
                env=subprocess_env,
            )

            print("mdsip server started")
            time.sleep(2)  # Wait a moment to ensure the server is ready
        except Exception as e:
            cls._port = None
            raise e

        max_retries = 10
        for i in range(max_retries):
            try:
                conn = mds.Connection(f"{cls.host}:{cls._port}")
                del conn
                break
            except Exception as e:
                print(
                    f"Failed to connect to MDSplus server at {cls.host}:{cls._port}. Retrying..."
                )
                time.sleep(1)

        if i == max_retries - 1:
            cls._port = None
            raise Exception(
                f"Failed to connect to MDSplus server at {cls.host}:{cls._port}"
            )

    @classmethod
    def stop_server(cls):
        if cls._port is not None:
            cls.mdsip_process.terminate()
            cls.mdsip_process.wait()
            cls._port = None

    @classmethod
    def find_free_port(cls):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    @classmethod
    def port(cls):
        return cls._port


def tearDownModule():
    MdsIpCache.stop_server()


class TestMdsTreePath(unittest.TestCase):
    def test_variable_name(self):
        varname = MdsTreePath.variable_name("d3d")
        self.assertEqual(varname, "d3d_path")

    def test_set_env(self):
        paths = {"d3d": "abcd", "nb": "efg"}
        tree_path = MdsTreePath(**paths)

        # Clear the variables in they already exist
        for key, val in paths.items():
            varname = MdsTreePath.variable_name(key)
            os.environ.pop(varname, None)
            #if varname in os.environ:
            #    del os.environ[varname]
            self.assertNotIn(varname, os.environ)

        # Check if the variables are set inside the context manager
        with tree_path.set_env():
            for key, val in paths.items():
                varname = MdsTreePath.variable_name(key)
                self.assertEqual(paths[key], os.environ[varname])

        # Now check that they've been cleared when outside the context
        # manager scope
        for key, val in paths.items():
            varname = MdsTreePath.variable_name(key)
            self.assertNotIn(varname, os.environ)


class TestMdsSignal(unittest.TestCase):

    def test_remote_location_grabbed_from_environment(self):
        # A URL in TOKSEARCH_MDS_DEFAULT is a server, not a tree path. It used
        # to become a treepath of the literal string "remote://fake.gat.com",
        # which no tree can be opened from -- so setting the environment
        # variable to a server silently produced a local signal that could only
        # fail later.
        server = "fake.gat.com"
        default_location = f"remote://{server}"

        with set_env("TOKSEARCH_MDS_DEFAULT", default_location):
            sig = MdsSignal("blah", "efit01")
        self.assertIsInstance(sig.sig, MdsRemoteSignal)
        self.assertEqual(sig.sig.server, server)

    def test_fdp_location_grabbed_from_environment(self):
        url = "fdp://origin.example.org:8443/mdsip"
        with set_env("TOKSEARCH_MDS_DEFAULT", url):
            sig = MdsSignal("blah", "efit01")
        self.assertIsInstance(sig.sig, MdsRemoteSignal)
        self.assertEqual(sig.sig.server, url)

    def test_local_location_grabbed_from_environment(self):
        default_location = "/some/fake/path"
        with set_env("TOKSEARCH_MDS_DEFAULT", default_location):
            sig = MdsSignal("blah", "efit01")
        self.assertIsInstance(sig.sig, MdsLocalSignal)
        self.assertIsInstance(sig.sig.treepath, MdsTreePath)
        self.assertEqual(sig.sig.treepath.paths["efit01"], default_location)

    def test_create_local_mdsignal(self):
        sig = MdsSignal("blah", "efit01", location="blah")
        self.assertIsInstance(sig.sig, MdsLocalSignal)

    def test_local_has_correct_treepath(self):
        sig = MdsSignal("blah", "efit01", location="abc")
        self.assertEqual(sig.sig.treepath, "abc")

    def test_local_with_double_colons(self):
        sig = MdsSignal("blah", "efit01", location="abc::")
        self.assertEqual(sig.sig.treepath, "abc::")

    def test_create_remote_mdsignal(self):
        sig = MdsSignal("blah", "efit01", location="remote://blah")
        self.assertIsInstance(sig.sig, MdsRemoteSignal)

    def test_create_local_mdsignal_with_treepath(self):
        sig = MdsSignal("blah", "efit01", location=MdsTreePath())
        self.assertIsInstance(sig.sig.treepath, MdsTreePath)
        self.assertIsInstance(sig.sig, MdsLocalSignal)

    def test_create_local_mdsignal_with_location_set_to_none(self):
        with unset_env("TOKSEARCH_MDS_DEFAULT"):
            sig = MdsSignal("blah", "efit01", location=None)
        self.assertIsInstance(sig.sig.treepath, MdsTreePath)
        self.assertIsInstance(sig.sig, MdsLocalSignal)

    def test_fdp_location_is_remote_and_keeps_the_whole_url(self):
        # The scheme selects the MDSplus transport and the path is the relay's
        # prefix on the origin, so unlike 'remote://' neither may be dropped.
        # Before this was handled, urlparse's path was taken as a treepath and
        # the host was discarded: the signal became a LOCAL read of "/mdsip".
        url = "fdp://origin.example.org:8443/mdsip"
        sig = MdsSignal("blah", "efit01", location=url)
        self.assertIsInstance(sig.sig, MdsRemoteSignal)
        self.assertEqual(sig.sig.server, url)

    def test_fdp_location_without_a_path(self):
        url = "fdp://origin.example.org:8443"
        sig = MdsSignal("blah", "efit01", location=url)
        self.assertIsInstance(sig.sig, MdsRemoteSignal)
        self.assertEqual(sig.sig.server, url)

    def test_other_mdsplus_transport_schemes_are_remote(self):
        for url in ("tcp://a.gat.com:8000", "tcpv6://a.gat.com", "udt://a.gat.com"):
            with self.subTest(url=url):
                sig = MdsSignal("blah", "efit01", location=url)
                self.assertIsInstance(sig.sig, MdsRemoteSignal)
                self.assertEqual(sig.sig.server, url)

    def test_unknown_scheme_raises_rather_than_becoming_a_treepath(self):
        # Silently dropping the host turns a mistyped server into a local read
        # that fails much later, somewhere less informative.
        for url in ("pelican://osg-htc.org:443/fdp-d3d/x", "http://example.org/x"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    MdsSignal("blah", "efit01", location=url)

    def test_host_double_colon_path_is_a_treepath(self):
        # urlparse reads 'atlas.gat.com::/trees' as scheme 'atlas.gat.com'
        # (dots are legal in a scheme), so this must be settled before any
        # scheme dispatch or it looks like a server.
        location = "atlas.gat.com::/trees"
        sig = MdsSignal("blah", "efit01", location=location)
        self.assertIsInstance(sig.sig, MdsLocalSignal)
        self.assertEqual(sig.sig.treepath, location)


class GenericTestMdsSignal(ABC):

    def signal(
        self,
        expression=None,
        dims=("times",),
        data_order=None,
        fetch_units=True,
        tree=None,
    ):
        expression = expression or DEFAULT_EXPRESSION
        tree = tree or DEFAULT_TREE
        return self._signal(expression, tree, dims, fetch_units, data_order)

    @abstractmethod
    def _signal(self, expression, tree, dims, fetch_units, data_order):
        pass

    def test_cleanup_shot(self):
        """Just see if it runs without throwing exception"""
        sig = self.signal()
        shot = DEFAULT_SHOT
        sig.cleanup_shot(shot)

    def test_cleanup(self):
        """Just see if it runs without throwing exception"""
        sig = self.signal()
        shot = DEFAULT_SHOT
        sig.cleanup()

    def test_fetch_returns_valid_data_times(self):
        sig = self.signal()
        shot = DEFAULT_SHOT
        results = sig.fetch(shot)
        self.assertGreater(len(results["data"]), 0)
        self.assertGreater(len(results["times"]), 0)

    def test_fetch_returns_valid_data_no_times(self):
        sig = self.signal(dims=None)
        shot = DEFAULT_SHOT
        results = sig.fetch(shot)
        self.assertGreater(len(results["data"]), 0)
        self.assertTrue("times" not in results)
        self.assertTrue("times" not in results["units"])

    def test_fetch_returns_valid_units(self):
        sig = self.signal()
        shot = DEFAULT_SHOT
        results = sig.fetch(shot)
        self.assertIsInstance(results["units"]["data"], str)
        self.assertIsInstance(results["units"]["times"], str)
        self.assertEqual(results["units"]["times"], "ms")
        self.assertEqual(results["units"]["data"], "A")

    def test_fetch_without_units(self):
        sig = self.signal(fetch_units=False)
        shot = DEFAULT_SHOT
        results = sig.fetch(shot)
        self.assertNotIn("units", results)

    def test_multidimensional_fetch(self):
        sig = self.signal(expression=r"\psirz", dims=("r", "z", "times"))
        shot = DEFAULT_SHOT

        results = sig.fetch(shot)
        self.assertIn("data", results)
        self.assertIn("times", results)
        self.assertIn("r", results)
        self.assertIn("z", results)

    def test_multidimensional_fetch_xarray(self):
        shot = DEFAULT_SHOT

        sig = self.signal(
            expression=r"\psirz",
            dims=("r", "z", "times"),
            data_order=("times", "r", "z"),
        )

        data_array = sig.fetch_as_xarray(shot)

        self.assertIn("times", data_array.dims)
        self.assertIn("r", data_array.dims)
        self.assertIn("z", data_array.dims)


class TestMdsRemoteSignal(GenericTestMdsSignal, unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        MdsIpCache.start_server()
        cls.port = MdsIpCache.port()
        cls.host = MdsIpCache.host
        cls.server = f"{cls.host}:{cls.port}"

    def _signal(self, expression, tree, dims, fetch_units, data_order):
        server = self.server
        return MdsRemoteSignal(
            expression,
            tree,
            server,
            dims=dims,
            data_order=data_order,
            fetch_units=fetch_units,
        )

    # def test_fetch_returns_units_missing(self):
    #    expression = DEFAULT_EXPRESSION
    #    sig = self.signal(tree=expression), expression=expression)
    #    #shot = self.defaults.unitless_shot() #use shot that doesnt have units defined
    #    shot = DEFAULT_SHOT
    #    results = sig.fetch(shot)
    #    self.assertEqual(results['units']['data']," ")
    #    self.assertEqual(results['units']['times']," ")


    def test_batched_and_serial_gather_agree(self):
        """The two paths must return the same thing, against a real server.

        Skips on a server that cannot run GetManyExecute. The mdsip server
        this suite starts is one: it fails with LIB-F-KEYNOTFOU regardless of
        MDS_PATH, so batching cannot be exercised here. That is precisely the
        case _do_gather falls back for, and it is covered hermetically by
        TestMdsRemoteBatchedGather.
        """
        sig = self.signal()
        connection = sig.connect()
        connection.openTree(DEFAULT_TREE, DEFAULT_SHOT)
        plan = sig._gather_plan()

        serial = sig._gather_serial(connection, plan)
        try:
            batched = sig._gather_batched(connection, plan)
        except _BatchedGatherFailed:
            self.skipTest("server does not support GetManyExecute")

        self.assertEqual(sorted(serial), sorted(batched))
        for key, expected in serial.items():
            got = batched[key]
            if isinstance(expected, dict):
                self.assertEqual(expected, got)
            else:
                np.testing.assert_array_equal(expected, got)



    def test_do_gather_agrees_with_serial(self):
        """Holds whether or not the server can batch, since _do_gather falls back."""
        sig = self.signal()
        connection = sig.connect()
        connection.openTree(DEFAULT_TREE, DEFAULT_SHOT)
        expected = sig._gather_serial(connection, sig._gather_plan())

        got = sig._do_gather(DEFAULT_SHOT)

        self.assertEqual(sorted(expected), sorted(got))
        for key, value in expected.items():
            if isinstance(value, dict):
                self.assertEqual(value, got[key])
            else:
                np.testing.assert_array_equal(value, got[key])

class TestMdsLocalSignal(GenericTestMdsSignal, unittest.TestCase):
    def _signal(self, expression, tree, dims, fetch_units, data_order):
        return MdsLocalSignal(
            expression,
            tree,
            treepath=DEFAULT_TREEPATH,
            dims=dims,
            data_order=data_order,
            fetch_units=fetch_units,
        )


class TestMdsConnectionRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        MdsIpCache.start_server()
        cls.port = MdsIpCache.port()
        cls.host = MdsIpCache.host
        cls.server = f"{cls.host}:{cls.port}"

    def setUp(self):
        self.registry = MdsConnectionRegistry()

    def tearDown(self):
        del self.registry

    def test_registry_is_singleton(self):
        registry = MdsConnectionRegistry()
        registry2 = MdsConnectionRegistry()
        self.assertIs(registry, registry2)

    def test_repeated_connects_give_same_connection_object(self):
        server = self.server
        registry = MdsConnectionRegistry()
        conn = registry.connect(server)
        conn2 = registry.connect(server)
        self.assertIs(conn, conn2)

    def test_connect_returns_valid_mds_connection(self):
        server = self.server
        registry = MdsConnectionRegistry()
        conn = registry.connect(server)
        self.assertIsInstance(conn, mds_connection_type)

    def test_disconnect_deletes_connection(self):
        server = self.server
        registry = MdsConnectionRegistry()
        conn = registry.connect(server)
        registry.disconnect(server)
        self.assertTrue(server not in registry._connection_map)
        conn2 = registry.connect(server)
        self.assertIsNot(conn, conn2)

    def test_disconnect_calls_conn_disconnect(self):
        # Regression: previous implementation removed the cached connection
        # from the map but never called conn.disconnect(), leaking the
        # underlying mdsip socket until garbage collection.
        from unittest import mock
        server = "fake.host:9999"
        registry = MdsConnectionRegistry()
        fake_conn = mock.MagicMock()
        registry._connection_map[server] = fake_conn
        registry.disconnect(server)
        fake_conn.disconnect.assert_called_once()
        self.assertNotIn(server, registry._connection_map)

    def test_disconnect_unknown_server_is_noop(self):
        registry = MdsConnectionRegistry()
        # Must not raise even if the server was never connected.
        registry.disconnect("never.connected:1234")


    def test_pickling_does_not_drop_the_live_connection_cache(self):
        # __getstate__ used to hand back self.__dict__ itself and blank the
        # map on it, so merely serializing the registry dropped the pickling
        # process's own connections.
        from unittest import mock

        registry = MdsConnectionRegistry()
        registry._connection_map["fake.host:9999"] = mock.MagicMock()
        before = dict(registry._connection_map)
        try:
            pickle.dumps(registry)
            self.assertEqual(registry._connection_map, before)
        finally:
            registry._connection_map.pop("fake.host:9999", None)

    def test_connections_are_not_serialized(self):
        from unittest import mock

        registry = MdsConnectionRegistry()
        registry._connection_map["fake.host:9999"] = mock.MagicMock()
        try:
            self.assertEqual(registry.__getstate__()["_connection_map"], {})
            # A MagicMock cannot be pickled, so getting through dumps at all
            # is itself proof the connections were left out.
            pickle.dumps(registry)
        finally:
            registry._connection_map.pop("fake.host:9999", None)

    def test_unpickling_keeps_the_receiving_process_cache(self):
        from unittest import mock

        registry = MdsConnectionRegistry()
        blob = pickle.dumps(registry)
        conn = mock.MagicMock()
        registry._connection_map["fake.host:9999"] = conn
        try:
            restored = pickle.loads(blob)
            # Unpickling returns this process's singleton, so a restore that
            # replaced _connection_map would be clearing a live cache.
            self.assertIs(restored, registry)
            self.assertIs(restored._connection_map["fake.host:9999"], conn)
        finally:
            registry._connection_map.pop("fake.host:9999", None)

    def test_non_connection_state_survives_a_round_trip(self):
        # Only the connections are process-local; anything else the registry
        # may carry should still travel.
        registry = MdsConnectionRegistry()
        registry.__dict__["_probe"] = 42
        try:
            blob = pickle.dumps(registry)
            del registry.__dict__["_probe"]
            pickle.loads(blob)
            self.assertEqual(registry.__dict__.get("_probe"), 42)
        finally:
            registry.__dict__.pop("_probe", None)


class TestMdsRemoteSignalRetry(unittest.TestCase):
    """Verify that MdsRemoteSignal.gather retries on MDSplusERROR.

    These tests exercise the wrapper logic in isolation by patching the
    inner _do_gather hook, so they don't need a running mdsip server.
    """

    def setUp(self):
        from unittest import mock
        # Wipe any cached connections to keep each test independent.
        MdsConnectionRegistry()._connection_map.clear()
        self.signal = MdsRemoteSignal(r"\foo", "mytree", "fake.host:9999")
        self._mock = mock

    def test_gather_returns_value_on_first_try(self):
        with self._mock.patch.object(
            MdsRemoteSignal, "_do_gather", return_value={"data": 42}
        ) as do_gather:
            result = self.signal.gather(123)
        self.assertEqual(result, {"data": 42})
        self.assertEqual(do_gather.call_count, 1)

    def test_gather_retries_once_on_mdsplus_error(self):
        from MDSplus.mdsExceptions import MDSplusERROR
        # First call raises MDSplusERROR, second call returns a value.
        side_effect = [MDSplusERROR(), {"data": 7}]
        with self._mock.patch.object(
            MdsRemoteSignal, "_do_gather", side_effect=side_effect
        ) as do_gather, self._mock.patch.object(
            MdsConnectionRegistry, "disconnect"
        ) as disconnect:
            result = self.signal.gather(123)
        self.assertEqual(result, {"data": 7})
        self.assertEqual(do_gather.call_count, 2)
        disconnect.assert_called_once_with(self.signal.server)

    def test_gather_does_not_retry_on_tree_errors(self):
        # Tree-class errors don't corrupt the connection -- they should
        # propagate without a reconnect.
        from MDSplus.mdsExceptions import TreeNODATA
        with self._mock.patch.object(
            MdsRemoteSignal, "_do_gather", side_effect=TreeNODATA()
        ) as do_gather, self._mock.patch.object(
            MdsConnectionRegistry, "disconnect"
        ) as disconnect:
            with self.assertRaises(TreeNODATA):
                self.signal.gather(123)
        self.assertEqual(do_gather.call_count, 1)
        disconnect.assert_not_called()

    def test_gather_propagates_second_failure(self):
        from MDSplus.mdsExceptions import MDSplusERROR
        with self._mock.patch.object(
            MdsRemoteSignal,
            "_do_gather",
            side_effect=[MDSplusERROR(), MDSplusERROR()],
        ), self._mock.patch.object(MdsConnectionRegistry, "disconnect"):
            with self.assertRaises(MDSplusERROR):
                self.signal.gather(123)


class TestMdsTreeRegistryPickling(unittest.TestCase):
    """Same defect as MdsConnectionRegistry: __getstate__ blanked the live map.

    Here the casualty is the open-tree cache, dropped without being closed.
    """

    def tearDown(self):
        MdsTreeRegistry().reset()

    def test_pickling_does_not_drop_the_live_tree_cache(self):
        from unittest import mock

        registry = MdsTreeRegistry()
        registry.reset()
        sentinel = mock.MagicMock()
        registry._tree_map["faketree"] = {1234: sentinel}

        pickle.dumps(registry)

        self.assertIs(registry._tree_map["faketree"][1234], sentinel)

    def test_trees_are_not_serialized(self):
        from unittest import mock

        registry = MdsTreeRegistry()
        registry.reset()
        registry._tree_map["faketree"] = {1234: mock.MagicMock()}

        self.assertEqual(registry.__getstate__()["_tree_map"], {})
        pickle.dumps(registry)

    def test_unpickling_keeps_the_receiving_process_trees(self):
        from unittest import mock

        registry = MdsTreeRegistry()
        registry.reset()
        blob = pickle.dumps(registry)
        sentinel = mock.MagicMock()
        registry._tree_map["faketree"] = {1234: sentinel}

        restored = pickle.loads(blob)

        self.assertIs(restored, registry)
        self.assertIs(restored._tree_map["faketree"][1234], sentinel)


class TestMdsTreeRegistry(unittest.TestCase):
    def test_get_tree_returns_none_with_empty_map(self):
        registry = MdsTreeRegistry()
        registry._tree_map = {}

        self.assertIs(registry._get_tree("blah", 123), None)

    def test_open_tree(self):
        registry = MdsTreeRegistry()
        registry.reset()

        treename = DEFAULT_TREE
        treepath = DEFAULT_TREEPATH
        shot = DEFAULT_SHOT

        tree = registry.open_tree(treename, shot, treepath=treepath)
        self.assertTrue(treename in registry._tree_map)
        self.assertTrue(shot in registry._tree_map[treename])
        self.assertIsInstance(tree, mds_tree_type)

        tree2 = registry.open_tree(treename, shot, treepath=treepath)
        self.assertIs(tree, tree2)

    def test_open_tree_with_empty_treepath(self):
        registry = MdsTreeRegistry()
        registry.reset()

        treename = DEFAULT_TREE
        treepath = DEFAULT_TREEPATH
        shot = DEFAULT_SHOT

        print("TREENAME: {}, TREEPATH: {}".format(treename, treepath))

        varname = MdsTreePath.variable_name(treename)
        if varname in os.environ:
            del os.environ[varname]
        with self.assertRaises(mds.TreeNOPATH):
            tree2 = registry.open_tree(treename, shot, treepath=None)

        try:
            with set_env("{}_path".format(treename), treepath):
                tree = registry.open_tree(treename, shot, treepath=None)
        except mds.TreeNOPATH:
            self.fail("open_tree failed to open tree using environment")


class TestMdsCleanupShotKey(unittest.TestCase):
    """cleanup_shot() on a remote signal is a network round trip.

    SignalRegistry sweeps every registered signal once per record, so signals
    that share the resource being cleaned have to advertise that or the sweep
    pays for the same close several times over.
    """

    def test_remote_signals_on_one_server_share_a_key(self):
        a = MdsSignal("a", "efit01", location="remote://atlas.gat.com")
        b = MdsSignal("b", "efit01", location="remote://atlas.gat.com")
        self.assertEqual(a.cleanup_shot_key(), b.cleanup_shot_key())

    def test_treename_is_not_part_of_the_remote_key(self):
        # closeAllTrees() closes every tree open on the connection, whichever
        # signal opened it, so two trees on one server are still one close.
        a = MdsSignal("a", "efit01", location="remote://atlas.gat.com")
        b = MdsSignal("b", "d3d", location="remote://atlas.gat.com")
        self.assertEqual(a.cleanup_shot_key(), b.cleanup_shot_key())

    def test_different_servers_do_not_share_a_key(self):
        a = MdsSignal("a", "efit01", location="remote://atlas.gat.com")
        b = MdsSignal("b", "efit01", location="remote://other.gat.com")
        self.assertNotEqual(a.cleanup_shot_key(), b.cleanup_shot_key())

    def test_local_signals_share_a_key_by_treename(self):
        a = MdsSignal("a", "efit01", location="/some/path")
        b = MdsSignal("b", "efit01", location="/some/path")
        self.assertEqual(a.cleanup_shot_key(), b.cleanup_shot_key())

    def test_local_signals_on_different_trees_do_not_share_a_key(self):
        # Local trees are closed per treename, so these are two closes.
        a = MdsSignal("a", "efit01", location="/some/path")
        b = MdsSignal("b", "d3d", location="/some/path")
        self.assertNotEqual(a.cleanup_shot_key(), b.cleanup_shot_key())

    def test_local_and_remote_keys_do_not_collide(self):
        local = MdsSignal("a", "efit01", location="/some/path")
        remote = MdsSignal("a", "efit01", location="remote://efit01")
        self.assertNotEqual(local.cleanup_shot_key(), remote.cleanup_shot_key())

    def test_mds_signal_delegates_to_the_underlying_signal(self):
        sig = MdsSignal("a", "efit01", location="remote://atlas.gat.com")
        self.assertEqual(sig.cleanup_shot_key(), sig.sig.cleanup_shot_key())


class _Value:
    """Stand-in for an MDSplus Data object: both gather paths unwrap .value."""

    def __init__(self, value):
        self.value = value


class _FakeMdsError(Exception):
    """Stand-in for a typed MDSplus exception, which only the serial path raises."""


class _FakeGetMany:
    def __init__(self, connection):
        self.connection = connection
        self.items = []
        self.result = None

    def append(self, name, expression):
        self.items.append((name, expression))

    def execute(self):
        self.connection.executes += 1
        if self.connection.execute_raises:
            raise RuntimeError("server has no GetManyExecute")
        self.result = {}
        for name, expression in self.items:
            if expression in self.connection.fail:
                # How MDSplus reports a failed entry: no type, no useful text.
                self.result[name] = {
                    "error": "%MDSPLUS-E-Unknown, Unknown exception"
                }
            else:
                self.result[name] = {
                    "value": _Value(self.connection.values[expression])
                }
        return self.result

    def get(self, name):
        entry = self.result[name]
        if "value" not in entry:
            raise _FakeMdsError(name)
        return entry["value"]


class _FakeConnection:
    """Counts round trips so a test can tell one batched call from several."""

    def __init__(self, values, fail=(), execute_raises=False):
        self.values = values
        self.fail = set(fail)
        self.execute_raises = execute_raises
        self.gets = []
        self.executes = 0
        self.opened = []
        self.closes = 0
        self.disconnected = False

    def openTree(self, treename, shot):
        self.opened.append((treename, shot))

    def closeAllTrees(self):
        self.closes += 1

    def disconnect(self):
        self.disconnected = True

    def get(self, expression):
        self.gets.append(expression)
        if expression in self.fail:
            raise _FakeMdsError(expression)
        return _Value(self.values[expression])

    def getMany(self):
        return _FakeGetMany(self)


class TestMdsRemoteBatchedGather(unittest.TestCase):
    """Data, dims and units are independent expressions against an already-open
    tree, so fetching them one at a time costs a round trip each for nothing."""

    EXPRESSION = r"\ipmhd"

    def _signal(self, fetch_units=True, dims=("times",)):
        return MdsRemoteSignal(
            self.EXPRESSION, "efit01", "fake.host:9999",
            dims=dims, fetch_units=fetch_units,
        )

    def tearDown(self):
        MdsConnectionRegistry()._connection_map.pop("fake.host:9999", None)

    def _connection(self, sig, **kwargs):
        values = {expr: f"value-of-{expr}" for _, _, expr in sig._gather_plan()}
        connection = _FakeConnection(values, **kwargs)
        # _do_gather asks the registry for its connection, so that is where a
        # stand-in has to go.
        MdsConnectionRegistry()._connection_map[sig.server] = connection
        return connection

    def test_plan_covers_data_then_dims_then_units(self):
        sig = self._signal(dims=("times", "space"))
        slots = [(slot, name) for slot, name, _ in sig._gather_plan()]
        self.assertEqual(
            slots,
            [("data", None), ("dim", "times"), ("dim", "space"),
             ("units", "data"), ("units", "times"), ("units", "space")],
        )

    def test_plan_omits_units_when_not_requested(self):
        sig = self._signal(fetch_units=False)
        slots = [(slot, name) for slot, name, _ in sig._gather_plan()]
        self.assertEqual(slots, [("data", None), ("dim", "times")])

    def test_batched_gather_is_a_single_round_trip(self):
        sig = self._signal()
        connection = self._connection(sig)

        sig._do_gather(1234)

        self.assertEqual(connection.executes, 1)
        self.assertEqual(connection.gets, [])

    def test_serial_gather_is_a_round_trip_per_expression(self):
        sig = self._signal()
        connection = self._connection(sig)
        sig._use_getmany = False

        sig._do_gather(1234)

        self.assertEqual(connection.executes, 0)
        self.assertEqual(len(connection.gets), 4)

    def test_batched_and_serial_agree(self):
        for fetch_units in (True, False):
            with self.subTest(fetch_units=fetch_units):
                sig = self._signal(fetch_units=fetch_units)
                self._connection(sig)
                batched = sig._do_gather(1234)
                sig._use_getmany = False
                serial = sig._do_gather(1234)
                self.assertEqual(batched, serial)

    def test_failed_entry_falls_back_so_the_real_error_surfaces(self):
        # A failed entry carries no exception type and no readable message, and
        # gather() keys its retry on the type, so the fetch is redone one at a
        # time to raise the real thing.
        sig = self._signal()
        connection = self._connection(sig, fail=[self.EXPRESSION])

        with self.assertRaises(_FakeMdsError):
            sig._do_gather(1234)

        self.assertEqual(connection.executes, 1)
        self.assertIn(self.EXPRESSION, connection.gets)

    def test_server_without_getmany_is_not_asked_twice(self):
        sig = self._signal()
        connection = self._connection(sig, execute_raises=True)

        sig._do_gather(1234)
        self.assertEqual(connection.executes, 1)
        self.assertFalse(sig._use_getmany)
        self.assertEqual(len(connection.gets), 4)

        sig._do_gather(1234)
        self.assertEqual(connection.executes, 1)
        self.assertEqual(len(connection.gets), 8)


class TestMdsRemoteOpenTreeDedup(unittest.TestCase):
    """openTree only sets the connection's current tree, and signals sharing a
    server share one connection -- so opening per signal costs a round trip
    per signal for a tree the previous one just opened.
    """

    SERVER = "fake.host:9999"

    def tearDown(self):
        MdsConnectionRegistry()._connection_map.pop(self.SERVER, None)

    def _signals(self, expressions, treename="efit01"):
        return [
            MdsRemoteSignal(expression, treename, self.SERVER,
                            dims=("times",), fetch_units=False)
            for expression in expressions
        ]

    def _install(self, sigs):
        values = {
            expression: f"value-of-{expression}"
            for sig in sigs
            for _, _, expression in sig._gather_plan()
        }
        connection = _FakeConnection(values)
        MdsConnectionRegistry()._connection_map[self.SERVER] = connection
        return connection

    def test_signals_sharing_a_tree_open_it_once(self):
        sigs = self._signals([r"\a", r"\b", r"\c"])
        connection = self._install(sigs)

        for sig in sigs:
            sig._do_gather(1234)

        self.assertEqual(connection.opened, [("efit01", 1234)])

    def test_a_new_shot_reopens(self):
        sigs = self._signals([r"\a", r"\b"])
        connection = self._install(sigs)

        for shot in (1234, 1235):
            for sig in sigs:
                sig._do_gather(shot)

        self.assertEqual(connection.opened,
                         [("efit01", 1234), ("efit01", 1235)])

    def test_a_different_tree_reopens(self):
        first = self._signals([r"\a"], treename="efit01")[0]
        second = self._signals([r"\b"], treename="d3d")[0]
        connection = self._install([first, second])

        first._do_gather(1234)
        second._do_gather(1234)
        first._do_gather(1234)

        self.assertEqual(
            connection.opened,
            [("efit01", 1234), ("d3d", 1234), ("efit01", 1234)],
        )

    def test_cleanup_shot_releases_nothing(self):
        # A remote tree is not a per-shot resource -- the next shot's open
        # replaces it -- so closing per record spent a round trip for nothing.
        sig = self._signals([r"\a"])[0]
        connection = self._install([sig])

        sig._do_gather(1234)
        sig.cleanup_shot(1234)
        sig._do_gather(1234)

        self.assertEqual(connection.closes, 0)
        self.assertEqual(connection.opened, [("efit01", 1234)])

    def test_cleanup_closes_the_trees_and_disconnects(self):
        sig = self._signals([r"\a"])[0]
        connection = self._install([sig])

        sig._do_gather(1234)
        sig.cleanup()

        self.assertEqual(connection.closes, 1)
        self.assertTrue(connection.disconnected)

    def test_cleanup_does_not_dial_a_connection_to_close_trees(self):
        # cleanup() on a signal that never fetched must not open a connection
        # purely so that it has something to close.
        sig = self._signals([r"\a"])[0]
        MdsConnectionRegistry()._connection_map.pop(self.SERVER, None)

        sig.cleanup()

        self.assertNotIn(self.SERVER, MdsConnectionRegistry()._connection_map)

    def test_a_failed_fetch_does_not_leave_the_tree_assumed_open(self):
        # If a fetch fails the tree may be gone, so the next signal must not
        # skip its open on the strength of the failed one.
        sigs = self._signals([r"\a", r"\b"])
        connection = self._install(sigs)
        connection.fail.add(r"\a")

        with self.assertRaises(_FakeMdsError):
            sigs[0]._do_gather(1234)
        sigs[1]._do_gather(1234)

        self.assertEqual(connection.opened,
                         [("efit01", 1234), ("efit01", 1234)])
