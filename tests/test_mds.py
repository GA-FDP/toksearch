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
        server = "fake.gat.com"
        default_location = f"remote://{server}"

        with set_env("TOKSEARCH_MDS_DEFAULT", default_location):
            sig = MdsSignal("blah", "efit01")
        self.assertIsInstance(sig.sig, MdsLocalSignal)
        self.assertIsInstance(sig.sig.treepath, MdsTreePath)
        self.assertEqual(sig.sig.treepath.paths["efit01"], default_location)

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
