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


"""This module provides classes for fetching data from MDSplus trees

The main class for user applications is the MdsSignal class. This class
provides a way to fetch data from MDSplus trees either on a local disk or
from a remote server. The MdsSignal class is a subclass of the Signal class
and provides the same interface as the Signal class.

The MdsSignal class is a wrapper around the MdsLocalSignal and MdsRemoteSignal
classes. The MdsLocalSignal class is used to fetch data from a local disk and
the MdsRemoteSignal class is used to fetch data from a remote server. The
MdsSignal class determines which class to use based on the location argument
provided to the constructor.

A location is read as a server when it carries a URL scheme, and as a tree path
otherwise. 'remote://host' names a server reached over MDSplus's default
transport, while a scheme in MDSIP_TRANSPORT_SCHEMES -- notably 'fdp://', which
reaches an FDP Pelican origin over HTTPS -- names the transport itself and is
passed to MDSplus.Connection whole.

Behind the scenes, the MdsLocalSignal uses the MdsTreeRegistry class to manage
MDSplus trees on a local disk. The MdsRemoteSignal uses the MdsConnectionRegistry
class to manage connections to remote servers.

The MdsTreePath class is used to set environment variables for MDSplus trees
on a local disk. The MdsTreePath class is used by the MdsLocalSignal class to
set the environment variables for the MDSplus trees.
"""

import logging

import MDSplus as mds
from MDSplus.mdsExceptions import MDSplusERROR
import pdb
import os
import contextlib
from urllib.parse import urlparse
import numpy as np
import psutil
from typing import Union, Iterable, Optional

from .signal import Signal
from ..utilities.utilities import set_env

_log = logging.getLogger(__name__)


# URL schemes that name an MDSplus *transport*, so a location using one is a
# connection string rather than a tree path.
#
# MDSplus picks the transport from the scheme itself: parse_host() splits
# <scheme>://<host>, and LoadIo() dlopens "libMdsIp" + SCHEME.upper() + ".so"
# and resolves the symbol Io. 'fdp' comes from the mdsip-fdp package
# (GA-FDP/xrdoss-mdsplus) and tunnels mdsip over a Pelican origin's HTTPS port;
# the rest ship with MDSplus.
#
# This is an allowlist rather than "any scheme" because LoadIo does not report
# an unknown scheme as an error -- it silently returns the ssh-tunnel routines.
# A typo would therefore try to ssh somewhere instead of failing, and a tree
# path that happens to look like a URL would be quietly misread as a server.
MDSIP_TRANSPORT_SCHEMES = frozenset(
    {"fdp", "tcp", "tcpv6", "udt", "udtv6", "gsi", "ssh"}
)


def _dim_of_expression(expression, dim=0):
    return "dim_of({}, {})".format(expression, dim)


def _units_of_expression(expression, dim=0):
    exp = "units({})"
    if dim == -1:
        return exp.format(expression)
    else:
        return exp.format(_dim_of_expression(expression, dim))


class MdsTreePath(object):
    def __init__(self, **paths):
        """Create an object to manage the paths to MDSplus trees

        Keyword Arguments:
            Accepts keyword arguments of the form
            treename=/some/mds/tree/path
        """
        self.paths = paths

    @contextlib.contextmanager
    def set_env(self):
        """Temporarily set mds treepath environment variables"""
        old_var_vals = {var: os.getenv(var, None) for var in self.paths.keys()}

        for var, val in self.paths.items():
            var_name = self.variable_name(var)
            old_var_vals[var_name] = os.getenv(var_name, None)

            os.environ[var_name] = val

        try:
            yield
        finally:
            for var, old_val in old_var_vals.items():
                var_name = self.variable_name(var)
                if old_val is None:
                    os.environ.pop(var_name, None)
                else:
                    os.environ[var_name] = old_val

    @classmethod
    def variable_name(cls, treename):
        """Return the name of the environment variable for the given treename"""
        return "{}_path".format(treename)


class MdsLocalSignal(Signal):
    def __init__(
        self, 
        expression: str,
        treename: str,
        treepath: Union[str, MdsTreePath] = None,
        dims: Iterable[str] = ("times",),
        data_order: Optional[Iterable[str]] = None,
        fetch_units: bool = True,
    ):
        """Create a signal object that fetches data from an MDSplus tree

        Arguments:
            expression: The tdi expression to fetch data from
            treename: The name of the tree to fetch from

            treepath: If not set, MDSplus will just use the
                environment variable of the form ${treename}_path. This
                kwarg can be either
                1) A string. In this case the environment variable
                  ${treename}_path is set to the value of the string
                  and this is used to locate the appropriate mdsplus files

                  or

                2) An MdsTreePath object

            dims: See documentation for the Signal class. Defaults to ('times',)
            data_order: See documentation for the Signal class. Defaults to the same
                as dims.
            fetch_units: See documentation for the Signal class. Defaults 
                to True.
        """
        super().__init__()

        self.expression = expression
        self.treename = treename
        self.treepath = treepath

        data_order = data_order or dims
        self.with_units = fetch_units

        self._shot_state = {}

        self.set_dims(dims, data_order)


    def gather(self, shot):
        """Gather the data for a shot
        
        Arguments:
            shot (int): The shot number to gather the data for

        Returns:
            dict: A dictionary containing the data gathered for the signal. The dictionary
                will contain a key 'data' with the data, and keys for each dimension of the
                data, with the values being the values of the dimensions. If the with_units
                attribute is True, the dictionary will also contain a key 'units' with the units
                of the data and dimensions.
        """
        results = {}

        tree = MdsTreeRegistry().open_tree(self.treename, shot, treepath=self.treepath)
        node = tree.getNode(self.expression)
        results["data"] = node.data()

        dims = self.dims

        dims_dict = {}
        if not dims:
            dims = []
        for i, dim in enumerate(dims):
            results[dim] = node.getDimensionAt(i).data()

        if self.with_units:
            units = {}
            units["data"] = str(node.getUnits().data())
            dims = self.dims
            if not dims:
                dims = []
            for i, dim in enumerate(dims):
                units[dim] = str(node.getDimensionAt(i).getUnits().data())

            results["units"] = units

        return results

    def cleanup_shot(self, shot: int):
        """Close the tree for this shot

        Arguments:
            shot (int): The shot number to close the tree for
        """
        MdsTreeRegistry().close_tree(self.treename, shot)

    def cleanup_shot_key(self):
        """Trees are held per treename, so one close per treename is enough."""
        return ("mds_local_tree", self.treename)

    def cleanup(self):
        """Close all trees"""
        MdsTreeRegistry().close_all_trees()


class MdsSignal(Signal):

    def __init__(
        self,
        expression: str,
        treename: str,
        location: Optional[Union[str, MdsTreePath]] = None,
        dims: Iterable[str] = ("times",),
        data_order: Optional[Iterable[str]] = None,
        fetch_units: bool = True,
    ):
        """Create a signal object that fetches data from an MDSplus tree

        Arguments:
            expression: The tdi expression to fetch data from
            treename: The name of the tree to fetch from
            location: The location of the tree.

                - If None, check if the environment variable TOKSEARCH_MDS_DEFAULT is
                set and use it, otherwise assume that the tree is on a local disk
                and that the treepath is available in the environment.

                - If a simple path is given
                (e.g. /some/path), then that will be used for the treepath. You can
                also specify a remote server by specifying the location as
                'remote://some.server'

                - If a URL with an MDSplus transport scheme is given, it is used
                as a connection string in full. In particular 'fdp://' reaches an
                FDP Pelican origin over HTTPS, e.g.
                'fdp://fdp-d3d-origin.nationalresearchplatform.org:8443/mdsip',
                where the path is the relay's prefix on that origin. This
                requires the mdsip-fdp package, which supplies the transport
                MDSplus loads for the scheme. See MDSIP_TRANSPORT_SCHEMES.

                - If an MdsTreePath object is provided, then the signal data is
                fetched from a local disk according to the path specifications in
                the MdsTreePath object.

                A location carrying a scheme that is none of the above raises
                ValueError rather than being read as a tree path, since dropping
                the host silently would otherwise turn a mistyped server into a
                local read.
            dims: See documentation for the Signal class. Defaults to ('times',)
            data_order: See documentation for the Signal class. Defaults to the same
                as dims.
            fetch_units: See documentation for the Signal class. Defaults 
                to True.
        """
        super().__init__()

        self.location = location
        self.sig = self.create_local_or_remote_signal(
            expression, treename, location, dims=dims, data_order=data_order, fetch_units=fetch_units
        )
        self.dims = self.sig.dims
        self.data_order = self.sig.data_order
        self.with_units = self.sig.with_units

    @classmethod
    def create_local_or_remote_signal(cls, expression, treename, location, **kwargs):
        """Create either an MdsLocalSignal or MdsRemoteSignal object based on the location

        See the docs for the MdsSignal class for more information on the arguments
        """

        if location is None:
            path_kwargs = {}

            tree_var = MdsTreePath.variable_name(treename)

            if (os.getenv(tree_var, None) is None) and (os.getenv("default_tree_path", None) is None):
                print(f"Warning, neither {tree_var} or default_tree_path are set")


                default_location = os.getenv("TOKSEARCH_MDS_DEFAULT", None)
                if default_location:
                    if "://" in default_location:
                        # A connection string, not a tree path. Dispatch it the
                        # same way an explicit location= would be, so that
                        # pointing a whole workflow at a server through the
                        # environment behaves like pointing one signal at it.
                        return cls.create_local_or_remote_signal(
                            expression, treename, default_location, **kwargs
                        )
                    path_kwargs = {treename: default_location}

            location = MdsTreePath(**path_kwargs)


        if isinstance(location, MdsTreePath):
            return MdsLocalSignal(expression, treename, treepath=location, **kwargs)

        # Anything without '://' is a tree path, not a URL, and is passed
        # through untouched. That covers plain directories and MDSplus's own
        # 'host::' / 'host::/path' syntax -- which must be settled before
        # urlparse gets involved, because it reads 'abc::' as scheme 'abc' and
        # 'atlas.gat.com::/trees' as scheme 'atlas.gat.com' (dots are legal in
        # a scheme), either of which would otherwise be mistaken for a server.
        if "://" not in location:
            return MdsLocalSignal(
                expression, treename, treepath=(location or None), **kwargs
            )

        parsed_location = urlparse(location)
        scheme = parsed_location.scheme.lower()

        if scheme == "remote":
            # toksearch's own marker, not an MDSplus scheme: connect to this
            # server over whatever transport MDSplus defaults to. The scheme is
            # dropped and only the host is passed on.
            return MdsRemoteSignal(
                expression, treename, parsed_location.netloc, **kwargs
            )

        if scheme in MDSIP_TRANSPORT_SCHEMES:
            # An MDSplus connection string. The WHOLE url goes to
            # MDSplus.Connection: the scheme selects the transport, and the path
            # is meaningful to it -- for fdp:// it is the relay's prefix on the
            # origin, e.g. fdp://origin.example.org:8443/mdsip -- so neither the
            # scheme nor the path may be dropped the way 'remote://' drops them.
            return MdsRemoteSignal(expression, treename, location, **kwargs)

        raise ValueError(
            f"Unrecognized scheme {scheme!r} in MdsSignal location {location!r}. "
            f"Use 'remote://<server>' for a plain mdsip server, one of "
            f"{sorted(MDSIP_TRANSPORT_SCHEMES)} for a specific MDSplus "
            f"transport, or a path with no scheme for a tree path."
        )


    def gather(self, shot):
        """Gather the data for a shot
        
        Arguments:
            shot (int): The shot number to gather the data for

        Returns:
            dict: A dictionary containing the data gathered for the signal. The dictionary
                will contain a key 'data' with the data, and keys for each dimension of the
                data, with the values being the values of the dimensions. If the with_units
                attribute is True, the dictionary will also contain a key 'units' with the units
                of the data and dimensions.
        """
        return self.sig.gather(shot)


    def cleanup_shot(self, shot: int):
        """Close the tree for this shot

        Arguments:
            shot (int): The shot number to close the tree for
        """
        self.sig.cleanup_shot(shot)

    def cleanup_shot_key(self):
        """Defer to the local or remote signal actually doing the cleanup."""
        return self.sig.cleanup_shot_key()

    def cleanup(self):
        """Close all trees or disconnect from the remote server"""
        self.sig.cleanup()


class MdsConnectionRegistry(object):
    __instance = None

    def __new__(cls):
        if MdsConnectionRegistry.__instance is None:
            MdsConnectionRegistry.__instance = object.__new__(cls)
            MdsConnectionRegistry.__instance._connection_map = {}
        return MdsConnectionRegistry.__instance

    def __getstate__(self):
        _dict = self.__dict__
        _dict["_connection_map"] = {}
        return _dict

    def connect(self, server):
        conn = self._connection_map.get(server, None)
        if conn is None:
            conn = mds.Connection(server)
            self._connection_map[server] = conn
        return conn

    def disconnect(self, server):
        """Drop the cached connection for a server and close it.

        After this call, the next :meth:`connect` for the same server
        creates a fresh :class:`MDSplus.Connection`.
        """
        conn = self._connection_map.pop(server, None)
        if conn is not None:
            try:
                conn.disconnect()
            except Exception:
                pass


class MdsRemoteSignal(Signal):
    def __init__(
        self,
        expression: str,
        treename: str,
        server: str,
        dims: Iterable[str] = ("times",),
        data_order: Optional[Iterable[str]] = None,
        fetch_units: bool = True,
    ):
        """Create a signal object that fetches data from a remote MDSplus tree

        Arguments:
            expression: The tdi expression to fetch data from
            treename: The name of the tree to fetch from
            server: What to hand MDSplus.Connection. Either a bare host
                (e.g. atlas.gat.com), which uses MDSplus's default transport, or
                a full connection URL whose scheme selects one
                (e.g. fdp://origin.example.org:8443/mdsip).
            dims: See documentation for the Signal class. Defaults to ('times',)
            data_order: See documentation for the Signal class. Defaults to the same
                as dims.
            fetch_units: See documentation for the Signal class. Defaults 
                to True.
        """
        super().__init__()

        self.expression = expression
        self.treename = treename
        self.server = server

        data_order = data_order or dims
        self.with_units = fetch_units

        self._shot_state = {}

        self.set_dims(dims, data_order)

    def connect(self) -> mds.Connection:
        """Open the connection to remote server"""
        return MdsConnectionRegistry().connect(self.server)


    def gather(self, shot):
        """Gather the data for a shot, with one retry on MDSplusERROR.

        The mdsip server can leave per-connection state wedged after returning
        the generic ``MDSplusERROR`` (``%MDSPLUS-E-ERROR``) for a query, so
        subsequent operations on the same connection -- including for unrelated
        shots -- can also return errors. To recover, drop the cached connection
        and retry once on a fresh one.

        Tree-class errors (``TreeNODATA``, ``TreeFOPENR``, ``TreeNOCURRENT``,
        ...) follow a different server-side path and don't corrupt connection
        state, so they are propagated unchanged.

        Arguments:
            shot (int): The shot number to gather the data for

        Returns:
            dict: A dictionary containing the data gathered for the signal. The dictionary
                will contain a key 'data' with the data, and keys for each dimension of the
                data, with the values being the values of the dimensions. If the with_units
                attribute is True, the dictionary will also contain a key 'units' with the units
                of the data and dimensions.
        """
        try:
            return self._do_gather(shot)
        except MDSplusERROR as e:
            _log.warning(
                "MDSplusERROR on %s for shot=%s expr=%r (%s); "
                "dropping connection and retrying once",
                self.server, shot, self.expression, e,
            )
            MdsConnectionRegistry().disconnect(self.server)
            return self._do_gather(shot)

    def _do_gather(self, shot):
        connection = self.connect()
        connection.openTree(self.treename, shot)

        results = {}
        results["data"] = connection.get(self.expression).value

        dims = self.dims
        dims_dict = {}
        if not dims:
            dims = []
        for i, dim in enumerate(dims):
            dim_expression = _dim_of_expression(self.expression, dim=i)
            results[dim] = connection.get(dim_expression).value


        if self.with_units:
            units = {}
            units["data"] = connection.get(
                _units_of_expression(self.expression, dim=-1)
            ).value
            dims = self.dims
            if not dims:
                dims = []
            for i, dim in enumerate(dims):
                units_expression = _units_of_expression(self.expression, dim=i)
                units[dim] = connection.get(units_expression).value

            results["units"] = units

        return results


    def cleanup_shot(self, shot):
        """Close all trees for the given shot

        Arguments:
            shot (int): The shot number to close the tree for
        """
        try:
            connection = self.connect()
            connection.closeAllTrees()
        except:
            pass

    def cleanup_shot_key(self):
        """Signals sharing a server share a connection -- and one round trip.

        MdsConnectionRegistry hands out one connection per server, and
        closeAllTrees() closes every tree open on it regardless of which
        signal opened it, so the treename is deliberately not part of the key.
        """
        return ("mds_remote_connection", self.server)

    def cleanup(self):
        """Disconnect from the remote server"""
        try:
            MdsConnectionRegistry().disconnect(self.server)
        except:
            pass


class MdsTreeRegistry(object):
    __instance = None

    def __new__(cls):
        if MdsTreeRegistry.__instance is None:
            MdsTreeRegistry.__instance = object.__new__(cls)
            MdsTreeRegistry.__instance._tree_map = {}
        return MdsTreeRegistry.__instance

    def __getstate__(self):
        _dict = self.__dict__
        _dict["_tree_map"] = {}
        return _dict

    def open_tree(self, treename, shot, treepath=None):
        tree = self._get_tree(treename, shot)
        if tree is None:

            if not treepath:
                treepath = MdsTreePath()
            elif isinstance(treepath, MdsTreePath):
                treepath = treepath
            else:
                treepath = MdsTreePath(**{treename: treepath})

            tree = self._open_tree(treename, shot, treepath)

            if treename not in self._tree_map:
                self._tree_map[treename] = {}

            self._tree_map[treename][shot] = tree

        return tree

    def reset(self):
        self._tree_map = {}

    def _open_tree(self, treename, shot, treepath):
        """treepath must be an MdsTreePathObject"""
        with treepath.set_env():
            tree = mds.Tree(treename, shot, mode="READONLY")
            return tree

    def _get_tree(self, treename, shot):
        return self._tree_map.get(treename, {}).get(shot, None)

    def close_tree(self, treename, shot):
        tree = self._tree_map.get(treename, {}).pop(shot, None)
        if tree is not None:
            try:
                tree.close()
            except Exception:
                pass

    def close_all_trees(self):
        for treename, shots_dict in list(self._tree_map.items()):
            for shot in list(shots_dict.keys()):
                self.close_tree(treename, shot)


