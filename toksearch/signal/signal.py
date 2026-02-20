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

import gc
import os
import inspect
import contextlib
import xarray as xr
import numpy as np
from typing import Any, Callable
import uuid
import warnings
from abc import ABC, abstractmethod


@contextlib.contextmanager
def _gc_disabled():
    """Context manager that disables the GC for the duration of the block.

    Prevents GC-triggered __del__ from firing mid-way through C library calls
    (e.g. MDSplus _TreeClose), which can cause re-entrant heap corruption.
    GC is restored to its prior state on exit.
    """
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


################################################################################
# Signal Abstract Base Class
################################################################################


class Signal(ABC):
    """Abstract base class for signals

    This class is intended to be subclassed to create signals that can be
    fetched from a data source. The class provides a fetch method that
    fetches the data for a given shot, and a fetch_as_xarray method that
    fetches the data as an xarray Dataset object.

    The general idea is that a signal is created with a set of parameters
    that are used to fetch the data for a given shot. The signal is then registered with a
    SignalRegistry object, which keeps track of all signals used in a toksearch
    application. This allows for easy cleanup of all signals (and associated
    resources) when the application is done.

    The fetch method is the main method that needs to be called to fetch the
    data for a shot. It first performs registration of the signal with the 
    SignalRegistry object. It then calls the gather method, which is an abstract
    method that needs to be implemented by subclasses. The gather method is
    responsible for collecting the data for a shot, and should return a dictionary
    containing the data fetched for the signal. The dictionary should contain a
    key 'data' with the data, and keys for each dimension of the data, with the
    values being the values of the dimensions. If the with_units attribute is True,
    the dictionary should also contain a key 'units' with the units of the data and
    dimensions.If a callback function has been set with the set_callback method,
    it is called after the data is fetched.

    Methods:
        fetch: Fetch the data for a shot.

        fetch_as_xarray: Fetch the data for a shot as an xarray DataArray object.

        gather: Abstract method. Collect the data for a shot. This method should be implemented
            by subclasses.

        cleanup_shot: Abstract method. Clean up any resources specific to a shot. For example, if
            an MDSplus tree is opened to fetch data, this method should close the tree.

        cleanup: Abstract method. Clean up any resources shared between shots. For example, if a
            network connection is opened to fetch data (and shared amongst multiple shots),
            this method should close it.

        set_callback: set a callback function to be called after the data is gathered in the fetch method

        set_dims: set the dimensions of the signal

    Note:


        gather, cleanup, and cleanup_shot are abstract methods that need to be implemented
        by subclasses.

    Attributes:
        dims: A list or other iterable of the labels
            for each dimension of the signals data. Most typically, this
            is just time, so the default is ('times',). If, for example,
            you have a time varying profile with, say, a radial dimension,
            you could pass ('times', 'radius'). The order is significant
            since, and in the case of MDSplus, it needs to match the way
            dim_of(0), dim_of(1),... are stored.


        data_order: A list or other iterable of the labels for each dimension of
            the signals data. This is to be used when the dimension order fetched in
            MDSplus, dim(0),dim(1)... does not match the shape of the data stored.
            This list should match the order that the dimensions are stored in the actual data,
            and should be complementary with the dims parameter.

            Ex) If in MDSplus, dim(0) = 'rho', dim(1) = 'times', then the following would be used:
                dims = ('rho','times')

            But, if the data shape (n_times, n_rho), then the following would be used to
            match the data shape:
                data_order = (1,0) or ("times","rho")

        with_units (bool): A boolean flag that indicates whether to fetch the units
            of the signal. If True, the fetch_units method will be called to fetch the
            units of the signal in the fetch method.
    """

    def __init__(self):
        self._callback = None
        self._state = {}
        self.dims: Iterable[str] = ("times",)
        self.data_order: Iterable[str] = self.dims
        self.with_units: bool = True

    def set_callback(self, func) -> "Signal":
        """Set a callback function to be called after the data is fetched in the fetch method

        Arguments:
            func (function): The callback function to call. The function should take a single
                argument, which is a dictionary containing the data fetched for the signal. The
                function should return a dictionary containing the modified data.

        Returns:
            Signal: The signal object. This allows for chaining of method calls.
                eg signal.set_callback(func).set_dims(dims).fetch(shot)
        """
        self._callback = func
        return self

    def set_dims(self, dims, data_order=None) -> "Signal":
        """Set the dimensions of the signal

        This method sets the dimensions of the signal. The dimensions are used
        to fetch the dimensions of the signal in the fetch_dims method. The
        dimensions are also used to create an xarray Dataset object in the
        fetch_as_xarray method.


        Arguments:
            dims (iterable of strings): A list or other iterable of the labels
                for each dimension of the signals data. Most typically, this
                is just time, so the default is ('times',). If, for example,
                you have a time varying profile with, say, a radial dimension,
                you could pass ('times', 'radius'). The order is significant
                since, and in the case of MDSplus, it needs to match the way
                dim_of(0), dim_of(1),... are stored.

        Keyword Arguments:
            data_order (list): A list or other iterable of the labels for each dimension of
                the signals data. This is to be used when the dimension order fetched in
                MDSplus, dim(0),dim(1)... does not match the shape of the data stored.
                This list should match the order that the dimensions are stored in the actual data,
                and should be complementary of the dims parameter.

                Ex) MDSplus storage -> dim(0) = 'rho', dim(1) = 'time'
                    dims = ('rho','time')

                Data shape -> (time X rho)
                    data_order = (1,0) or ("time","rho")

        Returns:
            Signal: The signal object. This allows for chaining of method calls.
                eg signal.set_callback(func).set_dims(dims).fetch(shot)
        """
        try:
            self.dims = dims
            self.data_order = (
                self.dims
                if data_order is None
                else [
                    self.dims[dim] if isinstance(dim, int) else dim
                    for dim in data_order
                ]
            )
        except IndexError as exc:
            msg = (
                "Error: index used in data_order must refer to index in dims parameter. This caused "
                + exc
            )
            raise Exception(msg)

        return self


    def fetch(self, shot: int) -> dict:
        """Fetch the data for a shot

        

        Arguments:
            shot (int): The shot number to fetch the data for

        Returns:
            dict: A dictionary containing the data fetched for the signal. The dictionary
                will contain a key 'data' with the data, and keys for each dimension of the
                data, with the values being the values of the dimensions. If the with_units
                attribute is True, the dictionary will also contain a key 'units' with the units
                of the data and dimensions.
        """

        SignalRegistry().register(self)

        with _gc_disabled():
            results = self.gather(shot)


        if results and (self._callback is not None):
            results = self._callback(results)

        return results

    @abstractmethod
    def gather(self, shot: int) -> dict:
        """Collect the data for a shot"""
        pass

    def fetch_as_xarray(self, shot: int) -> xr.DataArray:
        """Fetch the data for a shot as an xarray DataArray object

        Returns a DataArray object with dimensions specified in the
        dims attribute of the Signal object.

        Arguments:
            shot (int): The shot number to fetch the data for

        Returns:
            xr.DataArray: An xarray DataArray object containing the data fetched for the signal,
                with dimensions specified in the dims attribute of the Signal object.
        """

        signal_as_dict = self.fetch(shot)
        d = signal_as_dict["data"]
        coords = {}
        units = signal_as_dict.get("units", {})

        for dim in self.data_order:
            if dim is None:
                msg = "Value stored in data_order paramete must be an int or str, cannot be None"
                raise Exception(msg)
            coords[dim] = signal_as_dict[dim]
        results = xr.DataArray(d, dims=self.data_order, coords=coords)
        for unit in units:
            if unit == "data":
                results.attrs = {
                    "units": units[unit]
                }  # different syntax for non-dimension data
            else:
                results[unit].attrs = {"units": units[unit]}

        return results


    @abstractmethod
    def cleanup_shot(self, shot):
        """Close down any per-shot resources needed to fetch data a shot

        For instance, if an mds datasource is being used, this will close
        the tree.
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Close down any resources that are shared amongst multiple shots.

        This typically means closing a network connection to a remote server.
        """
        pass


################################################################################
# SignalRegistry Class
################################################################################

class SignalRegistry:
    """Class to keep track of all signals used in a toksearch application

    This allows for easy cleanup of all signals (and associated resources).

    The class is implemented as a singleton per process, so that all signals are
    registered to the same instance, regardless of where they are
    instantiated in the code.

    This class will generally not be used directly, but will be used by the
    Signal class to register signals.
    """

    _instance = None
    _pid = None

    def __new__(cls) -> "SignalRegistry":
        """Create a new instance of the class if one does not already exist for this process

        Returns:
            SignalRegistry: The instance of the SignalRegistry class
        """
        # Get the current process ID
        current_pid = os.getpid()
        if cls._instance is None or cls._pid != current_pid:
            cls._instance = super(SignalRegistry, cls).__new__(cls)
            cls._pid = current_pid
            cls._instance.signals = set()  # set of Signal objects
        return cls._instance

    def register(self, signal: Signal):
        """Register a signal with the registry

        Arguments:
            signal (Signal): The signal to register
        """
        self.signals.add(signal)

    def cleanup(self):
        """Clean up all resources shared between shots for all registered signals"""
        for signal in self.signals:
            try:
                signal.cleanup()
            except Exception as e:
                print(f"Warning: failed to cleanup signal {signal}: {e}")
        self.reset()

    def cleanup_shot(self, shot: int):
        """Clean up all resources specific to a shot for all registered signals

        Arguments:
            shot (int): The shot number to clean up resources for
        """

        with _gc_disabled():
            for signal in self.signals:
                try:
                    signal.cleanup_shot(shot)
                except Exception as e:
                    print(
                        f"Warning: failed to cleanup signal {signal} for shot {shot}: {e}"
                    )

    def reset(self):
        """Reset the registry to an empty state"""
        self.signals = set()

    def __contains__(self, signal):
        return signal in self.signals
