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
import pdb
import sys
import os
import numpy as np
import inspect
import types

import functools

import MDSplus as mds

try:
    import unittest.mock as mock
except:
    import mock


################################################################################
# Decide if we want to do integration testing (or, conversely, unit testing
# with mocked calls to external data sources
################################################################################
def do_integration_testing():
    if os.environ.get("TOKSEARCH_INTEGRATION", "No").lower() == "yes":
        return True
    else:
        return False


INTEGRATION = do_integration_testing()


def print_mock_info(test_name):
    if INTEGRATION:
        _data_type = "REAL"
    else:
        _data_type = "MOCKED"
    print("Running {} with {} data".format(test_name, _data_type))


def assert_mock_call_count_equal(mock_object, call_count):
    assert mock_object.call_count == call_count


def mockify_testcase(cls):
    test_method_names = [
        member[0]
        for member in inspect.getmembers(cls, predicate=inspect.ismethod)
        if member[0].startswith("test")
    ]
    for test_method_name in test_method_names:
        original_method = getattr(cls, test_method_name)

        def wrap_func(original_func):
            def mock_func(self, *mock_args):
                original_func(self)

            return mock_func

        setattr(cls, test_method_name, wrap_func(original_method))


# Preserve the connection and tree types so we can use it after
# mds.Connection has been mocked
mds_connection_type = mds.Connection
mds_tree_type = mds.Tree
