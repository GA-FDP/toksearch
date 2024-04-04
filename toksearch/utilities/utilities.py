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

from collections import OrderedDict
import contextlib
import os
import traceback


@contextlib.contextmanager
def set_env(var, val):
    """Context mananger to temporarily set environment variable"""
    old_environ = dict(os.environ)
    os.environ[var] = val
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@contextlib.contextmanager
def unset_env(var):
    """Context mananger to temporarily set environment variable"""
    old_environ = dict(os.environ)
    os.environ.pop(var, None)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


def chunk_it(seq, num):
    """Given a sequence, return a list of num sequences"""
    avg = len(seq) / float(num)
    return partition_it(seq, avg)


def partition_it(seq, part_size):
    """
    Given a sequence, return a list of sequences each with
    length close to part_size
    """
    avg = part_size
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last) : int(last + avg)])
        last += avg

    return out


def capture_exception(label, e):
    """Call this in an except block to get a nice dict that
    is serializable with fields:
    label: Some label provided by the caller.
    type: A string with the class of the exception
    traceback: A string of the call stack
    """
    traceback_string = traceback.format_exc()
    captured_error = {
        "label": label,
        "type": str(e.__class__),
        "traceback": traceback_string,
    }
    return captured_error
