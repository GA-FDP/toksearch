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
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import toksearch.backend.multiprocessing as mp_backend
from toksearch.backend.multiprocessing import _Mapper
from toksearch.pipeline.pipeline_funcs import _SafeFetch
from toksearch.record import Record
from toksearch.signal.mock_signal import MockSignal
from toksearch.signal.signal import SignalRegistry


class _CountingSignal(MockSignal):
    """MockSignal that tallies cleanup_shot calls across every instance.

    The tally is class-level on purpose: the behavior under test is what
    happens as *distinct* Signal objects pile up, which is what a worker sees
    when joblib unpickles the operations afresh for each batch.
    """

    calls = 0

    def cleanup_shot(self, shot):
        _CountingSignal.calls += 1


class TestMapperRegistryScoping(unittest.TestCase):
    """The registry must not accumulate signals from batch to batch.

    Each iteration below builds a new signal and a new _Mapper over it, which
    is what unpickling a dispatched batch produces in a worker process. Before
    the registry was scoped to the record, these piled up for the life of the
    worker and SignalRegistry.cleanup_shot() swept all of them once per
    record -- quadratic work, and quadratic network round trips for signals
    whose cleanup_shot() talks to a server.
    """

    def setUp(self):
        SignalRegistry().reset()
        _CountingSignal.calls = 0

    def tearDown(self):
        SignalRegistry().reset()

    def test_registry_does_not_grow_across_batches(self):
        reg = SignalRegistry()
        sizes = []

        for _ in range(5):
            operations = [_SafeFetch("sig", MockSignal())]
            mapper = _Mapper(operations)
            for shot in range(3):
                mapper(Record(shot))
            sizes.append(len(reg.signals))

        self.assertEqual(sizes, [1] * 5)

    def test_cleanup_shot_work_is_linear_in_records(self):
        num_batches, records_per_batch = 5, 4

        for _ in range(num_batches):
            operations = [_SafeFetch("sig", _CountingSignal())]
            mapper = _Mapper(operations)
            for shot in range(records_per_batch):
                mapper(Record(shot))

        # One sweep per record, not one per record per batch seen so far.
        self.assertEqual(_CountingSignal.calls, num_batches * records_per_batch)

    def test_every_signal_used_by_a_record_is_still_cleaned(self):
        """Guards the reset from being too aggressive.

        Scoping the registry to the record must not drop signals that the
        record itself is using -- both fetches below have to be cleaned up.
        """
        reg = SignalRegistry()
        operations = [
            _SafeFetch("a", _CountingSignal()),
            _SafeFetch("b", _CountingSignal()),
        ]

        _Mapper(operations)(Record(1))

        self.assertEqual(_CountingSignal.calls, 2)
        self.assertEqual(len(reg.signals), 2)

    def test_records_are_still_populated(self):
        operations = [_SafeFetch("sig", MockSignal())]
        record = _Mapper(operations)(Record(1))
        self.assertIsNotNone(record["sig"]["data"])


if __name__ == "__main__":
    unittest.main()


# Run as its own process, because the behaviour under test only happens when
# one exits: the workers must release what they hold before they go.
_PIPELINE_SCRIPT = """
import os
import sys

from toksearch import Pipeline
from toksearch.signal.mock_signal import MockSignal

MARKS = sys.argv[1]


class MarkerSignal(MockSignal):
    def cleanup(self):
        path = os.path.join(MARKS, "cleanup-%d" % os.getpid())
        open(path, "w").close()


if __name__ == "__main__":
    open(os.path.join(MARKS, "parent-%d" % os.getpid()), "w").close()
    pipe = Pipeline(list(range(40)))
    pipe.fetch("sig", MarkerSignal())
    pipe.keep([])
    pipe.compute_multiprocessing(num_workers=2)
"""


class TestWorkerCleanup(unittest.TestCase):
    """cleanup() has to actually run in the workers.

    Nothing on this path used to call it. The backend dispatches _map_single
    per record, which only calls cleanup_shot; _map_multiple, which does the
    cleanup(), is used by the other backends. Whatever a worker held -- a
    remote MDSplus connection above all -- was just dropped when it died.
    """

    def test_cleanup_runs_in_the_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            marks = os.path.join(tmp, "marks")
            os.makedirs(marks)
            script = os.path.join(tmp, "run_pipeline.py")
            with open(script, "w") as fh:
                fh.write(_PIPELINE_SCRIPT)

            subprocess.run([sys.executable, script, marks],
                           check=True, timeout=600)

            written = os.listdir(marks)
            parents = [f for f in written if f.startswith("parent-")]
            cleaned = [f for f in written if f.startswith("cleanup-")]
            parent_pid = parents[0].split("-", 1)[1]

            self.assertTrue(cleaned, "cleanup() never ran in any worker")
            # The workers, not the process that launched them.
            self.assertNotIn(f"cleanup-{parent_pid}", cleaned)


class TestWorkerCleanupRegistration(unittest.TestCase):

    def setUp(self):
        self._saved = mp_backend._cleanup_registered
        mp_backend._cleanup_registered = False
        SignalRegistry().reset()

    def tearDown(self):
        mp_backend._cleanup_registered = self._saved
        SignalRegistry().reset()

    def test_registers_once_however_many_records(self):
        registered = []
        with mock.patch.object(mp_backend.atexit, "register",
                               side_effect=registered.append):
            mapper = _Mapper([_SafeFetch("sig", MockSignal())])
            for shot in range(5):
                mapper(Record(shot))

        self.assertEqual(len(registered), 1)

    def test_the_handler_releases_registered_signals(self):
        sig = MockSignal()
        SignalRegistry().register(sig)

        mp_backend._cleanup_signals()

        self.assertNotIn(sig, SignalRegistry())

    def test_the_handler_does_not_raise_at_shutdown(self):
        # It runs during interpreter shutdown; throwing there only adds noise.
        with mock.patch.object(SignalRegistry, "cleanup",
                               side_effect=RuntimeError("boom")):
            mp_backend._cleanup_signals()
