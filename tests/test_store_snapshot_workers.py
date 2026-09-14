"""Every worker in a run reads from the same catalog snapshot.

This is the half of the defect that no per-process fix reaches. ptdata stops
a single process adopting a newer catalog mid-run, but under
``compute_multiprocessing``, Ray or Spark each worker builds its own resolver
at its own start time, so they can disagree from the very first fetch — and
the result set is then assembled from two states of the world, presenting as
success.

The fix is that ``compute()`` settles the snapshot before any worker exists
and exports it, so the workers inherit an answer rather than each computing
one. These tests assert the workers agree, and — separately — that this
harness would have noticed if they had not. A "they all agree" assertion is
worthless without the second: it passes just as readily when the collection
path silently collapses whatever the workers returned.

ptdata is deliberately not required. toksearch is device-neutral, the store
import is lazy, and CI has no ptdata installed — so the driver's resolver is
replaced and the workers only ever read the environment, which is what they
do in production too.
"""

import contextlib
import os
import unittest
from unittest import mock

from toksearch import Pipeline
from toksearch.signal import store_snapshot


# Duplicated from test_store_snapshot.py rather than imported. The canonical
# runner (tests/testit) cds into tests/ and discovers from there, so a plain
# cross-module import works under it and fails under `pytest` from the repo
# root. A fourteen-line fixture is cheaper than a test file that runs only
# one of the two ways people invoke it.
@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def report_pinned_snapshot(rec):
    """What this worker process believes the run is pinned to.

    Module level because loky pickles by reference; a closure or a lambda
    would not survive the trip to a worker.
    """
    rec["snap"] = os.environ.get(store_snapshot.VAR, "")
    return rec


def report_per_record_value(rec):
    """A value that genuinely differs per record, for the control below."""
    rec["snap"] = "snapshot-{}".format(rec["shot"])
    return rec


def fresh_workers():
    """Discard reused worker processes.

    joblib/loky keep a reusable executor, and a worker's environment is fixed
    when it is spawned. Without this, a test that sets the pin differently
    from the previous one runs on workers still holding the previous value --
    which is not a test artefact but the real constraint (see
    TestWorkersKeepTheEnvironmentTheyStartedWith below). Production is
    protected from it by SnapshotConflict; these tests defeat that by writing
    the variable directly, so they have to clear the workers themselves.
    """
    from joblib.externals.loky import get_reusable_executor

    get_reusable_executor().shutdown(wait=True)


def _snapshots_from(mapper, shots=8, workers=4):
    pipe = Pipeline(list(range(1, shots + 1)))
    pipe.map(mapper)
    pipe.keep(["snap"])
    return [rec["snap"] for rec in
            pipe.compute_multiprocessing(num_workers=workers, batch_size=1)]


class TestEveryWorkerAgrees(unittest.TestCase):
    def setUp(self):
        fresh_workers()

    def test_multiprocessing_workers_share_one_snapshot(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_RESOLVED"):
            got = _snapshots_from(report_pinned_snapshot)

        # Not merely "all equal": all equal to what the DRIVER resolved. A
        # run where every worker independently agreed on the empty string
        # would satisfy the weaker assertion.
        self.assertEqual(set(got), {"catalog_RESOLVED"})
        self.assertEqual(len(got), 8)

    def test_an_explicit_pin_reaches_every_worker_too(self):
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_FROM_CLI"):
            got = _snapshots_from(report_pinned_snapshot)

        self.assertEqual(set(got), {"catalog_FROM_CLI"})

    def test_this_harness_would_have_seen_a_disagreement(self):
        """The control, and the reason the tests above mean anything.

        `assertEqual(set(got), {...})` proves the workers agreed only if a
        disagreement could have reached the assertion in the first place. It
        could: the same pipeline shape, with a mapper whose value genuinely
        varies per record, returns every one of those values.

        Written as a permanent test rather than performed once by hand,
        because the property it protects — that this file can fail — is the
        one that silently stops holding when the harness changes.
        """
        got = _snapshots_from(report_per_record_value)
        self.assertEqual(len(set(got)), 8)


class TestWorkersKeepTheEnvironmentTheyStartedWith(unittest.TestCase):
    """The constraint that makes one snapshot per process a rule, not a style.

    A worker's environment is fixed when it is spawned, and loky reuses
    workers across runs. So a pin set in the driver after workers exist does
    not reach them. This is asserted rather than worked around, because it is
    the reason `pin_run` refuses to re-pin a process: the alternative is a
    second run that reports one snapshot and reads another.
    """

    def test_a_pin_set_after_the_workers_exist_does_not_reach_them(self):
        fresh_workers()
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_FIRST"):
            first = _snapshots_from(report_pinned_snapshot, shots=2)
            self.assertEqual(set(first), {"catalog_FIRST"})

            # Written directly, behind pin_run's back: production cannot
            # reach this state without raising SnapshotConflict.
            os.environ[store_snapshot.VAR] = "catalog_SECOND"
            second = _snapshots_from(report_pinned_snapshot, shots=2)

        self.assertEqual(set(second), {"catalog_FIRST"},
                         "workers picked up a pin set after they started; if "
                         "this now passes, the one-snapshot-per-process rule "
                         "in store_snapshot.pin_run can be relaxed")
        fresh_workers()

    def test_a_second_differently_pinned_run_is_refused(self):
        # What stands between that stale worker and a wrong answer.
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_PINNED_THIS_PROCESS", False), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_FIRST"):
            Pipeline([1]).compute_serial()
            with self.assertRaises(store_snapshot.SnapshotConflict) as cm:
                Pipeline.from_snapshot("catalog_SECOND", [1]).compute_serial()

        msg = str(cm.exception)
        self.assertIn("catalog_FIRST", msg)
        self.assertIn("catalog_SECOND", msg)
        # It must NOT tell them to unset the variable: they never set it, and
        # unsetting it would let the second run proceed on stale workers.
        self.assertIn("one per process", msg)
        self.assertNotIn("unset", msg)


class TestPinningIsNotContingentOnProvenance(unittest.TestCase):
    def setUp(self):
        fresh_workers()

    def test_a_run_with_no_provenance_backend_is_still_pinned(self):
        # The correctness fix cannot depend on opting into recording. In a
        # multiprocess run the consequence is visible: unpinned, the workers
        # would read an unset variable and report "".
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_RESOLVED"):
            got = _snapshots_from(report_pinned_snapshot)
        self.assertNotIn("", got)


if __name__ == "__main__":
    unittest.main()
