"""Settling one catalog snapshot for a run.

The defect these guard: "latest" is a lookup that moves, and a run that
performs it more than once can read early shots from one catalog and later
ones from the next. Fixing that per-process is not enough -- each worker
resolves independently, so the snapshot has to be settled before any worker
exists. See tests/test_store_snapshot_workers.py for that half, which is the
only one that needs more than one process.
"""

import contextlib
import os
import unittest
from unittest import mock

from toksearch import Pipeline
from toksearch.signal import store_snapshot


@contextlib.contextmanager
def env(**kw):
    """Set environment variables for the block. None means unset."""
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


class TestPinningARun(unittest.TestCase):
    def test_it_exports_the_resolved_snapshot(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_X"):
            self.assertEqual(store_snapshot.pin_run(), "catalog_X")
            self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"], "catalog_X")

    def test_it_does_not_overwrite_an_existing_pin(self):
        # `fdp run --snapshot` and a user's own export both precede the
        # pipeline, and both outrank it: there the environment was told,
        # here the pipeline is guessing.
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_USER"), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_X") as resolve:
            self.assertEqual(store_snapshot.pin_run(), "catalog_USER")
            self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"], "catalog_USER")
            resolve.assert_not_called()

    def test_no_store_root_is_not_an_error(self):
        with env(FDP_STORE_ROOT=None, FDP_STORE_SNAPSHOT=None):
            self.assertIsNone(store_snapshot.pin_run())
            self.assertNotIn("FDP_STORE_SNAPSHOT", os.environ)

    def test_a_missing_ptdata_is_not_an_error(self):
        # The one place the store guard skips rather than raising: nothing
        # was asked for. Contrast mds.py's _store_index, which raises because
        # by the time it runs a pin HAS been asked for.
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               side_effect=ImportError("no ptdata")):
            self.assertIsNone(store_snapshot.pin_run())
            self.assertNotIn("FDP_STORE_SNAPSHOT", os.environ)

    def test_an_unreachable_store_is_not_an_error(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               side_effect=OSError("origin down")):
            self.assertIsNone(store_snapshot.pin_run())

    def test_a_store_with_no_catalog_is_not_an_error(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve", return_value=""):
            self.assertIsNone(store_snapshot.pin_run())
            self.assertNotIn("FDP_STORE_SNAPSHOT", os.environ)


class TestPinningInCode(unittest.TestCase):
    def test_from_snapshot_wins_over_an_unset_environment(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_NEWER") as resolve:
            Pipeline.from_snapshot("catalog_PINNED", [1]).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"],
                             "catalog_PINNED")
            resolve.assert_not_called()

    def test_a_conflicting_environment_pin_raises_naming_both(self):
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_FROM_CLI"):
            pipe = Pipeline.from_snapshot("catalog_FROM_CODE", [1])
            with self.assertRaises(store_snapshot.SnapshotConflict) as cm:
                pipe.compute_serial()
        msg = str(cm.exception)
        self.assertIn("catalog_FROM_CLI", msg)
        self.assertIn("catalog_FROM_CODE", msg)

    def test_the_same_value_from_both_sources_is_not_a_conflict(self):
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_SAME"):
            Pipeline.from_snapshot("catalog_SAME", [1]).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"], "catalog_SAME")

    def test_a_continuation_keeps_the_pin(self):
        # A guarantee that quietly stops applying when a pipeline is extended
        # is the toksearch_mast defect's shape.
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None):
            pinned = Pipeline.from_snapshot("catalog_PINNED", [1])
            extended = Pipeline(pinned)
            extended.map(lambda rec: rec.update({"x": 1}))
            extended.compute_serial()
            self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"],
                             "catalog_PINNED")

    def test_from_snapshot_composes_with_another_source(self):
        inner = Pipeline([1, 2])
        pipe = Pipeline.from_snapshot("catalog_PINNED", inner)
        self.assertEqual(pipe._snapshot, "catalog_PINNED")


class TestLatestNamesNothing(unittest.TestCase):
    """A sweep script defaults --snapshot to the word; it must not have to
    special-case it, and the shortest special case drops the pin entirely."""

    def test_latest_means_no_explicit_pin(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_NEWEST"):
            Pipeline.from_snapshot("latest", [1]).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"],
                             "catalog_NEWEST")

    def test_latest_does_not_override_the_environment(self):
        # What keeps `fdp run --snapshot X python sweep.py` working: a
        # default nobody typed must not outrank a value the user did.
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_FROM_CLI"), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_NEWEST") as resolve:
            Pipeline.from_snapshot("latest", [1]).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"],
                             "catalog_FROM_CLI")
            resolve.assert_not_called()

    def test_latest_is_not_a_conflict(self):
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_FROM_CLI"):
            # Must not raise: the word names nothing to conflict with.
            Pipeline.from_snapshot("latest", [1]).compute_serial()

    def test_latest_is_case_insensitive_and_stripped(self):
        for word in ("LATEST", " latest ", "Latest"):
            with self.subTest(word=word):
                with env(FDP_STORE_ROOT="/some/root",
                         FDP_STORE_SNAPSHOT=None), \
                     mock.patch.object(store_snapshot, "_resolve",
                                       return_value="catalog_NEWEST"):
                    Pipeline.from_snapshot(word, [1]).compute_serial()
                    self.assertEqual(os.environ["FDP_STORE_SNAPSHOT"],
                                     "catalog_NEWEST")

    def test_the_word_never_reaches_the_environment(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_NEWEST"):
            Pipeline.from_snapshot("latest", [1]).compute_serial()
            got = os.environ.get("FDP_STORE_SNAPSHOT", "")
        self.assertNotIn("latest", got.lower())


class TestTheRunContextCarriesIt(unittest.TestCase):
    def _record(self, pipe):
        seen = []

        class Backend:
            run_id = "r1"

            def on_compute_start(self, ctx):
                seen.append(ctx)

            def on_compute_end(self, ctx, result):
                pass

        pipe.compute_serial(provenance=Backend())
        return seen[0]

    def test_the_snapshot_reaches_a_provenance_backend(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_snapshot, "_resolve",
                               return_value="catalog_X"):
            ctx = self._record(Pipeline([1, 2]))
        self.assertEqual(ctx.to_dict()["store"], {"snapshot": "catalog_X"})

    def test_no_store_leaves_the_field_none(self):
        with env(FDP_STORE_ROOT=None, FDP_STORE_SNAPSHOT=None):
            ctx = self._record(Pipeline([1, 2]))
        self.assertIsNone(ctx.to_dict()["store"])

    def test_the_snapshot_changes_the_input_identity(self):
        # input_identity is "what data this run reads". The snapshot decides
        # WHICH VERSION of each shot is read, so two runs with identical
        # signals but different snapshots read different bytes and must not
        # share an input artifact id in the lineage graph.
        ids = []
        for snap in ("catalog_A", "catalog_B"):
            with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None), \
                 mock.patch.object(store_snapshot, "_resolve",
                                   return_value=snap):
                ids.append(self._record(Pipeline([1, 2])).input_identity())
        self.assertNotEqual(ids[0], ids[1])

    def test_an_unpinned_run_keeps_its_previous_identity(self):
        # The key is omitted entirely when nothing is pinned, so a device
        # with no store -- MAST -- keeps the artifact ids it already has.
        # Including `None` would churn every existing id for no new fact.
        with env(FDP_STORE_ROOT=None, FDP_STORE_SNAPSHOT=None):
            ctx = self._record(Pipeline([1, 2]))
        self.assertNotIn("store", ctx._input_identity_payload())


if __name__ == "__main__":
    unittest.main()
