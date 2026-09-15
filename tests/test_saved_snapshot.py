"""Replaying a saved snapshot.

A saved snapshot names the exact version of every shot and shard a run read.
Replaying one needs no new resolution path: the client holds the whole
mapping before the run starts, so the shots expand into records carrying an
explicit `version` -- B5's existing per-record path, which already works on
both transports and every compute backend.

Shards are the one thing B5 does not give: a shot's version pin does not
reach them, so a snapshot's shard versions have to be applied where a tree
is opened.
"""

import contextlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from toksearch import Pipeline
from toksearch.signal import mds as mds_module
from toksearch.signal import store_catalog

CATALOG = "catalog_20260907T232802Z"


@contextlib.contextmanager
def env(**kw):
    old = {k: os.environ.get(k) for k in kw}
    for k, v in kw.items():
        os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def a_snapshot(shots=((165920, 2), (165921, 3)),
               shared=(("efit01-0", 5),), catalog=CATALOG):
    return {
        "schema": "fdp-snapshot/1",
        "catalog": catalog,
        "store_root": "pelican://osg-htc.org:443/fdp-d3d",
        "created_at": "2026-09-14T20:00:00Z",
        "shots": [{"shot": s, "version": v, "dir_hash": "%032x" % s}
                  for s, v in shots],
        "shared": [{"shard": k, "version": v, "dir_hash": "a" * 32}
                   for k, v in shared],
    }


class SnapshotFileTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, doc=None, name="snap.json"):
        path = self.dir / name
        path.write_text(json.dumps(doc if doc is not None else a_snapshot()))
        return str(path)


class TestItExpandsToPinnedRecords(SnapshotFileTest):
    def test_one_record_per_entry_carrying_its_version(self):
        recs = Pipeline.from_snapshot(self.write()).compute_serial()
        self.assertEqual([(r["shot"], r["version"]) for r in recs],
                         [(165920, 2), (165921, 3)])

    def test_it_pins_the_catalog_from_the_file(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_CATALOG=None,
                 FDP_STORE_SHARDS=None):
            Pipeline.from_snapshot(self.write()).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_CATALOG"], CATALOG)

    def test_it_carries_the_shard_versions(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_CATALOG=None,
                 FDP_STORE_SHARDS=None):
            Pipeline.from_snapshot(self.write()).compute_serial()
            self.assertEqual(json.loads(os.environ["FDP_STORE_SHARDS"]),
                             {"efit01-0": 5})

    def test_a_snapshot_with_no_shards_is_still_valid(self):
        path = self.write(a_snapshot(shared=()))
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_CATALOG=None,
                 FDP_STORE_SHARDS=None):
            recs = Pipeline.from_snapshot(path).compute_serial()
        self.assertEqual(len(recs), 2)


class TestItRefusesRubbish(SnapshotFileTest):
    def test_a_missing_file_names_the_path(self):
        with self.assertRaises(store_catalog.SnapshotFileError) as cm:
            Pipeline.from_snapshot(str(self.dir / "nope.json"))
        self.assertIn("nope.json", str(cm.exception))

    def test_an_unknown_schema_is_refused(self):
        path = self.write(dict(a_snapshot(), schema="fdp-snapshot/99"))
        with self.assertRaises(store_catalog.SnapshotFileError) as cm:
            Pipeline.from_snapshot(path)
        self.assertIn("fdp-snapshot/99", str(cm.exception))

    def test_a_catalog_stamp_is_refused_naming_from_catalog(self):
        # It used to mean this. A pipeline written last week would otherwise
        # have the stamp taken as a filename and be silently unpinned.
        with self.assertRaises(ValueError) as cm:
            Pipeline.from_snapshot(CATALOG)
        self.assertIn("from_catalog", str(cm.exception))

    def test_an_empty_shot_list_is_refused(self):
        path = self.write(dict(a_snapshot(), shots=[]))
        with self.assertRaises(store_catalog.SnapshotFileError):
            Pipeline.from_snapshot(path)


class TestShardsBeatTheCatalog(unittest.TestCase):
    """Spec rule 6a. A shot's version pin does not reach a shard, so without
    this a replay reads the right data and the wrong instrument description."""

    def _resolve(self, shard_version, pinned):
        got = mock.Mock(found=True, version=7, snapshot=CATALOG, detail="")
        shared = mock.Mock(found=True, version=shard_version, snapshot=CATALOG)
        index = mock.Mock(**{"resolve_version.return_value": got,
                             "resolve_shared_version.return_value": shared})
        seen = {}

        def capture(views_root, shard, version, treename):
            seen["shard"], seen["version"] = shard, version
            return ["/p"]

        with env(FDP_STORE_SHARDS=pinned), \
             mock.patch.object(mds_module, "_store_index", return_value=index), \
             mock.patch.object(mds_module, "shared_tree_paths", capture):
            mds_module._resolve_store_path(
                "efit01", 165920, None, None,
                "pelican://host/ns", "/views", "archives", "subject")
        return seen

    def test_a_named_shard_beats_the_catalog(self):
        seen = self._resolve(shard_version=2, pinned=json.dumps({"efit01-0": 5}))
        self.assertEqual(seen["version"], 5)

    def test_an_unnamed_shard_falls_back_to_the_catalog(self):
        seen = self._resolve(shard_version=2, pinned=json.dumps({"bci-0": 9}))
        self.assertEqual(seen["version"], 2)

    def test_no_pinned_shards_at_all_falls_back(self):
        seen = self._resolve(shard_version=2, pinned=None)
        self.assertEqual(seen["version"], 2)


class TestSelfContainment(unittest.TestCase):
    """The claim that makes a citation outlive catalog pruning.

    A snapshot naming its shards should need nothing from the catalog: the
    version of every shot and shard is written down, and §7 of the store spec
    says the path is DERIVABLE from (shot, version). Asserted on the index,
    not assumed -- "self-contained" is either a fact or a slogan.
    """

    def test_a_fully_pinned_read_consults_no_catalog(self):
        got = mock.Mock(found=True, version=2, snapshot=CATALOG, detail="")
        index = mock.Mock(**{"resolve_version.return_value": got})

        with env(FDP_STORE_SHARDS=json.dumps({"efit01-0": 5}),
                 FDP_STORE_CATALOG=CATALOG), \
             mock.patch.object(mds_module, "_store_index", return_value=index), \
             mock.patch.object(mds_module, "shared_tree_paths",
                               lambda *a, **k: ["/p"]):
            mds_module._resolve_store_path(
                "efit01", 165920, 2, None,
                "pelican://host/ns", "/views", "archives", "subject")

        # The shard came from the file, so the catalog was not asked for it.
        index.resolve_shared_version.assert_not_called()


class TestRunContextCarriesTheShots(unittest.TestCase):
    """A provenance backend receives a RunContext and nothing else, so if it
    is to build a snapshot the shots have to be in there. SourceSpec records
    a count and a hash of them, which cannot be turned back into a list."""

    def _ctx(self, pipe):
        seen = []

        class B:
            run_id = "r1"
            def on_compute_start(self, ctx): seen.append(ctx)
            def on_compute_end(self, ctx, result): pass

        with env(FDP_STORE_ROOT=None, FDP_STORE_CATALOG=None,
                 FDP_STORE_SHARDS=None):
            pipe.compute_serial(provenance=B())
        return seen[0]

    def test_the_shots_reach_the_backend(self):
        ctx = self._ctx(Pipeline([165921, 165920]))
        self.assertEqual(ctx.shots, (165920, 165921))

    def test_they_survive_to_dict(self):
        ctx = self._ctx(Pipeline([165920]))
        self.assertEqual(ctx.to_dict()["shots"], (165920,))

    def test_they_do_not_change_the_input_identity(self):
        # source.hash already covers the shot list. Adding it to the hash
        # would churn every artifact id in the lineage graph to record a
        # fact already recorded.
        from toksearch.provenance.context import (
            BackendSpec, CodeSpec, RunContext, SourceSpec)

        common = dict(
            source=SourceSpec(kind="shotlist", count=1, hash="abc"),
            ops=(), signals={},
            backend=BackendSpec(kind="SerialRecordSet", config={}),
            code=CodeSpec(commit=None, dirty=False, repo_root=None,
                          script=None, argv=()),
        )
        without = RunContext(**common)
        with_shots = RunContext(**common, shots=(165920, 165921))
        self.assertEqual(without.input_identity(), with_shots.input_identity())

    def test_a_recordset_parent_has_no_shots_rather_than_a_wrong_list(self):
        pipe = Pipeline([165920])
        ctx = self._ctx(Pipeline(pipe.compute_serial()))
        self.assertIn(ctx.shots, (None, (165920,)))


class TestItWarnsAboutTreesTheSnapshotDoesNotCover(SnapshotFileTest):
    """A snapshot naming SOME shards looks better protected than one naming
    none, and `save` cannot know which trees a later run will open. The
    driver can: at compute time it holds both the snapshot's shards and the
    trees its signals will read.

    Without this, a replay pinned to efit01-0 that reads `bci` resolves bci's
    model tree through whatever catalog is current -- reproducible for the
    measurement, not for what it means -- and says nothing.
    """

    def _warnings(self, doc, treenames):
        from toksearch import MdsSignal
        pipe = Pipeline.from_snapshot(self.write(doc))
        for i, tree in enumerate(treenames):
            pipe.fetch("s%d" % i, MdsSignal(r"\x", tree))
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_CATALOG=None,
                 FDP_STORE_SHARDS=None), \
             mock.patch.object(mds_module, "_store_index",
                               side_effect=AssertionError("no fetch here")):
            import warnings
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                pipe._warn_uncovered_trees()
            return [str(w.message) for w in caught]

    def test_a_tree_the_snapshot_does_not_name_is_reported(self):
        msgs = self._warnings(a_snapshot(shared=(("efit01-0", 5),)), ["bci"])
        self.assertEqual(len(msgs), 1)
        self.assertIn("bci", msgs[0])
        self.assertIn("catalog", msgs[0].lower())

    def test_a_covered_tree_is_silent(self):
        self.assertEqual(
            self._warnings(a_snapshot(shared=(("efit01-0", 5),)), ["efit01"]),
            [])

    def test_a_snapshot_naming_no_shards_reports_every_tree(self):
        msgs = self._warnings(a_snapshot(shared=()), ["efit01", "bci"])
        self.assertEqual(len(msgs), 1)
        self.assertIn("bci", msgs[0])
        self.assertIn("efit01", msgs[0])

    def test_a_pipeline_with_no_tree_signals_is_silent(self):
        self.assertEqual(self._warnings(a_snapshot(), []), [])


class TestAnOrdinaryRunIsUnaffected(unittest.TestCase):
    """Every MDSplus pipeline now passes through _warn_uncovered_trees, so a
    mistake in it breaks every run rather than only replays. Removing the
    "am I replaying?" guard failed no test until this one existed."""

    def test_a_pipeline_that_is_not_a_replay_is_silent(self):
        import warnings as _w
        from toksearch import MdsSignal
        pipe = Pipeline([165920])
        pipe.fetch("s", MdsSignal(r"\x", "bci"))
        with _w.catch_warnings(record=True) as caught:
            _w.simplefilter("always")
            pipe._warn_uncovered_trees()
        self.assertEqual([str(x.message) for x in caught], [])

    def test_it_does_not_raise_when_there_is_no_snapshot(self):
        # `self._shards` is None off a replay, and `{e["shard"] for e in None}`
        # is a TypeError -- which would surface at compute() on every run.
        pipe = Pipeline([165920])
        pipe._warn_uncovered_trees()      # must simply return


class TestABogusCatalogIsRefusedForShardsToo(unittest.TestCase):
    """Spec rule 6, the half that did not hold.

    The shot half works by construction: a pinned version outlives its
    catalog. The shard half did not. `if shared.found:` skipped a failed
    resolution without a word, so a replay naming a pruned catalog and no
    shards read its measurements from the pin and quietly dropped the model
    tree out of the search path -- and under a pin the archives fallback is
    dropped too, so there was nothing behind it.

    Whether a user ever saw this depended on whether the shot's own tree
    happened to carry the node they asked for. Against the live origin it
    did: three shots returned data at `catalog_nope`, which is the acceptance
    control passing when it was supposed to fail.
    """

    def _resolve(self, miss, detail="no such snapshot"):
        got = mock.Mock(found=True, version=7, snapshot=CATALOG, detail="")
        shared = mock.Mock(found=False, miss=miss, detail=detail)
        index = mock.Mock(**{"resolve_version.return_value": got,
                             "resolve_shared_version.return_value": shared})
        with env(FDP_STORE_SHARDS=None, FDP_STORE_CATALOG="catalog_nope"), \
             mock.patch.object(mds_module, "_store_index", return_value=index), \
             mock.patch.object(mds_module, "_snapshot_missing",
                               lambda: "SnapshotMissing"), \
             mock.patch.object(mds_module, "shared_tree_paths",
                               lambda *a, **k: ["/p"]):
            return mds_module._resolve_store_path(
                "bci", 165920, 1, None,
                "pelican://host/ns", "/views", "archives", "subject")

    def test_a_catalog_that_does_not_exist_raises(self):
        with self.assertRaises(mds_module.StoreVersionError) as cm:
            self._resolve("SnapshotMissing")
        self.assertIn("catalog_nope", str(cm.exception))

    def test_the_message_says_how_to_stop_needing_the_catalog(self):
        # Naming the shards is the fix, and it is the whole point of the
        # feature. A message that only reports the breakage sends the user
        # looking for a catalog that was pruned on purpose.
        with self.assertRaises(mds_module.StoreVersionError) as cm:
            self._resolve("SnapshotMissing")
        self.assertIn("--tree", str(cm.exception))
        self.assertIn("bci", str(cm.exception))

    def test_a_shard_simply_not_in_the_catalog_still_skips(self):
        # Different fact, different handling: the catalog exists and does not
        # list this shard, which is an un-ingested tree, not a broken pin.
        # Failing here would break every pinned read of a tree the shared
        # area has not absorbed -- the migration story B5 deliberately kept.
        version, path = self._resolve("ShardNotInCatalog")
        self.assertEqual(version, 7)


if __name__ == "__main__":
    unittest.main()
