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


if __name__ == "__main__":
    unittest.main()


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
