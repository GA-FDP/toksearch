"""Finding a snapshot: where they live, which exist, which one this run reads.

No DuckDB here. HTTP is mocked at `snapshot._request`, the one function
that touches the network, so these run in the conda package's test phase.
"""

import contextlib
import json
import os
import unittest
from unittest import mock

from fdp_schema import SqlSnapshotLocator

from toksearch.sql import snapshot

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

LOC = SqlSnapshotLocator(
    name="d3drdb", base_url="https://origin.example/fdp-d3d/metadata/d3drdb",
    id_pattern="d3drdb_*")

PROPFIND_BODY = b"""<?xml version="1.0"?>
<D:multistatus xmlns:D="DAV:">
 <D:response><D:href>/fdp-d3d/metadata/d3drdb/</D:href></D:response>
 <D:response><D:href>/fdp-d3d/metadata/d3drdb/_spike</D:href></D:response>
 <D:response><D:href>/fdp-d3d/metadata/d3drdb/d3drdb_20260901T000000Z</D:href></D:response>
 <D:response><D:href>/fdp-d3d/metadata/d3drdb/d3drdb_20261005T120000Z/</D:href></D:response>
</D:multistatus>"""

MANIFEST = {
    "schema": "fdp-sql-snapshot/1", "id": "d3drdb_20261005T120000Z",
    "created_at": "2026-10-05T12:00:00Z",
    "source": {"server": "d3drdb.gat.com:8001", "database": "d3drdb",
               "shot_ceiling": 207912, "isolation": "read committed"},
    "transforms": {"rtrim_char": True, "collation": "nocase"},
    "excluded": [{"table": "PERSONNEL", "reason": "people"}],
    "skipped": [],
    "tables": [{"name": "SHOTS", "rows": 3, "primary_key": ["SHOT"],
                "sorted_by": ["SHOT"], "read_at": "2026-10-05T12:00:01Z",
                "files": [{"path": "SHOTS.parquet", "bytes": 10,
                           "sha256": "0" * 64, "rows": 3}]}],
}

class TestBaseUrl(unittest.TestCase):
    def test_https_is_verbatim_without_trailing_slash(self):
        self.assertEqual(snapshot.resolve_base("https://h/a/b/"), "https://h/a/b")

    def test_file_url_becomes_a_path(self):
        self.assertEqual(snapshot.resolve_base("file:///tmp/snaps/"), "/tmp/snaps")

    def test_pelican_goes_through_the_well_known_document(self):
        with mock.patch.object(snapshot, "_request", return_value=(200, {},
                json.dumps({"director_endpoint": "https://director.example"}).encode())) as req:
            snapshot._director_endpoint.cache_clear()
            base = snapshot.resolve_base("pelican://osg-htc.org:443/fdp-d3d/metadata/d3drdb")
        self.assertEqual(base, "https://director.example/fdp-d3d/metadata/d3drdb")
        req.assert_called_once_with(
            "https://osg-htc.org/.well-known/pelican-configuration", token=None)

    def test_other_schemes_are_refused(self):
        with self.assertRaises(snapshot.SnapshotError):
            snapshot.resolve_base("ftp://h/x")

class TestListing(unittest.TestCase):
    def test_propfind_filtered_by_pattern_and_sorted(self):
        with mock.patch.object(snapshot, "_request",
                               return_value=(207, {}, PROPFIND_BODY)) as req:
            ids = snapshot.list_ids("https://h/fdp-d3d/metadata/d3drdb", "d3drdb_*", token="t")
        self.assertEqual(ids, ["d3drdb_20260901T000000Z", "d3drdb_20261005T120000Z"])
        req.assert_called_once_with(
            "https://h/fdp-d3d/metadata/d3drdb/", token="t",
            method="PROPFIND", headers={"Depth": "1"})

    def test_local_directory(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            for name in ("d3drdb_a", "_spike", "d3drdb_b", "notes.txt"):
                os.makedirs(os.path.join(d, name), exist_ok=True)
            self.assertEqual(snapshot.list_ids(d, "d3drdb_*", token=None),
                             ["d3drdb_a", "d3drdb_b"])

class TestManifest(unittest.TestCase):
    def test_fetched_and_validated(self):
        with mock.patch.object(snapshot, "_request",
                               return_value=(200, {}, json.dumps(MANIFEST).encode())) as req:
            m = snapshot.fetch_manifest("https://h/x", "d3drdb_20261005T120000Z", token="t")
        self.assertEqual(m["id"], "d3drdb_20261005T120000Z")
        req.assert_called_once_with("https://h/x/d3drdb_20261005T120000Z/manifest.json", token="t")

    def test_missing_snapshot_names_the_id_and_base(self):
        with mock.patch.object(snapshot, "_request", return_value=(404, {}, b"")):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.fetch_manifest("https://h/x", "d3drdb_nope", token="t")
        self.assertIn("d3drdb_nope", str(cm.exception))
        self.assertIn("https://h/x", str(cm.exception))

    def test_wrong_schema_is_refused(self):
        bad = dict(MANIFEST, schema="fdp-d3drdb-snapshot/0")
        with mock.patch.object(snapshot, "_request", return_value=(200, {}, json.dumps(bad).encode())):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.fetch_manifest("https://h/x", "d3drdb_x", token="t")
        self.assertIn("fdp-d3drdb-snapshot/0", str(cm.exception))

    def test_unknown_collation_is_refused(self):
        bad = dict(MANIFEST, transforms={"collation": "latin1_ci"})
        with mock.patch.object(snapshot, "_request", return_value=(200, {}, json.dumps(bad).encode())):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.fetch_manifest("https://h/x", "d3drdb_x", token="t")
        self.assertIn("latin1_ci", str(cm.exception))

    def test_local_manifest(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "d3drdb_x"))
            with open(os.path.join(d, "d3drdb_x", "manifest.json"), "w") as fh:
                json.dump(MANIFEST, fh)
            self.assertEqual(snapshot.fetch_manifest(d, "d3drdb_x", token=None)["id"],
                             MANIFEST["id"])

class TestResolve(unittest.TestCase):
    """Mirrors tests/test_store_catalog.py: code, environment, newest."""

    def setUp(self):
        snapshot._pinned.clear()

    def test_code_wins_and_is_exported(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None):
            sid = snapshot.resolve(LOC, snapshot="d3drdb_X", token="t")
            self.assertEqual(sid, "d3drdb_X")
            self.assertEqual(os.environ["FDP_SQL_SNAPSHOT_D3DRDB"], "d3drdb_X")

    def test_environment_is_honoured_without_a_lookup(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB="d3drdb_USER"), \
             mock.patch.object(snapshot, "list_ids") as listing:
            self.assertEqual(snapshot.resolve(LOC, token="t"), "d3drdb_USER")
            listing.assert_not_called()

    def test_conflict_names_both(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB="d3drdb_USER"):
            with self.assertRaises(snapshot.SnapshotConflict) as cm:
                snapshot.resolve(LOC, snapshot="d3drdb_CODE", token="t")
        msg = str(cm.exception)
        self.assertIn("d3drdb_USER", msg)
        self.assertIn("d3drdb_CODE", msg)
        self.assertIn("FDP_SQL_SNAPSHOT_D3DRDB", msg)

    def test_a_second_pin_in_one_process_says_so(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None):
            snapshot.resolve(LOC, snapshot="d3drdb_A", token="t")
            with self.assertRaises(snapshot.SnapshotConflict) as cm:
                snapshot.resolve(LOC, snapshot="d3drdb_B", token="t")
        self.assertIn("earlier", str(cm.exception))
        self.assertIn("worker processes", str(cm.exception))

    def test_the_same_value_from_both_is_not_a_conflict(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB="d3drdb_A"):
            self.assertEqual(snapshot.resolve(LOC, snapshot="d3drdb_A", token="t"), "d3drdb_A")

    def test_newest_is_resolved_and_exported(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_STORE_ROOT=None, FDP_STORE_CATALOG=None), \
             mock.patch.object(snapshot, "list_ids", return_value=["d3drdb_1", "d3drdb_2"]):
            self.assertEqual(snapshot.resolve(LOC, token="t"), "d3drdb_2")
            self.assertEqual(os.environ["FDP_SQL_SNAPSHOT_D3DRDB"], "d3drdb_2")

    def test_nothing_published_is_an_error_not_a_fallback(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_STORE_ROOT=None, FDP_STORE_CATALOG=None), \
             mock.patch.object(snapshot, "list_ids", return_value=[]):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.resolve(LOC, token="t")
        self.assertIn(LOC.base_url, str(cm.exception))
        self.assertNotIn("FDP_SQL_SNAPSHOT_D3DRDB", os.environ)


class TestCatalogPairing(unittest.TestCase):
    def setUp(self):
        snapshot._pinned.clear()

    def test_the_catalog_pairing_outranks_newest(self):
        meta = json.dumps({"sql_snapshots": {"d3drdb": "d3drdb_PAIRED"}}).encode()
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_STORE_ROOT="https://h/fdp-d3d",
                 FDP_STORE_CATALOG="catalog_X"), \
             mock.patch.object(snapshot, "_request", return_value=(200, {}, meta)) as req, \
             mock.patch.object(snapshot, "list_ids") as listing:
            self.assertEqual(snapshot.resolve(LOC, token="t"), "d3drdb_PAIRED")
        req.assert_called_once_with("https://h/fdp-d3d/catalog/catalog_X/meta.json", token="t")
        listing.assert_not_called()

    def test_no_meta_file_means_newest(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_STORE_ROOT="https://h/fdp-d3d",
                 FDP_STORE_CATALOG="catalog_X"), \
             mock.patch.object(snapshot, "_request", return_value=(404, {}, b"")), \
             mock.patch.object(snapshot, "list_ids", return_value=["d3drdb_1"]):
            self.assertEqual(snapshot.resolve(LOC, token="t"), "d3drdb_1")

    def test_meta_without_this_locator_means_newest(self):
        meta = json.dumps({"sql_snapshots": {"other": "x"}}).encode()
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_STORE_ROOT="https://h/fdp-d3d",
                 FDP_STORE_CATALOG="catalog_X"), \
             mock.patch.object(snapshot, "_request", return_value=(200, {}, meta)), \
             mock.patch.object(snapshot, "list_ids", return_value=["d3drdb_1"]):
            self.assertEqual(snapshot.resolve(LOC, token="t"), "d3drdb_1")

    def test_no_catalog_pinned_skips_the_step(self):
        with env(FDP_SQL_SNAPSHOT_D3DRDB=None, FDP_STORE_ROOT="https://h/fdp-d3d",
                 FDP_STORE_CATALOG=None), \
             mock.patch.object(snapshot, "_request") as req, \
             mock.patch.object(snapshot, "list_ids", return_value=["d3drdb_1"]):
            snapshot.resolve(LOC, token="t")
        req.assert_not_called()


if __name__ == "__main__":
    unittest.main()
