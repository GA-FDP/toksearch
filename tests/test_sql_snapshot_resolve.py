"""Finding a snapshot: where they live, which exist, which one this run reads.

No DuckDB here. HTTP is mocked at `snapshot._request`, the one function
that touches the network, so these run in the conda package's test phase.
"""

import contextlib
import email.message
import io
import json
import os
import socket
import tempfile
import unittest
import urllib.error
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
            "https://osg-htc.org:443/.well-known/pelican-configuration", token=None)

    def test_other_schemes_are_refused(self):
        with self.assertRaises(snapshot.SnapshotError):
            snapshot.resolve_base("ftp://h/x")

    def test_plain_http_to_loopback_is_verbatim_and_remote(self):
        # the test server's scheme; it is read over HTTP, not as a path
        base = snapshot.resolve_base("http://127.0.0.1:1234/x/")
        self.assertEqual(base, "http://127.0.0.1:1234/x")
        self.assertFalse(snapshot.is_local(base))

    def test_plain_http_to_any_other_host_is_refused(self):
        with self.assertRaises(snapshot.SnapshotError):
            snapshot.resolve_base("http://example.com/x")

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
            for name in ("d3drdb_20261005T120000Z", "_spike", "d3drdb_20260901T000000Z",
                         "d3drdb_staging", "d3drdb_a+b", "notes.txt",
                         "d3drdb_copy_20250101T000000Z",
                         "d3drdb_20270101T000000Z\n"):
                os.makedirs(os.path.join(d, name), exist_ok=True)
            with open(os.path.join(d, "d3drdb_20270101T000000Z.txt"), "w"):
                pass
            self.assertEqual(snapshot.list_ids(d, "d3drdb_*", token=None),
                             ["d3drdb_copy_20250101T000000Z",
                              "d3drdb_20260901T000000Z",
                              "d3drdb_20261005T120000Z"])

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


LISTING_BODY = b"""<?xml version="1.0"?>
<D:multistatus xmlns:D="DAV:">
 <D:response><D:href>/b/d3drdb/</D:href></D:response>
 <D:response><D:href>/b/d3drdb/d3drdb_staging</D:href></D:response>
 <D:response><D:href>/b/d3drdb/manifest.json</D:href></D:response>
 <D:response><D:href>/b/d3drdb/d3drdb_a%2Bb</D:href></D:response>
 <D:response><D:href>/b/d3drdb/_spike</D:href></D:response>
 <D:response><D:href/></D:response>
 <D:response><D:href>/b/d3drdb/d3drdb_20261005T120000Z/</D:href></D:response>
 <D:response><D:href>/b/d3drdb/d3drdb_20260901T000000Z</D:href></D:response>
</D:multistatus>"""


class TestListingStrictness(unittest.TestCase):
    def test_only_stamped_ids_are_listed(self):
        with mock.patch.object(snapshot, "_request", return_value=(207, {}, LISTING_BODY)):
            ids = snapshot.list_ids("https://h/b/d3drdb", "d3drdb_*", token="t")
        self.assertEqual(ids, ["d3drdb_20260901T000000Z", "d3drdb_20261005T120000Z"])

    def test_percent_escapes_are_decoded_before_matching(self):
        body = (b'<D:multistatus xmlns:D="DAV:"><D:response><D:href>'
                b'/b/d3drdb_20261005T120000Z%2F</D:href></D:response></D:multistatus>')
        with mock.patch.object(snapshot, "_request", return_value=(207, {}, body)):
            self.assertEqual(snapshot.list_ids("https://h/b", "d3drdb_*", token="t"),
                             ["d3drdb_20261005T120000Z"])

    def test_malformed_xml_is_a_snapshot_error(self):
        with mock.patch.object(snapshot, "_request", return_value=(207, {}, b"<D:multi")):
            with self.assertRaises(snapshot.SnapshotError):
                snapshot.list_ids("https://h/b", "d3drdb_*", token="t")


class TestRobustness(unittest.TestCase):
    def test_manifest_that_is_not_json(self):
        with mock.patch.object(snapshot, "_request", return_value=(200, {}, b"<html>")):
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.fetch_manifest("https://h/x", "d3drdb_x", token="t")
        self.assertIn("https://h/x/d3drdb_x/manifest.json", str(cm.exception))

    def test_local_manifest_that_is_not_json(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "d3drdb_x"))
            with open(os.path.join(d, "d3drdb_x", "manifest.json"), "w") as fh:
                fh.write("{")
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.fetch_manifest(d, "d3drdb_x", token=None)
        self.assertIn("manifest.json", str(cm.exception))

    def test_well_known_without_director_endpoint(self):
        with mock.patch.object(snapshot, "_request",
                               return_value=(200, {}, b'{"other": 1}')):
            snapshot._director_endpoint.cache_clear()
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.resolve_base("pelican://fed.example/ns/x")
        self.assertIn("fed.example", str(cm.exception))

    def test_well_known_that_is_not_json(self):
        with mock.patch.object(snapshot, "_request", return_value=(200, {}, b"nope")):
            snapshot._director_endpoint.cache_clear()
            with self.assertRaises(snapshot.SnapshotError) as cm:
                snapshot.resolve_base("pelican://fed2.example/ns/x")
        self.assertIn("fed2.example", str(cm.exception))

    def test_pelican_port_is_kept_for_the_well_known_fetch(self):
        doc = json.dumps({"director_endpoint": "https://d.example/"}).encode()
        with mock.patch.object(snapshot, "_request", return_value=(200, {}, doc)) as req:
            snapshot._director_endpoint.cache_clear()
            snapshot.resolve_base("pelican://fed3.example:8444/ns")
        req.assert_called_once_with(
            "https://fed3.example:8444/.well-known/pelican-configuration", token=None)


class _Resp:
    def __init__(self, status=200, headers=None, body=b"ok", read_exc=None):
        self.status, self.headers, self._body, self._exc = status, headers or {}, body, read_exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        if self._exc:
            raise self._exc
        return self._body


def _redirect(url, location, code=307):
    h = email.message.Message()
    if location is not None:
        h["Location"] = location
    return urllib.error.HTTPError(url, code, "r", h, io.BytesIO(b""))


class _Opener:
    """Scripted: each call pops the next item, returning or raising it."""

    def __init__(self, *script):
        self.script, self.seen = list(script), []

    def open(self, req, timeout=None):
        self.seen.append((req.full_url, req.get_method(),
                          req.get_header("Authorization"), req.get_header("Depth")))
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class TestRequest(unittest.TestCase):
    def run_with(self, opener, url="https://a.example/dir/x", **kw):
        with mock.patch.object(snapshot, "_opener", opener):
            return snapshot._request(url, **kw)

    def test_relative_location_resolves_against_the_current_url(self):
        op = _Opener(_redirect("u", "../y"), _Resp())
        self.run_with(op)
        self.assertEqual(op.seen[1][0], "https://a.example/y")

    def test_auth_kept_same_host_dropped_cross_host_depth_survives(self):
        op = _Opener(_redirect("u", "/same"),
                     _redirect("u", "https://b.example/other"), _Resp())
        status, _, body = self.run_with(op, token="t", method="PROPFIND",
                                        headers={"Depth": "1"})
        self.assertEqual((status, body), (200, b"ok"))
        self.assertEqual([s[2] for s in op.seen], ["Bearer t", "Bearer t", None])
        self.assertEqual([s[3] for s in op.seen], ["1", "1", "1"])
        self.assertEqual({s[1] for s in op.seen}, {"PROPFIND"})

    def test_auth_stays_dropped_after_a_cross_host_hop(self):
        op = _Opener(_redirect("u", "https://b.example/1"),
                     _redirect("u", "https://a.example/2"), _Resp())
        self.run_with(op, token="t")
        self.assertEqual([s[2] for s in op.seen], ["Bearer t", None, None])

    def test_out_of_hops(self):
        op = _Opener(*[_redirect("u", "/again") for _ in range(3)])
        with self.assertRaises(snapshot.SnapshotError) as cm:
            self.run_with(op, hops=3)
        self.assertIn("redirect", str(cm.exception))

    def test_3xx_without_location_is_returned(self):
        op = _Opener(_redirect("u", None, code=304))
        self.assertEqual(self.run_with(op)[0], 304)

    def test_redirect_to_file_is_refused(self):
        op = _Opener(_redirect("u", "file:///x"))
        with self.assertRaises(snapshot.SnapshotError) as cm:
            self.run_with(op)
        self.assertIn("refusing redirect", str(cm.exception))
        self.assertEqual(len(op.seen), 1)

    def test_https_to_http_downgrade_is_refused_and_scrubbed(self):
        op = _Opener(_redirect("u", "http://a.example/x?authz=SECRET"))
        with self.assertRaises(snapshot.SnapshotError) as cm:
            self.run_with(op, token="t")
        self.assertIn("refusing redirect", str(cm.exception))
        self.assertNotIn("SECRET", str(cm.exception))

    def test_4xx_is_returned_not_raised(self):
        h = email.message.Message()
        op = _Opener(urllib.error.HTTPError("u", 404, "nf", h, io.BytesIO(b"gone")))
        self.assertEqual(self.run_with(op)[0::2], (404, b"gone"))

    def test_scrub_redacts_authz(self):
        self.assertEqual(snapshot._scrub("GET https://h/x?a=1&authz=abc.def&b=2 failed"),
                         "GET https://h/x?a=1&authz=<redacted>&b=2 failed")

    def test_read_timeout_becomes_snapshot_error(self):
        op = _Opener(_Resp(read_exc=socket.timeout("timed out")))
        with self.assertRaises(snapshot.SnapshotError) as cm:
            self.run_with(op)
        self.assertIn("cannot read https://a.example/dir/x", str(cm.exception))

    def test_error_body_read_failure_becomes_snapshot_error(self):
        class Bad(io.BytesIO):
            def read(self, *a):
                raise ConnectionResetError("reset")
        op = _Opener(urllib.error.HTTPError("u", 500, "e", email.message.Message(), Bad()))
        with self.assertRaises(snapshot.SnapshotError):
            self.run_with(op)


class TestLocalCatalogPairing(unittest.TestCase):
    def setUp(self):
        snapshot._pinned.clear()

    def _root(self, d, doc=None):
        if doc is not None:
            os.makedirs(os.path.join(d, "catalog", "catalog_X"))
            with open(os.path.join(d, "catalog", "catalog_X", "meta.json"), "w") as fh:
                json.dump(doc, fh)
        return env(FDP_STORE_ROOT="file://" + d, FDP_STORE_CATALOG="catalog_X")

    def test_local_root_reads_the_file(self):
        with tempfile.TemporaryDirectory() as d, \
             self._root(d, {"sql_snapshots": {"d3drdb": "d3drdb_P"}}), \
             mock.patch.object(snapshot, "_request") as req:
            self.assertEqual(snapshot.catalog_pairing("d3drdb", token=None), "d3drdb_P")
        req.assert_not_called()

    def test_local_root_missing_file_is_none(self):
        with tempfile.TemporaryDirectory() as d, self._root(d):
            self.assertIsNone(snapshot.catalog_pairing("d3drdb", token=None))

    def test_a_pairing_that_is_not_a_string_is_an_error(self):
        for bad in (7, "", ["x"]):
            with tempfile.TemporaryDirectory() as d, \
                 self._root(d, {"sql_snapshots": {"d3drdb": bad}}):
                with self.assertRaises(snapshot.SnapshotError) as cm:
                    snapshot.catalog_pairing("d3drdb", token=None)
                self.assertIn("catalog_X", str(cm.exception))
                self.assertIn(repr(bad), str(cm.exception))


class TestLocatorLookup(unittest.TestCase):
    def test_finds_the_sql_snapshot_locator_by_tokamak_and_name(self):
        from fdp_schema import Tokamak
        tk = Tokamak(name="dev", locators=[
            {"kind": "sql", "name": "db", "driver": "mssql", "host": "h", "database": "d"},
            {"kind": "sql_snapshot", "name": "db", "base_url": "https://x/y", "id_pattern": "db_*"},
        ])
        with mock.patch("toksearch.sql.mssql._discover_catalogs", return_value={"dev": tk}):
            loc = snapshot.locator_for("dev", "db")
        self.assertEqual(loc.kind, "sql_snapshot")
        self.assertEqual(loc.base_url, "https://x/y")

    def test_names_what_exists_when_missing(self):
        from fdp_schema import Tokamak
        tk = Tokamak(name="dev", locators=[
            {"kind": "sql_snapshot", "name": "other", "base_url": "https://x/y", "id_pattern": "o_*"},
        ])
        with mock.patch("toksearch.sql.mssql._discover_catalogs", return_value={"dev": tk}):
            with self.assertRaises(KeyError) as cm:
                snapshot.locator_for("dev", "db")
        self.assertIn("other", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
