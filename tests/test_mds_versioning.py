"""Version awareness in the MDSplus connection registry.

These use a fake connection rather than a live relay. What is under test is
the registry's decision to reopen, which is where a dropped version would
silently serve the wrong data -- no exception, no warning, just an earlier
version's bytes behind the right name.

The behaviour they encode was measured against the production origin; see
d3d-origin-admin/tests/integration/mds-session-repin.py. An mdsip session
re-reads its path on every open of a tree it does not already hold, only an
already-open (tree, shot) is latched, and TreeClose releases that latch.

TestCase methods, not module-level functions: tests/testit.py collects with
unittest.TestLoader().discover(), which sees TestCase subclasses only.
"""

import unittest

from unittest import mock

from toksearch.signal.mds import (
    MdsConnectionRegistry,
    MdsTreePath,
    MdsTreeRegistry,
)


class FakeConnection:
    """Records the calls a real MDSplus.Connection would receive."""

    def __init__(self):
        self.calls = []

    def get(self, expr, *args):
        self.calls.append(("get", expr, args))
        return ""

    def openTree(self, tree, shot):
        self.calls.append(("openTree", tree, shot))

    def closeAllTrees(self):
        self.calls.append(("closeAllTrees",))


class RegistryTest(unittest.TestCase):
    def setUp(self):
        # A __new__ singleton with no reset(): clear the map on the instance.
        self.registry = MdsConnectionRegistry()
        self.registry._connection_map.clear()
        self.conn = FakeConnection()
        self.registry._connection_map["srv"] = self.conn

    def tearDown(self):
        self.registry._connection_map.clear()

    def kinds(self):
        return [c[0] for c in self.conn.calls]

    def opens(self):
        return [c for c in self.conn.calls if c[0] == "openTree"]


class TestVersionedOpens(RegistryTest):
    def test_same_tree_shot_and_version_opens_once(self):
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.assertEqual(len(self.opens()), 1)

    def test_a_changed_version_forces_a_reopen(self):
        # The whole point. MDSplus serves an already-open tree regardless of
        # the current path, so without this the second read returns v1's data
        # while the caller believes it asked for v2.
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.open_tree("srv", "bci", 165920, version=2, tree_path="/p2")
        self.assertEqual(len(self.opens()), 2)

    def test_a_changed_version_closes_before_reopening(self):
        # Measured: an already-open tree+shot ignores a later setenv, and
        # TreeClose is what releases it.
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.open_tree("srv", "bci", 165920, version=2, tree_path="/p2")

        kinds = self.kinds()
        self.assertIn("closeAllTrees", kinds)
        # The close must precede the LAST open, not merely appear somewhere.
        last_open = len(kinds) - 1 - kinds[::-1].index("openTree")
        self.assertLess(kinds.index("closeAllTrees"), last_open)

    def test_the_path_is_set_before_the_open(self):
        # After the first open the tree is cached, so a setenv that arrives
        # late is silently ignored. Order is the whole mechanism.
        self.registry.open_tree("srv", "bci", 165920, version=2, tree_path="/p2")

        kinds = self.kinds()
        self.assertLess(kinds.index("get"), kinds.index("openTree"))
        self.assertIn("/p2", str(self.conn.calls[kinds.index("get")]))

    def test_a_different_shot_reopens_without_closing(self):
        # The open-tree cache is keyed by (name, shot), so a new shot needs no
        # close -- and paying one per shot would cost a round trip per record.
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.conn.calls.clear()
        self.registry.open_tree("srv", "bci", 165921, version=1, tree_path="/p1b")

        self.assertNotIn("closeAllTrees", self.kinds())
        self.assertEqual(len(self.opens()), 1)

    def test_a_different_tree_reopens_without_closing(self):
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.conn.calls.clear()
        self.registry.open_tree("srv", "bes", 165920, version=1, tree_path="/p1")

        self.assertNotIn("closeAllTrees", self.kinds())
        self.assertEqual(len(self.opens()), 1)


class TestUnversionedCallersAreUnaffected(RegistryTest):
    """Every existing caller passes no version and must keep working."""

    def test_no_version_still_caches_the_open(self):
        self.registry.open_tree("srv", "bci", 165920)
        self.registry.open_tree("srv", "bci", 165920)
        self.assertEqual(len(self.opens()), 1)

    def test_no_tree_path_sends_no_setenv(self):
        # An origin with no store must see exactly the traffic it saw before.
        self.registry.open_tree("srv", "bci", 165920)
        self.assertEqual([c for c in self.conn.calls if c[0] == "get"], [])

    def test_no_version_never_closes(self):
        self.registry.open_tree("srv", "bci", 165920)
        self.registry.open_tree("srv", "bci", 165921)
        self.assertNotIn("closeAllTrees", self.kinds())


class TestTheMarkerTracksWhatIsOpen(RegistryTest):
    def test_invalidate_forces_the_next_open(self):
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.invalidate_open_tree("srv")
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.assertEqual(len(self.opens()), 2)

    def test_close_all_trees_forgets_the_version(self):
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.close_all_trees("srv")
        self.conn.calls.clear()
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.assertEqual(len(self.opens()), 1)


class LocalRegistryTest(unittest.TestCase):
    """MdsTreeRegistry caches trees by (treename, shot), the same latch the
    connection registry had: two versions of a shot are different data behind
    one name, so a pinned re-read was served whatever was opened first.
    """

    def setUp(self):
        self.registry = MdsTreeRegistry()
        self.registry.reset()          # clears _tree_map on the singleton
        self.opened = []

    def tearDown(self):
        self.registry.reset()

    def fake_open(self, treename, shot, treepath):
        self.opened.append((treename, shot, dict(treepath.paths)))
        return mock.MagicMock()

    def patched(self):
        return mock.patch.object(
            self.registry, "_open_tree", side_effect=self.fake_open)


class TestLocalVersionedOpens(LocalRegistryTest):
    def test_a_changed_version_reopens_the_tree(self):
        with self.patched():
            self.registry.open_tree(
                "bci", 165920, treepath=MdsTreePath(bci="/v1"), version=1)
            self.registry.open_tree(
                "bci", 165920, treepath=MdsTreePath(bci="/v2"), version=2)

        self.assertEqual(len(self.opened), 2)
        self.assertEqual(self.opened[0][2]["bci"], "/v1")
        self.assertEqual(self.opened[1][2]["bci"], "/v2")

    def test_the_same_version_is_served_from_cache(self):
        with self.patched():
            self.registry.open_tree(
                "bci", 165920, treepath=MdsTreePath(bci="/v1"), version=1)
            self.registry.open_tree(
                "bci", 165920, treepath=MdsTreePath(bci="/v1"), version=1)
        self.assertEqual(len(self.opened), 1)

    def test_versions_of_one_shot_coexist(self):
        with self.patched():
            a = self.registry.open_tree("bci", 165920, version=1)
            b = self.registry.open_tree("bci", 165920, version=2)
            again = self.registry.open_tree("bci", 165920, version=1)
        self.assertIsNot(a, b)
        self.assertIs(a, again)


class TestLocalUnversionedCallersAreUnaffected(LocalRegistryTest):
    def test_no_version_still_caches(self):
        with self.patched():
            self.registry.open_tree("bci", 165920)
            self.registry.open_tree("bci", 165920)
        self.assertEqual(len(self.opened), 1)

    def test_get_tree_still_takes_two_arguments(self):
        # tests/test_mds.py calls it this way, and so may anything else.
        self.assertIsNone(self.registry._get_tree("blah", 123))


class TestLocalClosing(LocalRegistryTest):
    def test_closing_without_a_version_sweeps_every_version(self):
        # cleanup_shot wants the shot gone, and does not know which versions
        # were opened for it. Lookup is exact; closing defaults to sweeping.
        with self.patched():
            self.registry.open_tree("bci", 165920, version=1)
            self.registry.open_tree("bci", 165920, version=2)
            self.registry.close_tree("bci", 165920)

            self.registry.open_tree("bci", 165920, version=1)
        self.assertEqual(len(self.opened), 3)

    def test_closing_one_version_leaves_the_others(self):
        with self.patched():
            self.registry.open_tree("bci", 165920, version=1)
            self.registry.open_tree("bci", 165920, version=2)
            self.registry.close_tree("bci", 165920, version=2)

            self.registry.open_tree("bci", 165920, version=1)   # still cached
            self.registry.open_tree("bci", 165920, version=2)   # reopened
        self.assertEqual(len(self.opened), 3)

    def test_close_all_trees_handles_versioned_keys(self):
        # The keys are tuples now; iterating them as bare shots would raise.
        with self.patched():
            self.registry.open_tree("bci", 165920, version=1)
            self.registry.open_tree("bes", 165921, version=2)
            self.registry.close_all_trees()

            self.registry.open_tree("bci", 165920, version=1)
        self.assertEqual(len(self.opened), 3)


# ---------------------------------------------------------------------------
# Carrying a pin from the record to the open.
#
# These assert on the version actually used, never on the absence of an
# error. The toksearch_d3d>=0.13.0 / toksearch>=2.12.0 floor is unguarded: a
# dropped pin returns unversioned data with errors == {}, so "no error" is
# evidence of nothing at all.
# ---------------------------------------------------------------------------

VIEWS = "/mnt/beegfs/data/views"


class FakeLookup:
    def __init__(self, version=None, snapshot="catalog_X", miss="NoMiss",
                 detail=""):
        self.found = version is not None
        self.version = version or 0
        self.snapshot = snapshot
        self.miss = miss
        self.detail = detail


class FakeIndex:
    """Stands in for ptdata.StoreIndex, recording what it was asked."""

    def __init__(self, shots=None, shards=None):
        self.shots = shots if shots is not None else {165920: 1}
        self.shards = shards if shards is not None else {"bci-0": 4}
        self.calls = []

    def resolve_version(self, shot, version=None, snapshot=None):
        self.calls.append(("shot", shot, version, snapshot))
        if snapshot == "bogus":
            return FakeLookup(miss="SnapshotMissing", detail="no such snapshot")
        if version is not None:
            if version in (1, 2):
                return FakeLookup(version=version)
            return FakeLookup(miss="VersionMissing", detail="never minted")
        got = self.shots.get(shot)
        return FakeLookup(version=got) if got else FakeLookup(
            miss="ShotNotInCatalog")

    def resolve_shared_version(self, shard, version=None, snapshot=None):
        self.calls.append(("shard", shard, version, snapshot))
        got = self.shards.get(shard)
        return FakeLookup(version=got) if got else FakeLookup(
            miss="ShotNotInCatalog")


class EnvConnection(FakeConnection):
    """A connection whose sandbox declares a views root."""

    def __init__(self, views_root=VIEWS, archives=""):
        super().__init__()
        self.views_root = views_root
        self.archives = archives

    def get(self, expr, *args):
        self.calls.append(("get", expr, args))
        if "fdp_views_root" in expr:
            return self.views_root
        if "default_tree_path" in expr:
            return self.archives
        return ""


class PinTest(unittest.TestCase):
    def setUp(self):
        from toksearch.signal import mds as mds_mod

        self.mds = mds_mod
        self.registry = MdsConnectionRegistry()
        self.registry._connection_map.clear()
        self.conn = EnvConnection()
        self.registry._connection_map["fdp://host/mdsip"] = self.conn

        self.index = FakeIndex()
        self._patch = mock.patch.object(
            mds_mod, "_store_index", return_value=self.index)
        self._patch.start()
        self.addCleanup(self._patch.stop)

        # The CLIENT's catalog root. Deliberately different from the sandbox
        # path the origin declares -- conflating the two is the bug this
        # fixture exists to keep out.
        self._env = mock.patch.dict(
            "os.environ",
            {"FDP_STORE_ROOT": "pelican://osg-htc.org:443/fdp-d3d"},
            clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

        self.seen = []
        self._open = mock.patch.object(
            MdsConnectionRegistry, "open_tree",
            side_effect=self.capture_open)
        self._open.start()
        self.addCleanup(self._open.stop)

    def tearDown(self):
        self.registry._connection_map.clear()

    def capture_open(self, server, treename, shot, version=None, tree_path=None):
        self.seen.append(dict(server=server, treename=treename, shot=shot,
                              version=version, tree_path=tree_path))
        raise RuntimeError("stop here: the open itself is not under test")

    def signal(self, treename="bci"):
        return self.mds.MdsRemoteSignal(
            r"\bci::top", treename, server="fdp://host/mdsip")

    def gather(self, **pin):
        with self.assertRaises(Exception):
            self.signal().gather(165920, **pin)


class TestThePinReachesTheOpen(PinTest):
    def test_a_records_version_selects_the_version_and_path(self):
        self.gather(version=2)
        self.assertEqual(self.seen[-1]["version"], 2)
        self.assertIn("/165920/v2/", self.seen[-1]["tree_path"])

    def test_an_unpinned_record_resolves_the_latest(self):
        self.gather()
        self.assertEqual(self.seen[-1]["version"], 1)
        self.assertIn("/165920/v1/", self.seen[-1]["tree_path"])

    def test_a_pinned_path_carries_no_archives_entry(self):
        self.gather(version=2)
        self.assertNotIn("archives", self.seen[-1]["tree_path"])

    def test_the_shard_entry_is_appended(self):
        # bci-0 holds bci's model tree, so it belongs on the search path even
        # for an ordinary per-shot tree.
        self.gather(version=2)
        self.assertIn("/shared/bci-0/v4/", self.seen[-1]["tree_path"])

    def test_the_shots_pin_does_not_apply_to_the_shard(self):
        # A shard has its own version chain. Passing the shot's version here
        # would pin an unrelated artifact to a number that means nothing for
        # it -- only the snapshot is shared.
        self.gather(version=2, snapshot="catalog_Y")
        shard_calls = [c for c in self.index.calls if c[0] == "shard"]
        self.assertTrue(shard_calls)
        self.assertIsNone(shard_calls[-1][2])            # version
        self.assertEqual(shard_calls[-1][3], "catalog_Y")  # snapshot


class TestAnUnsatisfiablePinFailsLoudly(PinTest):
    def test_a_version_never_minted_raises(self):
        with self.assertRaises(self.mds.StoreVersionError):
            self.signal().gather(165920, version=99)

    def test_an_unknown_snapshot_raises(self):
        with self.assertRaises(self.mds.StoreVersionError):
            self.signal().gather(165920, snapshot="bogus")

    def test_a_failed_pin_never_opens_anything(self):
        # The failure that matters is not raising late -- it is opening an
        # unpinned tree and returning its data as though it were the pin.
        try:
            self.signal().gather(165920, version=99)
        except Exception:
            pass
        self.assertEqual(self.seen, [])

    def test_a_pin_against_an_origin_with_no_store_raises(self):
        self.conn.views_root = ""
        with self.assertRaises(self.mds.StoreVersionError):
            self.signal().gather(165920, version=1)


class TestOriginsWithoutAStoreAreUnchanged(PinTest):
    def test_an_unpinned_read_with_no_store_opens_as_before(self):
        self.conn.views_root = ""
        self.gather()
        self.assertIsNone(self.seen[-1]["version"])
        self.assertIsNone(self.seen[-1]["tree_path"])

    def test_no_record_at_all_opens_as_before(self):
        self.conn.views_root = ""
        with self.assertRaises(Exception):
            self.signal().gather(165920)
        self.assertIsNone(self.seen[-1]["version"])

    def test_a_shot_absent_from_the_catalog_is_not_an_error(self):
        # Unpinned and unknown: fall back rather than fail. A store that has
        # not minted a shot yet must not break reads that worked before.
        self.gather()
        self.seen.clear()
        with self.assertRaises(Exception):
            self.signal().gather(999999)
        self.assertIsNone(self.seen[-1]["version"])
        self.assertIsNone(self.seen[-1]["tree_path"])


class TestTheRetryPathCarriesTheRecord(PinTest):
    def test_the_retry_after_an_mdsplus_error_keeps_the_pin(self):
        # gather() retries once on MDSplusERROR against a fresh connection.
        # Miss the record there and a pin is dropped exactly when a
        # connection has been re-dialled -- the hardest case to notice.
        calls = {"n": 0}

        def fail_once_then_capture(server, treename, shot, version=None,
                                   tree_path=None):
            calls["n"] += 1
            if calls["n"] == 1:
                raise self.mds.MDSplusERROR("%MDSPLUS-E-ERROR, Error")
            self.seen.append(dict(version=version, tree_path=tree_path))
            raise RuntimeError("stop")

        self._open.stop()
        with mock.patch.object(MdsConnectionRegistry, "open_tree",
                               side_effect=fail_once_then_capture):
            with mock.patch.object(MdsConnectionRegistry, "disconnect"):
                self.registry._connection_map["fdp://host/mdsip"] = self.conn
                with self.assertRaises(Exception):
                    self.signal().gather(165920, version=2)
        self._open.start()

        self.assertEqual(calls["n"], 2)
        self.assertEqual(self.seen[-1]["version"], 2)


class TestTheArchivesFallback(PinTest):
    """A setenv replaces the whole search path, so archives has to be carried.

    Without this, an unpinned read of a tree the store has not absorbed yet
    would start failing the moment its shot gained a version -- a regression
    caused entirely by resolving successfully.
    """

    ARCHIVES = "/mnt/beegfs/data/archives/mdsplus/codes/~t;/x/archives/shots/~t"

    def setUp(self):
        super().setUp()
        self.conn.archives = self.ARCHIVES

    def test_an_unpinned_read_keeps_the_archives_entries(self):
        self.gather()
        self.assertIn("/archives/", self.seen[-1]["tree_path"])

    def test_a_pinned_read_drops_them(self):
        self.gather(version=2)
        self.assertNotIn("/archives/", self.seen[-1]["tree_path"])

    def test_the_store_entries_come_first_either_way(self):
        # MDSplus walks the path in order, so the store must be consulted
        # before archives or an unpinned read answers from the old tier.
        self.gather()
        path = self.seen[-1]["tree_path"]
        self.assertLess(path.index("/views/"), path.index("/archives/"))


class TestTheStoreProbeIsCheap(PinTest):
    """Discovering whether an origin has a store costs round trips, so it has
    to happen once per connection rather than once per read.

    A pipeline fetches several signals per shot across thousands of shots on
    one connection. Re-probing per gather would add two round trips to every
    one of them, which is the kind of cost that only shows up in production.
    """

    def probes(self):
        return [c for c in self.conn.calls
                if c[0] == "get" and "getenv" in str(c[1])]

    def test_the_probe_happens_once_across_many_reads(self):
        for shot in (165920, 165920, 165921):
            try:
                self.signal().gather(shot)
            except Exception:
                pass
        # Two variables, read together, exactly once.
        self.assertEqual(len(self.probes()), 2)

    def test_a_fresh_connection_probes_again(self):
        try:
            self.signal().gather(165920)
        except Exception:
            pass
        before = len(self.probes())

        fresh = EnvConnection()
        self.registry._connection_map["fdp://host/mdsip"] = fresh
        try:
            self.signal().gather(165920)
        except Exception:
            pass

        self.assertEqual(before, 2)
        self.assertEqual(
            len([c for c in fresh.calls
                 if c[0] == "get" and "getenv" in str(c[1])]), 2)


# ---------------------------------------------------------------------------
# The pelican:// half. A local read has no session, so there is nothing to
# getenv -- the store root comes from the environment, which `fdp env`
# composes from the device locator.
# ---------------------------------------------------------------------------

PELICAN_ROOT = "pelican://osg-htc.org:443/fdp-d3d"


class LocalPinTest(unittest.TestCase):
    def setUp(self):
        from toksearch.signal import mds as mds_mod

        self.mds = mds_mod
        self.registry = MdsTreeRegistry()
        self.registry.reset()

        self.index = FakeIndex()
        self._patch = mock.patch.object(
            mds_mod, "_store_index", return_value=self.index)
        self._patch.start()
        self.addCleanup(self._patch.stop)

        self.seen = []
        self._open = mock.patch.object(
            MdsTreeRegistry, "open_tree", side_effect=self.capture_open)
        self._open.start()
        self.addCleanup(self._open.stop)

        self._env = mock.patch.dict(
            "os.environ", {"FDP_STORE_ROOT": PELICAN_ROOT}, clear=False)
        self._env.start()
        self.addCleanup(self._env.stop)

    def tearDown(self):
        self.registry.reset()

    def capture_open(self, treename, shot, treepath=None, version=None):
        paths = dict(treepath.paths) if treepath is not None else {}
        self.seen.append(dict(treename=treename, shot=shot, version=version,
                              paths=paths))
        raise RuntimeError("stop here: the open itself is not under test")

    def signal(self, treename="bci"):
        return self.mds.MdsLocalSignal(r"\bci::top", treename)

    def gather(self, **pin):
        with self.assertRaises(Exception):
            self.signal().gather(165920, **pin)

    def path(self):
        return self.seen[-1]["paths"].get("bci", "")


class TestLocalPinReachesTheOpen(LocalPinTest):
    def test_a_records_version_selects_the_version_and_path(self):
        self.gather(version=2)
        self.assertEqual(self.seen[-1]["version"], 2)
        self.assertIn("/165920/v2/", self.path())

    def test_the_path_is_a_pelican_url(self):
        # Path() collapses '//' in a scheme, so a mangled root shows up here.
        self.gather(version=2)
        self.assertTrue(self.path().startswith("pelican://osg-htc.org:443/"))

    def test_an_unsatisfiable_pin_raises(self):
        with self.assertRaises(self.mds.StoreVersionError):
            self.signal().gather(165920, version=99)

    def test_no_store_configured_behaves_as_before(self):
        with mock.patch.dict("os.environ", {"FDP_STORE_ROOT": ""}, clear=False):
            self.gather()
        self.assertIsNone(self.seen[-1]["version"])

    def test_a_pin_with_no_store_configured_raises(self):
        with mock.patch.dict("os.environ", {"FDP_STORE_ROOT": ""}, clear=False):
            with self.assertRaises(self.mds.StoreVersionError):
                self.signal().gather(165920, version=1)

    def test_an_unpinned_read_keeps_the_ambient_archives_path(self):
        # Setting <tree>_path overrides default_tree_path for that tree, so
        # the ambient value has to be carried or the fallback disappears.
        with mock.patch.dict(
                "os.environ",
                {"default_tree_path": "/x/archives/codes/~t"}, clear=False):
            self.gather()
        self.assertIn("/archives/", self.path())

    def test_a_pinned_read_drops_the_ambient_archives_path(self):
        with mock.patch.dict(
                "os.environ",
                {"default_tree_path": "/x/archives/codes/~t"}, clear=False):
            self.gather(version=2)
        self.assertNotIn("/archives/", self.path())


class TestAMismatchedPtdataSaysSo(unittest.TestCase):
    """toksearch does not depend on ptdata, so the pair can be mismatched.

    Store support needs ptdata >= 2.7.0 for StoreIndex. Without a named
    error, a user on an older ptdata sees a bare ImportError from a lazy
    import three frames down and has nothing to act on.
    """

    def setUp(self):
        from toksearch.signal import mds as mds_mod
        self.mds = mds_mod
        mds_mod._STORE_INDEX.clear()
        self.addCleanup(mds_mod._STORE_INDEX.clear)

    def test_a_missing_store_index_names_the_version_needed(self):
        import builtins

        real_import = builtins.__import__

        def no_store_index(name, *args, **kw):
            if name == "ptdata":
                raise ImportError("cannot import name 'StoreIndex'")
            return real_import(name, *args, **kw)

        with mock.patch.object(builtins, "__import__", no_store_index):
            with self.assertRaises(self.mds.StoreVersionError) as caught:
                self.mds._store_index("/some/root")

        self.assertIn("2.7.0", str(caught.exception))


class TestStoreVersionErrorIsCatchable(unittest.TestCase):
    """The docs tell users to catch it, so it has to be reachable by name."""

    def test_it_is_exported_from_the_package_root(self):
        import toksearch
        from toksearch.signal.mds import StoreVersionError

        self.assertIs(toksearch.StoreVersionError, StoreVersionError)


class TestTheSetenvIsNotRepeated(RegistryTest):
    """Several trees at one shot resolve to the same path.

    Re-sending an identical setenv costs a round trip per tree per shot for
    nothing. The tree still has to be opened -- that is what the marker is
    for -- but the path only has to be sent when it changes.
    """

    def setenvs(self):
        return [c for c in self.conn.calls
                if c[0] == "get" and "setenv" in str(c[1])]

    def test_a_second_tree_on_the_same_path_sends_no_setenv(self):
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.open_tree("srv", "bes", 165920, version=1, tree_path="/p1")

        self.assertEqual(len(self.setenvs()), 1)
        self.assertEqual(len(self.opens()), 2)   # both trees still opened

    def test_a_changed_path_is_sent(self):
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.open_tree("srv", "bci", 165921, version=1, tree_path="/p2")
        self.assertEqual(len(self.setenvs()), 2)

    def test_a_changed_version_on_one_shot_resends(self):
        self.registry.open_tree("srv", "bci", 165920, version=1, tree_path="/p1")
        self.registry.open_tree("srv", "bci", 165920, version=2, tree_path="/p2")
        self.assertEqual(len(self.setenvs()), 2)


class TestTheWrapperForwardsThePin(unittest.TestCase):
    """MdsSignal is what users instantiate; MdsRemoteSignal is an
    implementation detail it delegates to.

    Every other test in this file exercises the inner class directly, and all
    of them passed while the wrapper accepted the pin and threw it away -- so
    a pinned pipeline read the latest version and returned perfectly good data
    for the wrong one, with no error. Test the class the user reaches for, not
    only the one that does the work.
    """

    def signal(self, location="fdp://host/mdsip"):
        from toksearch.signal.mds import MdsSignal
        return MdsSignal(r"\bci::top:denr0", "bci", location=location)

    def test_the_pin_reaches_the_delegate(self):
        sig = self.signal()
        seen = {}

        def capture(shot, version=None, snapshot=None):
            seen.update(shot=shot, version=version, snapshot=snapshot)
            return {"data": None}

        with mock.patch.object(sig.sig, "gather", side_effect=capture):
            sig.gather(165920, version=3, snapshot="catalog_X")

        self.assertEqual(seen["version"], 3, "the pin was dropped")
        self.assertEqual(seen["snapshot"], "catalog_X")

    def test_no_pin_still_works(self):
        sig = self.signal()
        with mock.patch.object(sig.sig, "gather",
                               return_value={"data": None}) as inner:
            sig.gather(165920)
        inner.assert_called_once_with(165920, version=None, snapshot=None)

    def test_the_local_wrapper_forwards_too(self):
        sig = self.signal(location="/some/tree/path")
        seen = {}

        def capture(shot, version=None, snapshot=None):
            seen.update(version=version)
            return {"data": None}

        with mock.patch.object(sig.sig, "gather", side_effect=capture):
            sig.gather(165920, version=3)

        self.assertEqual(seen["version"], 3, "the pin was dropped")


class TestTheTwoRootsAreNotTheSameThing(PinTest):
    """The catalog root and the views root are different, and conflating them
    fails in the most misleading way available.

    The catalog is read by THIS client. The tree path is read by whoever opens
    the tree -- over fdp:// that is the origin's sandbox, whose filesystem the
    client cannot see. Deriving one from the other made the client try to list
    the origin's disk: pinned reads then raised for a reason unrelated to the
    pin, and unpinned reads fell back to archives having resolved nothing.

    That state passes a naive acceptance test, because the controls still
    "error" and the unpinned cases still "return data". It was found only by
    reading the error text.
    """

    def test_the_catalog_is_read_from_the_client_root(self):
        self.gather(version=2)
        # FakeIndex is handed whatever root _store_index was called with; the
        # patch records the call rather than the value, so assert on the path
        # that came out instead: it must name the SANDBOX root, not the
        # client's.
        self.assertIn("/mnt/beegfs/data/views/shots/", self.seen[-1]["tree_path"])

    def test_a_sandbox_path_is_never_used_as_a_catalog_root(self):
        from toksearch.signal import mds as mds_mod

        seen_roots = []
        with mock.patch.object(mds_mod, "_store_index",
                               side_effect=lambda r: seen_roots.append(r) or self.index):
            self.gather(version=2)

        self.assertTrue(seen_roots, "the resolver was never consulted")
        for root in seen_roots:
            self.assertFalse(
                root.startswith("/mnt/beegfs"),
                f"resolved the catalog from the origin's filesystem: {root}")

    def test_a_views_root_without_a_catalog_root_is_no_store(self):
        # The origin declares where its store lives, but this client has not
        # been told where to READ it from. A pin cannot be honoured.
        from toksearch.signal.mds import _resolve_store_path, StoreVersionError
        from toksearch.record import Record

        rec = Record.from_dict({"shot": 165920, "version": 2})
        with self.assertRaises(StoreVersionError):
            _resolve_store_path("bci", 165920, 2, None,
                                "", "/mnt/beegfs/data/views", "", "origin")

        # ... and an unpinned read is unchanged rather than broken.
        self.assertEqual(
            _resolve_store_path("bci", 165920, None, None,
                                "", "/mnt/beegfs/data/views", "", "origin"),
            (None, None))
