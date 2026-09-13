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
