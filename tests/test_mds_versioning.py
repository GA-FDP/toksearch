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

from toksearch.signal.mds import MdsConnectionRegistry


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
