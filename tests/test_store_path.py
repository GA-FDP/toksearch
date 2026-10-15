"""Where a versioned MDSplus tree lives in the store.

Pure construction, tested without a tree, a server, or a catalog. The layout
facts here were verified against the production store on 2026-09-13.

Note these are TestCase methods: tests/testit.py collects with
unittest.TestLoader().discover(), which sees TestCase subclasses only. A
module-level def test_* is imported and silently never run.
"""

import unittest

from toksearch.signal.store_path import (
    SHARED_BRANCHES,
    SHOT_BRANCHES,
    join_tree_path,
    shard_key,
    shared_tree_paths,
    shot_tree_paths,
)


class TestShardKey(unittest.TestCase):
    """The shard key is (tree, number // 1_000_000), rendered <tree>-<n>."""

    def test_a_transp_run_id(self):
        self.assertEqual(shard_key("transp", 2077222602), "transp-2077")

    def test_small_numbers_land_in_shard_zero(self):
        # bci-0 really exists on production: it holds bci's model tree. A
        # shard hit is ordinary for any tree, not a sign the tree is shared.
        self.assertEqual(shard_key("bci", 165920), "bci-0")

    def test_a_file_with_no_number_keys_on_zero(self):
        self.assertEqual(shard_key("aot01", None), "aot01-0")

    def test_the_boundary_is_exactly_one_million(self):
        self.assertEqual(shard_key("t", 999999), "t-0")
        self.assertEqual(shard_key("t", 1000000), "t-1")


class TestShotTreePaths(unittest.TestCase):
    def test_one_entry_per_branch_under_the_version(self):
        got = shot_tree_paths("/mnt/beegfs/data/views", 165920, 2)
        self.assertEqual(got, [
            "/mnt/beegfs/data/views/shots/1659/165920/v2/mdsplus/codes",
            "/mnt/beegfs/data/views/shots/1659/165920/v2/mdsplus/shots",
            "/mnt/beegfs/data/views/shots/1659/165920/v2/mdsplus/usershots",
        ])

    def test_bucket_and_shot_are_zero_padded(self):
        got = shot_tree_paths("/r", 903, 1)
        self.assertEqual(got[0], "/r/shots/0009/000903/v1/mdsplus/codes")

    def test_per_shot_trees_are_flat_within_a_branch(self):
        # archives/ nests by digit group; the store does not. All of a shot's
        # trees share one directory, so the path depends on the shot but not
        # the tree -- which is why no tree name appears here.
        got = shot_tree_paths("/r", 165920, 1)
        self.assertNotIn("bci", " ".join(got))

    def test_a_pelican_root_keeps_its_double_slash(self):
        # pathlib.Path collapses '//' in a URL scheme, so this module must
        # never use it.
        got = shot_tree_paths("pelican://osg-htc.org:443/fdp-d3d/views", 165920, 1)
        self.assertTrue(got[0].startswith("pelican://osg-htc.org:443/"))


class TestSharedTreePaths(unittest.TestCase):
    def test_shard_paths_carry_the_group_level(self):
        got = shared_tree_paths("/r", "transp-2077", 1, "transp")
        self.assertEqual(got, [
            "/r/shared/transp-2077/v1/mdsplus/models/transp",
            "/r/shared/transp-2077/v1/mdsplus/codes/transp",
            "/r/shared/transp-2077/v1/mdsplus/shots/transp",
            "/r/shared/transp-2077/v1/mdsplus/usershots/transp",
        ])

    def test_a_shard_carries_models_and_a_shot_does_not(self):
        # A model tree has no shot to belong to, so it always lands in
        # shared/. Dropping models/ here makes every model tree unreachable.
        self.assertIn("models", SHARED_BRANCHES)
        self.assertNotIn("models", SHOT_BRANCHES)

    def test_a_model_tree_is_reachable_for_an_ordinary_tree(self):
        # bci is a per-shot tree, yet bci_model.tree lives in shared/bci-0/.
        # Verified against production 2026-09-13.
        got = shared_tree_paths("/r", "bci-0", 1, "bci")
        self.assertIn("/r/shared/bci-0/v1/mdsplus/models/bci", got)


class TestJoinTreePath(unittest.TestCase):
    def test_entries_are_semicolon_separated(self):
        self.assertEqual(join_tree_path(["/a", "/b"]), "/a;/b")

    def test_unpinned_keeps_every_entry(self):
        joined = join_tree_path(["/views/a", "/archives/b"])
        self.assertEqual(joined, "/views/a;/archives/b")

    def test_a_pinned_path_never_retains_an_archives_fallback(self):
        # The characteristic failure of this stack is resolving from another
        # tier instead of erroring. If a views path is wrong and archives is
        # still on the search path, the read succeeds with unversioned data
        # and no error at all.
        joined = join_tree_path(["/views/a", "/archives/b"], pinned=True)
        self.assertEqual(joined, "/views/a")

    def test_a_pinned_path_keeps_shared_entries(self):
        joined = join_tree_path(
            ["/d/views/shots/x", "/d/views/shared/y", "/d/archives/z"],
            pinned=True)
        self.assertEqual(joined, "/d/views/shots/x;/d/views/shared/y")

    def test_a_pin_with_nothing_left_is_empty_rather_than_a_fallback(self):
        # Better an unusable path than one that silently reads archives.
        self.assertEqual(join_tree_path(["/archives/b"], pinned=True), "")
