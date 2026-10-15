"""Where a versioned MDSplus tree lives in the per-shot versioned store.

Pure construction: no I/O, no MDSplus, no network. Versions come from
``ptdata.StoreIndex``; this module turns a resolved version into the search
path MDSplus needs.

It exists because MDSplus's own substitution grammar cannot express ``v<N>``.
The grammar reaches everything else -- ``~t`` is the tree name, ``~f~e~d~c``
the bucket, ``~f~e~d~c~b~a`` the padded shot -- but the version differs per
shot, and a static template has no slot for it. That single limitation is the
whole reason this module exists.

Paths are built with f-strings, never ``pathlib``: ``Path()`` collapses the
``//`` in a URL scheme, and these are ``pelican://`` URLs as often as they are
filesystem paths.
"""

# A shot has three branches. A shard has four: a model tree has no shot to
# belong to, so it always lands in shared/ -- bci_model.tree lives under
# shared/bci-0/v1/mdsplus/models/bci/ even though bci is an ordinary per-shot
# tree. Verified against production 2026-09-13. These four match the entries
# the archives locator in d3d.yaml has always carried.
SHOT_BRANCHES = ("codes", "shots", "usershots")
SHARED_BRANCHES = ("models", "codes", "shots", "usershots")

# A shard spans a million numbers.
SHARD_SPAN = 1_000_000


def shard_key(treename, number):
    """The shared shard a tree's file belongs to.

    Keyed ``(tree, number // 1_000_000)``, which buys locality rather than
    balance: run ids rise with time, so a night's arrivals land in the newest
    shard for their tree and old shards are never rewritten. A file carrying
    no number keys on 0.

    Note that every tree has a shard, not only ``transp``/``ddb_*``/``tip-*``:
    an ordinary per-shot tree's *model* has no shot, so it lives in shard 0.
    A hit here is the ordinary case, not evidence the tree is a shared one.
    """
    n = 0 if number is None else int(number) // SHARD_SPAN
    return f"{treename}-{n}"


def shot_tree_paths(views_root, shot, version):
    """Search-path entries for one shot at one version, one per branch.

    Per-shot trees are flat within a branch, where ``archives/`` nests by
    digit group -- so all of a shot's trees share one directory and the path
    depends on the shot but not on the tree.
    """
    shot = int(shot)
    bucket = f"{shot // 100:04d}"
    stem = f"{views_root}/shots/{bucket}/{shot:06d}/v{int(version)}/mdsplus"
    return [f"{stem}/{branch}" for branch in SHOT_BRANCHES]


def shared_tree_paths(views_root, shard, version, group):
    """Search-path entries for a shared shard.

    Shards carry one level more than per-shot trees: the tree group sits below
    the branch, so a file is at ``<branch>/<group>/<group>_<number>.tree``.
    """
    stem = f"{views_root}/shared/{shard}/v{int(version)}/mdsplus"
    return [f"{stem}/{branch}/{group}" for branch in SHARED_BRANCHES]


def join_tree_path(entries, pinned=False):
    """Render entries as MDSplus expects, dropping archives under a pin.

    A pin is a guarantee. Leaving an ``archives/`` entry on the search path
    behind a pinned store entry would let a wrong path resolve from the old
    tier and return unversioned data with no error -- this stack's
    characteristic failure. When pinned, only store paths survive, even if
    that leaves nothing: an unusable path fails loudly, which a silent
    fallback does not.
    """
    if pinned:
        entries = [e for e in entries if "/views/" in e]
    return ";".join(entries)
