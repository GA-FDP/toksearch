"""B7b stage 0: the old spellings are refused, not ignored.

A rename that silently drops a pin is worse than one that breaks. The run
then reports a catalog it did not read, which is exactly the failure a pin
exists to prevent -- and the shape of the toksearch 2.15.0 defect.

`Pipeline.from_snapshot` is the sharp case: it SURVIVES the rename but means
something else afterwards (a saved-snapshot file, B7b Task 7). A pipeline
written last week that passes a catalog stamp must be told, not quietly
unpinned.
"""

import contextlib
import os
import unittest
from unittest import mock

from toksearch import Pipeline
from toksearch.record import InvalidRecordField
from toksearch.signal import store_catalog


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


class TestTheOldEnvironmentVariable(unittest.TestCase):
    def test_it_raises_naming_both_spellings(self):
        with env(FDP_STORE_ROOT="/some/root",
                 FDP_STORE_SNAPSHOT="catalog_OLD", FDP_STORE_CATALOG=None):
            with self.assertRaises(store_catalog.RenamedVariable) as cm:
                Pipeline([1]).compute_serial()
        msg = str(cm.exception)
        self.assertIn("FDP_STORE_SNAPSHOT", msg)
        self.assertIn("FDP_STORE_CATALOG", msg)

    def test_the_new_one_works(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_SNAPSHOT=None,
                 FDP_STORE_CATALOG="catalog_NEW"):
            Pipeline([1]).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_CATALOG"], "catalog_NEW")


class TestTheOldRecordField(unittest.TestCase):
    def test_a_record_naming_snapshot_is_refused(self):
        # {'shot': N, 'snapshot': 'catalog_...'} meant the catalog. Left
        # working, such a record would silently stop pinning.
        # Refused where the record is built, which is Pipeline construction
        # -- earlier than compute, and the earliest point it can be caught.
        with self.assertRaises(InvalidRecordField) as cm:
            Pipeline([{"shot": 1, "snapshot": "catalog_OLD"}])
        msg = str(cm.exception)
        self.assertIn("catalog", msg)
        self.assertIn("from_snapshot", msg)

    def test_a_record_naming_catalog_is_accepted(self):
        recs = Pipeline([{"shot": 1, "catalog": "catalog_NEW"}]).compute_serial()
        self.assertEqual(recs[0]["catalog"], "catalog_NEW")


class TestTheOldClassmethod(unittest.TestCase):
    def test_from_snapshot_refuses_a_catalog_stamp(self):
        with self.assertRaises(ValueError) as cm:
            Pipeline.from_snapshot("catalog_20260907T232802Z")
        self.assertIn("from_catalog", str(cm.exception))

    def test_from_catalog_does_what_from_snapshot_used_to(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_CATALOG=None,
                 FDP_STORE_SNAPSHOT=None):
            Pipeline.from_catalog("catalog_PINNED", [1]).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_CATALOG"], "catalog_PINNED")

    def test_from_catalog_still_treats_latest_as_no_pin(self):
        with env(FDP_STORE_ROOT="/some/root", FDP_STORE_CATALOG=None,
                 FDP_STORE_SNAPSHOT=None), \
             mock.patch.object(store_catalog, "_resolve",
                               return_value="catalog_NEWEST"):
            Pipeline.from_catalog("latest", [1]).compute_serial()
            self.assertEqual(os.environ["FDP_STORE_CATALOG"], "catalog_NEWEST")


if __name__ == "__main__":
    unittest.main()
