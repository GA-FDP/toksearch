"""Naming a version when fetching, and what happens to signals that cannot.

A pin is a locating coordinate -- which bytes to read -- so it travels as two
plain values, version and snapshot. Signals never see a Record: the pipeline
owns that type and extracts from it, which keeps a data-fetching strategy
from having to know the shape of the pipeline's rows.

The compatibility matrix below is the load-bearing part. toksearch_mast 0.1.0
declares gather(self, shot), and an earlier attempt at threading the pin
through as a record broke it outright with a TypeError -- in a blessed set,
undetected, because toksearch's tests do not exercise MAST.

TestCase methods: tests/testit.py collects by unittest discovery.
"""

import unittest

from toksearch.signal.signal import Signal


class _Base(Signal):
    def cleanup_shot(self, shot):
        pass

    def cleanup(self):
        pass

    def cleanup_shot_key(self):
        return (type(self).__name__, id(self))


class LegacySignal(_Base):
    """Exactly the shape toksearch_mast 0.1.0 ships."""

    def gather(self, shot):
        return {"data": [1, 2, 3]}


class InterimSignal(_Base):
    """The record= shape toksearch_d3d 0.13.1 ships."""

    def gather(self, shot, record=None):
        return {"data": [4, 5]}


class ModernSignal(_Base):
    def __init__(self):
        super().__init__()
        self.seen = []

    def gather(self, shot, version=None, snapshot=None):
        self.seen.append((shot, version, snapshot))
        return {"data": [1]}


class KwargsSignal(_Base):
    """Accepts anything. A signal delegating to something else may do this."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def gather(self, shot, **kw):
        self.seen.append(kw)
        return {"data": [1]}


class TestThePinReachesGather(unittest.TestCase):
    def setUp(self):
        self.sig = ModernSignal()

    def test_a_version(self):
        self.sig.fetch(165920, version=2)
        self.assertEqual(self.sig.seen[-1], (165920, 2, None))

    def test_a_snapshot(self):
        self.sig.fetch(165920, snapshot="catalog_X")
        self.assertEqual(self.sig.seen[-1], (165920, None, "catalog_X"))

    def test_both(self):
        self.sig.fetch(165920, version=3, snapshot="catalog_X")
        self.assertEqual(self.sig.seen[-1], (165920, 3, "catalog_X"))

    def test_no_pin_passes_none(self):
        self.sig.fetch(165920)
        self.assertEqual(self.sig.seen[-1], (165920, None, None))

    def test_a_version_of_zero_is_still_a_pin(self):
        # Falsy but meaningful; `if version:` would drop it.
        self.sig.fetch(165920, version=0)
        self.assertEqual(self.sig.seen[-1], (165920, 0, None))

    def test_no_record_object_is_involved(self):
        # The point of the change: a signal never has to know what a Record
        # is in order to be told which version to read.
        self.sig.fetch(165920, version=2)
        for value in self.sig.seen[-1]:
            self.assertNotIsInstance(value, object.__class__)
        self.assertEqual(self.sig.seen[-1][1], 2)


class TestSignalsThatPredatePinning(unittest.TestCase):
    """Fetching must keep working; being ASKED to pin must not be ignored."""

    def test_a_legacy_signal_still_fetches(self):
        # The regression this fixes: this raised TypeError in 2.12.0-2.13.2.
        self.assertEqual(LegacySignal().fetch(165920), {"data": [1, 2, 3]})

    def test_a_legacy_signal_refuses_a_pin(self):
        with self.assertRaises(ValueError) as caught:
            LegacySignal().fetch(165920, version=2)
        msg = str(caught.exception)
        self.assertIn("version", msg)
        # The error has to say how to fix it, not merely that it failed.
        self.assertIn("gather(self, shot, version=None, snapshot=None)", msg)

    def test_the_interim_record_shape_still_fetches(self):
        self.assertEqual(InterimSignal().fetch(165920), {"data": [4, 5]})

    def test_the_interim_record_shape_refuses_a_pin(self):
        # It would accept the CALL and quietly ignore the pin, which is the
        # failure mode pinning exists to prevent. Refuse instead.
        with self.assertRaises(ValueError):
            InterimSignal().fetch(165920, version=2)

    def test_a_signal_taking_kwargs_is_trusted(self):
        sig = KwargsSignal()
        sig.fetch(165920, version=2)
        self.assertEqual(sig.seen[-1], {"version": 2, "snapshot": None})


class TestTheCapabilityCheck(unittest.TestCase):
    def test_each_class_answers_for_itself(self):
        self.assertFalse(LegacySignal._gather_accepts_pin())
        self.assertFalse(InterimSignal._gather_accepts_pin())
        self.assertTrue(ModernSignal._gather_accepts_pin())
        self.assertTrue(KwargsSignal._gather_accepts_pin())

    def test_a_subclass_does_not_inherit_its_parents_answer(self):
        # Cached in the class's own __dict__, so a subclass that overrides
        # gather is judged on its own signature.
        class Upgraded(LegacySignal):
            def gather(self, shot, version=None, snapshot=None):
                return {"data": [9]}

        self.assertFalse(LegacySignal._gather_accepts_pin())
        self.assertTrue(Upgraded._gather_accepts_pin())
        self.assertEqual(Upgraded().fetch(165920, version=2), {"data": [9]})
