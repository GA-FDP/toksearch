"""B6: naming a version when fetching one shot directly.

A pipeline pins per shot, by putting version/snapshot on the record --
different shots can name different versions, which is the whole reason the
pin lives there. But a signal fetched on its own, outside a pipeline, has no
record, and building one by hand to read a single shot is silly.

So fetch grows the kwargs. It builds the record internally and hands it to
gather, so the pin travels the one path that is already proven and gather's
signature -- which every Signal subclass overrides -- does not change.

Deliberately NOT on the constructor: a version is per-shot, so a signal-wide
version= would pin every shot to its own Nth mint, which is almost never
what anyone means.

TestCase methods: tests/testit.py collects by unittest discovery.
"""

import unittest

from toksearch.record import Record
from toksearch.signal.signal import Signal


class RecordingSignal(Signal):
    """Captures whatever record reaches gather."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def gather(self, shot, record=None):
        self.seen.append(record)
        return {"data": [1, 2, 3]}

    def cleanup_shot(self, shot):
        pass

    def cleanup(self):
        pass

    def cleanup_shot_key(self):
        return ("recording", id(self))


class TestFetchTakesAPin(unittest.TestCase):
    def setUp(self):
        self.sig = RecordingSignal()

    def pin_seen(self):
        rec = self.sig.seen[-1]
        if rec is None:
            return (None, None)
        return rec.get("version", None), rec.get("snapshot", None)

    def test_a_version_reaches_gather(self):
        self.sig.fetch(165920, version=2)
        self.assertEqual(self.pin_seen(), (2, None))

    def test_a_snapshot_reaches_gather(self):
        self.sig.fetch(165920, snapshot="catalog_20260907T232802Z")
        self.assertEqual(self.pin_seen(), (None, "catalog_20260907T232802Z"))

    def test_both_together_reach_gather(self):
        self.sig.fetch(165920, version=3, snapshot="catalog_X")
        self.assertEqual(self.pin_seen(), (3, "catalog_X"))

    def test_the_synthesized_record_carries_the_shot(self):
        # Signals are entitled to read record.shot; a record without one is
        # not a record a pipeline would ever have produced.
        self.sig.fetch(165920, version=2)
        self.assertEqual(self.sig.seen[-1].shot, 165920)

    def test_it_is_a_real_record_not_a_dict(self):
        self.sig.fetch(165920, version=2)
        self.assertIsInstance(self.sig.seen[-1], Record)

    def test_no_pin_passes_no_record(self):
        # Unchanged behaviour: a bare fetch is exactly what it always was.
        self.sig.fetch(165920)
        self.assertIsNone(self.sig.seen[-1])

    def test_a_version_of_zero_is_still_a_pin(self):
        # Falsy but meaningful -- `if version:` would drop it.
        self.sig.fetch(165920, version=0)
        self.assertEqual(self.pin_seen(), (0, None))


class TestAPipelineRecordStillWins(unittest.TestCase):
    def setUp(self):
        self.sig = RecordingSignal()

    def test_a_record_passes_through_untouched(self):
        rec = Record.from_dict({"shot": 165920, "version": 5})
        self.sig.fetch(165920, record=rec)
        self.assertIs(self.sig.seen[-1], rec)

    def test_a_record_and_a_pin_together_is_an_error(self):
        # The pipeline supplies records and never supplies these kwargs; a
        # direct caller supplies the kwargs and has no record. Both at once
        # is ambiguous about which pin is authoritative, and this project
        # does not answer that kind of question by picking one quietly.
        rec = Record.from_dict({"shot": 165920, "version": 5})
        with self.assertRaises(ValueError) as caught:
            self.sig.fetch(165920, record=rec, version=2)
        self.assertIn("not both", str(caught.exception).lower())

    def test_a_record_with_no_pin_plus_a_pin_is_also_an_error(self):
        # Still ambiguous: the record is the pipeline's, and silently
        # decorating it would mutate state the caller shares.
        rec = Record.from_dict({"shot": 165920})
        with self.assertRaises(ValueError):
            self.sig.fetch(165920, record=rec, snapshot="catalog_X")

    def test_the_callers_record_is_never_mutated(self):
        rec = Record.from_dict({"shot": 165920})
        self.sig.fetch(165920, record=rec)
        self.assertNotIn("version", rec.keys())
