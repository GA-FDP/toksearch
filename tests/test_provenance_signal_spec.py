# Copyright 2026 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

from toksearch.signal.signal import Signal
from toksearch.signal.mock_signal import MockSignal
from toksearch.signal.mds import MdsSignal, MdsTreePath
from toksearch.signal.zarr import ZarrSignal


class _UnspeccedSignal(Signal):
    """A third-party-style signal that never implements _spec_fields."""

    def __init__(self, thing, count=3):
        super().__init__()
        self.thing = thing
        self.count = count
        self._private = "should not appear"

    def gather(self, shot):
        return {"data": [0]}

    def cleanup_shot(self, shot):
        pass

    def cleanup(self):
        pass


class _SpeccedSignal(Signal):
    def __init__(self, thing):
        super().__init__()
        self.thing = thing

    def _spec_fields(self):
        return {"thing": self.thing}

    def gather(self, shot):
        return {"data": [0]}

    def cleanup_shot(self, shot):
        pass

    def cleanup(self):
        pass


def _a_callback(result):
    return result


class TestSignalSpec(unittest.TestCase):
    def test_spec_reports_class_and_module(self):
        spec = _SpeccedSignal("x").spec()
        self.assertEqual(spec["class"], "_SpeccedSignal")
        self.assertEqual(spec["module"], __name__)

    def test_spec_includes_declared_fields(self):
        spec = _SpeccedSignal("x").spec()
        self.assertEqual(spec["fields"], {"thing": "x"})

    def test_spec_is_not_marked_incomplete_when_declared(self):
        self.assertNotIn("spec_incomplete", _SpeccedSignal("x").spec())

    def test_spec_records_dims_and_units(self):
        spec = _SpeccedSignal("x").spec()
        self.assertEqual(spec["dims"], ["times"])
        self.assertTrue(spec["with_units"])

    def test_spec_records_callback(self):
        sig = _SpeccedSignal("x").set_callback(_a_callback)
        self.assertEqual(sig.spec()["callback"]["name"], "_a_callback")

    def test_spec_callback_is_none_by_default(self):
        self.assertIsNone(_SpeccedSignal("x").spec()["callback"])

    def test_two_identical_signals_have_equal_specs(self):
        self.assertEqual(_SpeccedSignal("x").spec(), _SpeccedSignal("x").spec())

    def test_different_signals_have_different_specs(self):
        self.assertNotEqual(_SpeccedSignal("x").spec(), _SpeccedSignal("y").spec())


class TestReflectiveFallback(unittest.TestCase):
    def test_unspecced_signal_does_not_raise(self):
        _UnspeccedSignal("x").spec()

    def test_unspecced_signal_is_marked_incomplete(self):
        self.assertTrue(_UnspeccedSignal("x").spec()["spec_incomplete"])

    def test_reflective_fields_capture_public_attributes(self):
        fields = _UnspeccedSignal("x", count=7).spec()["fields"]
        self.assertEqual(fields["thing"], "x")
        self.assertEqual(fields["count"], 7)

    def test_reflective_fields_exclude_private_attributes(self):
        self.assertNotIn("_private", _UnspeccedSignal("x").spec()["fields"])

    def test_reflective_fields_exclude_base_class_state(self):
        fields = _UnspeccedSignal("x").spec()["fields"]
        for excluded in ("dims", "data_order", "with_units"):
            self.assertNotIn(excluded, fields)


class TestMockSignalSpec(unittest.TestCase):
    def test_mock_signal_declares_its_fields(self):
        spec = MockSignal().spec()
        self.assertNotIn("spec_incomplete", spec)

    def test_mock_signal_spec_reflects_data(self):
        a = MockSignal(data=[1, 2, 3]).spec()
        b = MockSignal(data=[9, 9, 9]).spec()
        self.assertNotEqual(a, b)


class TestSpecDeterminism(unittest.TestCase):
    """The spec feeds an input-identity hash, so it must serialize stably."""

    def test_specced_signal_spec_is_canonically_serializable(self):
        from toksearch.provenance.hashing import canonical_json

        canonical_json(_SpeccedSignal("x").spec())

    def test_reflective_spec_is_canonically_serializable(self):
        from toksearch.provenance.hashing import canonical_json

        canonical_json(_UnspeccedSignal("x").spec())

    def test_mock_signal_spec_hashes_identically_for_equal_signals(self):
        from toksearch.provenance.hashing import sha256_of

        self.assertEqual(sha256_of(MockSignal().spec()), sha256_of(MockSignal().spec()))

    def test_reflective_spec_carries_no_memory_address(self):
        from toksearch.provenance.hashing import canonical_json

        class _HoldsAnObject(Signal):
            def __init__(self):
                super().__init__()
                self.thing = object()

            def gather(self, shot):
                return {"data": [0]}

            def cleanup_shot(self, shot):
                pass

            def cleanup(self):
                pass

        self.assertNotIn("0x", canonical_json(_HoldsAnObject().spec()))


class TestMdsSignalSpec(unittest.TestCase):
    def test_declares_its_fields(self):
        sig = MdsSignal(r"\ipmhd", "efit01", location="remote://atlas.gat.com")
        self.assertNotIn("spec_incomplete", sig.spec())

    def test_captures_expression_and_tree(self):
        sig = MdsSignal(r"\ipmhd", "efit01", location="remote://atlas.gat.com")
        fields = sig.spec()["fields"]
        self.assertEqual(fields["expression"], r"\ipmhd")
        self.assertEqual(fields["treename"], "efit01")

    def test_captures_location(self):
        sig = MdsSignal(r"\ipmhd", "efit01", location="remote://atlas.gat.com")
        self.assertEqual(sig.spec()["fields"]["location"], "remote://atlas.gat.com")

    def test_tree_path_location_is_serializable(self):
        sig = MdsSignal(r"\ipmhd", "efit01", location=MdsTreePath(efit01="/a/b"))
        location = sig.spec()["fields"]["location"]
        self.assertIsInstance(location, (str, dict))

    def test_different_expressions_differ(self):
        a = MdsSignal(r"\ipmhd", "efit01", location="remote://atlas.gat.com")
        b = MdsSignal(r"\betan", "efit01", location="remote://atlas.gat.com")
        self.assertNotEqual(a.spec(), b.spec())

    def test_spec_carries_no_memory_address(self):
        from toksearch.provenance.hashing import canonical_json

        sig = MdsSignal(r"\ipmhd", "efit01", location="remote://atlas.gat.com")
        self.assertNotIn("0x", canonical_json(sig.spec()))

    def test_identical_signals_hash_alike(self):
        from toksearch.provenance.hashing import sha256_of

        a = MdsSignal(r"\ipmhd", "efit01", location="remote://atlas.gat.com")
        b = MdsSignal(r"\ipmhd", "efit01", location="remote://atlas.gat.com")
        self.assertEqual(sha256_of(a.spec()), sha256_of(b.spec()))


class TestZarrSignalSpec(unittest.TestCase):
    def test_declares_its_fields(self):
        sig = ZarrSignal(path="s3://bucket", treepath="magnetics/ip")
        self.assertNotIn("spec_incomplete", sig.spec())

    def test_captures_path_and_treepath(self):
        sig = ZarrSignal(path="s3://bucket", treepath="magnetics/ip")
        fields = sig.spec()["fields"]
        self.assertEqual(fields["path"], "s3://bucket")
        self.assertEqual(fields["treepath"], "magnetics/ip")

    def test_captures_file_name_format(self):
        sig = ZarrSignal(
            path="s3://bucket", treepath="magnetics/ip", file_name_format="x-{shot}.zarr"
        )
        self.assertEqual(sig.spec()["fields"]["file_name_format"], "x-{shot}.zarr")

    def test_fetch_units_changes_the_spec(self):
        # ZarrSignal.__init__ does not sync fetch_units into the base class's
        # with_units, so spec()["with_units"] is always True here. Without
        # fetch_units in fields, these two would collide.
        a = ZarrSignal(path="s3://bucket", treepath="magnetics/ip", fetch_units=True)
        b = ZarrSignal(path="s3://bucket", treepath="magnetics/ip", fetch_units=False)
        self.assertNotEqual(a.spec(), b.spec())

    def test_filesystem_object_is_not_in_the_spec(self):
        sig = ZarrSignal(path="s3://bucket", treepath="magnetics/ip")
        self.assertNotIn("fs", sig.spec()["fields"])

    def test_spec_carries_no_memory_address(self):
        from toksearch.provenance.hashing import canonical_json

        sig = ZarrSignal(path="s3://bucket", treepath="magnetics/ip")
        self.assertNotIn("0x", canonical_json(sig.spec()))
