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
import warnings

from toksearch.provenance.base import Provenance, safe_call


class _ExplodingProvenance(Provenance):
    def on_compute_start(self, ctx):
        raise RuntimeError("boom")

    def on_compute_end(self, ctx, recordset):
        pass

    def output(self, *paths, **custom_properties):
        pass

    def metrics(self, name, values):
        pass

    def finalize(self):
        pass


class TestSafeCall(unittest.TestCase):
    def test_swallows_errors_by_default(self):
        prov = _ExplodingProvenance()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            safe_call(prov, "on_compute_start", None)
        self.assertEqual(len(caught), 1)

    def test_warning_names_the_backend_and_hook(self):
        prov = _ExplodingProvenance()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            safe_call(prov, "on_compute_start", None)
        message = str(caught[0].message)
        self.assertIn("_ExplodingProvenance", message)
        self.assertIn("on_compute_start", message)

    def test_warning_says_results_are_unaffected(self):
        # The message is the only thing standing between a user and the
        # assumption that their compute failed. It must say otherwise.
        prov = _ExplodingProvenance()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            safe_call(prov, "on_compute_start", None)
        self.assertIn("unaffected", str(caught[0].message))

    def test_strict_mode_reraises(self):
        prov = _ExplodingProvenance()
        prov.strict = True
        with self.assertRaises(RuntimeError):
            safe_call(prov, "on_compute_start", None)

    def test_none_provenance_is_a_no_op(self):
        safe_call(None, "on_compute_start", None)

    def test_none_provenance_returns_none(self):
        self.assertIsNone(safe_call(None, "on_compute_start", None))

    def test_returns_the_hook_result(self):
        class _Ok(_ExplodingProvenance):
            def on_compute_start(self, ctx):
                return "ok"

        self.assertEqual(safe_call(_Ok(), "on_compute_start", None), "ok")

    def test_passes_through_arguments(self):
        seen = {}

        class _Recorder(_ExplodingProvenance):
            def on_compute_start(self, ctx):
                seen["ctx"] = ctx

        safe_call(_Recorder(), "on_compute_start", "the-context")
        self.assertEqual(seen["ctx"], "the-context")

    def test_strict_defaults_to_false(self):
        self.assertFalse(_ExplodingProvenance().strict)

    def test_run_id_defaults_to_none(self):
        self.assertIsNone(_ExplodingProvenance().run_id)


class TestProvenanceIsAbstract(unittest.TestCase):
    def test_cannot_instantiate_the_abc(self):
        with self.assertRaises(TypeError):
            Provenance()

    def test_all_five_hooks_are_abstract(self):
        self.assertEqual(
            sorted(Provenance.__abstractmethods__),
            ["finalize", "metrics", "on_compute_end", "on_compute_start", "output"],
        )

    def test_a_partial_implementation_cannot_be_instantiated(self):
        class _Partial(Provenance):
            def on_compute_start(self, ctx):
                pass

        with self.assertRaises(TypeError):
            _Partial()
