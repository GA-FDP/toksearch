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


import json
import os
import tempfile

from toksearch import Pipeline
from toksearch.provenance import JsonProvenance
from toksearch.signal.mock_signal import MockSignal
from toksearch.backend.serial import SerialRecordSet


def _ctx():
    pipeline = Pipeline([1, 2, 3])
    pipeline.fetch("ip", MockSignal())
    return pipeline._run_context(SerialRecordSet, None)


class TestJsonProvenance(unittest.TestCase):
    def _finalized(self, directory, act=None):
        """Run a JsonProvenance through a full cycle, return the payload."""
        path = os.path.join(directory, "run.json")
        prov = JsonProvenance("study", path=path)
        prov.on_compute_start(_ctx())
        if act is not None:
            act(prov)
        prov.finalize()
        with open(path) as fh:
            return json.load(fh)

    def test_writes_a_file_on_finalize(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.json")
            prov = JsonProvenance("study", path=path)
            prov.on_compute_start(_ctx())
            prov.finalize()
            self.assertTrue(os.path.exists(path))

    def test_records_the_pipeline_name(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._finalized(d)["pipeline_name"], "study")

    def test_stage_defaults_to_the_pipeline_name(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._finalized(d)["stage"], "study")

    def test_records_the_run_context(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIn("ip", self._finalized(d)["context"]["signals"])

    def test_records_the_input_identity(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(len(self._finalized(d)["input_identity"]), 64)

    def test_records_declared_outputs(self):
        with tempfile.TemporaryDirectory() as d:
            payload = self._finalized(d, lambda p: p.output("a.nc", "b.png"))
            self.assertEqual([o["path"] for o in payload["outputs"]],
                             ["a.nc", "b.png"])

    def test_output_custom_properties_are_kept(self):
        with tempfile.TemporaryDirectory() as d:
            payload = self._finalized(d, lambda p: p.output("a.nc", kind="figure"))
            self.assertEqual(payload["outputs"][0]["kind"], "figure")

    def test_records_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            payload = self._finalized(d, lambda p: p.metrics("eval", {"rmse": 0.1}))
            self.assertEqual(payload["metrics"]["eval"]["rmse"], 0.1)

    def test_has_a_run_id(self):
        self.assertTrue(JsonProvenance("study").run_id)

    def test_run_ids_are_distinct(self):
        self.assertNotEqual(JsonProvenance("a").run_id, JsonProvenance("b").run_id)

    def test_creates_missing_parent_directories(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "deeper", "run.json")
            prov = JsonProvenance("study", path=path)
            prov.on_compute_start(_ctx())
            prov.finalize()
            self.assertTrue(os.path.exists(path))

    def test_strict_is_settable(self):
        self.assertTrue(JsonProvenance("study", strict=True).strict)

    def test_payload_is_json_round_trippable(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsInstance(self._finalized(d), dict)

    def test_on_compute_end_logs_write_directories_from_the_context(self):
        # Not from the recordset: see RunContext.write_directories. Passing
        # None as the recordset proves the paths never came from iterating it.
        from toksearch.provenance.context import OpSpec

        ctx = _ctx()
        ctx_with_write = type(ctx)(
            source=ctx.source,
            ops=ctx.ops + (OpSpec("write", {"directory": "/tmp/peaks",
                                            "track": "directory"}),),
            signals=ctx.signals,
            backend=ctx.backend,
            code=ctx.code,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "run.json")
            prov = JsonProvenance("study", path=path)
            prov.on_compute_start(ctx_with_write)
            prov.on_compute_end(ctx_with_write, None)
            prov.finalize()
            with open(path) as fh:
                outputs = [o["path"] for o in json.load(fh)["outputs"]]
        self.assertEqual(outputs, ["/tmp/peaks"])
