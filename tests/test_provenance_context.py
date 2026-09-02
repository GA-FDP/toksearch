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

import os
import subprocess
import tempfile
import unittest

from toksearch.provenance.code import capture_code


def _init_repo(path):
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@e",
               GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@e")
    subprocess.run(["git", "init", "-q", path], check=True, env=env)
    with open(os.path.join(path, "f.txt"), "w") as fh:
        fh.write("one\n")
    subprocess.run(["git", "-C", path, "add", "f.txt"], check=True, env=env)
    subprocess.run(["git", "-C", path, "commit", "-qm", "init"], check=True, env=env)
    return env


class TestCaptureCode(unittest.TestCase):
    def test_returns_commit_in_a_repo(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            code = capture_code(cwd=d)
            self.assertEqual(len(code.commit), 40)

    def test_reports_clean_tree(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            self.assertFalse(capture_code(cwd=d).dirty)

    def test_reports_dirty_tree(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            with open(os.path.join(d, "f.txt"), "w") as fh:
                fh.write("two\n")
            self.assertTrue(capture_code(cwd=d).dirty)

    def test_records_repo_root(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            self.assertIsNotNone(capture_code(cwd=d).repo_root)

    def test_outside_a_repo_returns_none_commit_not_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            code = capture_code(cwd=d)
            self.assertIsNone(code.commit)

    def test_outside_a_repo_dirty_is_none_not_false(self):
        # None means "unknown", False means "known clean". Conflating them
        # would let a provenance record claim a clean tree it never saw.
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(capture_code(cwd=d).dirty)

    def test_records_argv(self):
        code = capture_code()
        self.assertIsInstance(code.argv, tuple)

    def test_is_json_serializable(self):
        from toksearch.provenance.hashing import canonical_json
        canonical_json(capture_code().to_dict())

    def test_to_dict_round_trips_every_field(self):
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            code = capture_code(cwd=d)
            as_dict = code.to_dict()
            for field in ("commit", "dirty", "repo_root", "script", "argv"):
                self.assertIn(field, as_dict)

    def test_untracked_files_alone_count_as_dirty(self):
        # `git status --porcelain` lists untracked files, so a tree with only
        # untracked additions reads as dirty. That is the intended reading:
        # the run may depend on a file that is not in the recorded commit.
        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            with open(os.path.join(d, "scratch.py"), "w") as fh:
                fh.write("x = 1\n")
            self.assertTrue(capture_code(cwd=d).dirty)

    def test_is_frozen(self):
        import dataclasses

        with tempfile.TemporaryDirectory() as d:
            _init_repo(d)
            code = capture_code(cwd=d)
            with self.assertRaises(dataclasses.FrozenInstanceError):
                code.commit = "nope"


from toksearch import Pipeline
from toksearch.signal.mock_signal import MockSignal


def _a_map_func(rec):
    rec["doubled"] = 2


def _a_where_func(rec):
    return True


def _specs_for(build):
    pipeline = Pipeline([1, 2])
    build(pipeline)
    return [op.spec().to_dict() for op in pipeline._operations]


class TestOperationSpecs(unittest.TestCase):
    def test_fetch_reports_op_name_and_field(self):
        op_specs = _specs_for(lambda p: p.fetch("ip", MockSignal()))
        self.assertEqual(op_specs[0]["op"], "fetch")
        self.assertEqual(op_specs[0]["detail"]["name"], "ip")

    def test_fetch_includes_the_signal_spec(self):
        op_specs = _specs_for(lambda p: p.fetch("ip", MockSignal()))
        self.assertEqual(op_specs[0]["detail"]["signal"]["class"], "MockSignal")

    def test_map_reports_the_function(self):
        op_specs = _specs_for(lambda p: p.map(_a_map_func))
        self.assertEqual(op_specs[0]["op"], "map")
        self.assertEqual(op_specs[0]["detail"]["func"]["name"], "_a_map_func")

    def test_keep_is_reported_as_keep_not_map(self):
        op_specs = _specs_for(lambda p: p.keep(["ip"]))
        self.assertEqual(op_specs[0]["op"], "keep")
        self.assertEqual(op_specs[0]["detail"]["fields"], ["ip"])

    def test_where_reports_the_predicate(self):
        op_specs = _specs_for(lambda p: p.where(_a_where_func))
        self.assertEqual(op_specs[0]["op"], "where")
        self.assertEqual(op_specs[0]["detail"]["func"]["name"], "_a_where_func")

    def test_fetch_dataset_reports_dataset_and_signal_names(self):
        op_specs = _specs_for(
            lambda p: p.fetch_dataset("ds", {"ip": MockSignal()})
        )
        self.assertEqual(op_specs[0]["op"], "fetch_dataset")
        self.assertEqual(op_specs[0]["detail"]["ds_name"], "ds")
        self.assertEqual(op_specs[0]["detail"]["signame"], "ip")

    def test_align_reports_its_configuration_not_a_repr(self):
        op_specs = _specs_for(
            lambda p: p.align("ds", [0, 1, 2], dim="times", method="nearest")
        )
        detail = op_specs[0]["detail"]
        self.assertEqual(op_specs[0]["op"], "align")
        self.assertEqual(detail["align_with"], [0, 1, 2])
        self.assertEqual(detail["dim"], "times")
        self.assertEqual(detail["method"], "nearest")

    def test_align_spec_carries_no_memory_address(self):
        from toksearch.provenance.hashing import canonical_json

        op_specs = _specs_for(lambda p: p.align("ds", [0, 1, 2]))
        self.assertNotIn("0x", canonical_json(op_specs))

    def test_two_identical_aligns_produce_equal_specs(self):
        a = _specs_for(lambda p: p.align("ds", [0, 1, 2]))
        b = _specs_for(lambda p: p.align("ds", [0, 1, 2]))
        self.assertEqual(a, b)

    def test_align_with_a_numpy_array_is_serializable(self):
        import numpy as np

        from toksearch.provenance.hashing import canonical_json

        op_specs = _specs_for(lambda p: p.align("ds", np.array([0.0, 1.0])))
        canonical_json(op_specs)

    def test_op_order_is_preserved(self):
        def build(p):
            p.fetch("ip", MockSignal())
            p.map(_a_map_func)
            p.keep(["ip"])

        self.assertEqual([s["op"] for s in _specs_for(build)],
                         ["fetch", "map", "keep"])

    def test_every_op_spec_is_canonically_serializable(self):
        from toksearch.provenance.hashing import canonical_json

        def build(p):
            p.fetch("ip", MockSignal())
            p.fetch_dataset("ds", {"bt": MockSignal()})
            p.map(_a_map_func)
            p.keep(["ip"])
            p.align("ds", [0, 1])
            p.where(_a_where_func)

        canonical_json(_specs_for(build))


from toksearch.backend.serial import SerialRecordSet
from toksearch.backend.multiprocessing import (
    MultiprocessingRecordSet,
    MultiprocessingConfig,
)


class TestRunContext(unittest.TestCase):
    def _ctx(self, shots=(1, 2, 3)):
        pipeline = Pipeline(list(shots))
        pipeline.fetch("ip", MockSignal())
        pipeline.map(_a_map_func)
        return pipeline._run_context(SerialRecordSet, None)

    def test_source_kind_is_shotlist(self):
        self.assertEqual(self._ctx().source.kind, "shotlist")

    def test_source_records_the_count(self):
        self.assertEqual(self._ctx().source.count, 3)

    def test_source_hash_is_stable_for_the_same_shots(self):
        self.assertEqual(self._ctx().source.hash, self._ctx().source.hash)

    def test_source_hash_ignores_shot_order(self):
        a = self._ctx(shots=(1, 2, 3)).source.hash
        b = self._ctx(shots=(3, 1, 2)).source.hash
        self.assertEqual(a, b)

    def test_source_hash_differs_for_different_shots(self):
        self.assertNotEqual(self._ctx(shots=(1, 2)).source.hash,
                            self._ctx(shots=(1, 2, 3)).source.hash)

    def test_signals_are_collected_by_field_name(self):
        self.assertIn("ip", self._ctx().signals)

    def test_ops_are_recorded_in_order(self):
        self.assertEqual([op.op for op in self._ctx().ops], ["fetch", "map"])

    def test_backend_kind_is_the_recordset_class_name(self):
        self.assertEqual(self._ctx().backend.kind, "SerialRecordSet")

    def test_backend_config_is_captured(self):
        pipeline = Pipeline([1])
        config = MultiprocessingConfig(num_workers=4)
        ctx = pipeline._run_context(MultiprocessingRecordSet, config)
        self.assertEqual(ctx.backend.config["num_workers"], 4)

    def test_code_is_captured(self):
        self.assertIsInstance(self._ctx().code.argv, tuple)

    def test_parent_run_is_none_without_a_parent(self):
        self.assertIsNone(self._ctx().parent_run)

    def test_to_dict_is_canonically_serializable(self):
        from toksearch.provenance.hashing import canonical_json
        canonical_json(self._ctx().to_dict())

    def test_to_dict_carries_no_memory_address(self):
        from toksearch.provenance.hashing import canonical_json
        self.assertNotIn("0x", canonical_json(self._ctx().to_dict()))

    def test_two_identical_pipelines_share_an_input_identity(self):
        self.assertEqual(self._ctx().input_identity(), self._ctx().input_identity())

    def test_input_identity_changes_with_signals(self):
        a = self._ctx()
        p = Pipeline([1, 2, 3])
        p.fetch("ip", MockSignal(data=[9, 9]))
        b = p._run_context(SerialRecordSet, None)
        self.assertNotEqual(a.input_identity(), b.input_identity())

    def test_input_identity_ignores_the_backend(self):
        # Two runs reading the same data share an input artifact even when
        # computed on different backends. That shared artifact is what
        # connects the lineage graph.
        p = Pipeline([1, 2, 3])
        p.fetch("ip", MockSignal())
        a = p._run_context(SerialRecordSet, None)
        b = p._run_context(MultiprocessingRecordSet, MultiprocessingConfig(num_workers=4))
        self.assertEqual(a.input_identity(), b.input_identity())

    def test_input_identity_ignores_map_functions(self):
        # Same reasoning: what you do with the data does not change which
        # data you read.
        p1 = Pipeline([1, 2, 3])
        p1.fetch("ip", MockSignal())
        p2 = Pipeline([1, 2, 3])
        p2.fetch("ip", MockSignal())
        p2.map(_a_map_func)
        self.assertEqual(
            p1._run_context(SerialRecordSet, None).input_identity(),
            p2._run_context(SerialRecordSet, None).input_identity(),
        )

    def test_is_frozen(self):
        import dataclasses

        with self.assertRaises(dataclasses.FrozenInstanceError):
            self._ctx().device = "nope"


class TestSqlSource(unittest.TestCase):
    def test_shotlist_pipeline_has_no_sql_source(self):
        self.assertIsNone(Pipeline([1, 2])._sql_source)

    def test_chained_pipeline_inherits_sql_source(self):
        parent = Pipeline([1, 2])
        parent._sql_source = {"query": "select shot from shots", "params": ()}
        self.assertEqual(Pipeline(parent)._sql_source, parent._sql_source)

    def test_sql_source_produces_a_sql_source_spec(self):
        pipeline = Pipeline([1, 2])
        pipeline._sql_source = {"query": "select shot from shots", "params": ("a",)}
        source = pipeline._source_spec()
        self.assertEqual(source.kind, "sql")
        self.assertEqual(source.query, "select shot from shots")
        self.assertEqual(source.params, ("a",))
        self.assertIsNotNone(source.hash)


class TestRecordSetSource(unittest.TestCase):
    def test_recordset_parent_is_reported_as_recordset(self):
        first = Pipeline([1, 2, 3])
        first.fetch("ip", MockSignal())
        results = first.compute_serial()
        source = Pipeline(results)._source_spec()
        self.assertEqual(source.kind, "recordset")
        self.assertEqual(source.count, 3)

class TestSqlSourceEndToEnd(unittest.TestCase):
    """Exercise the real from_sql path, not a hand-set _sql_source.

    from_sql is the only pre-existing public API this work modifies, so the
    round trip through it needs direct coverage: a hand-set attribute would
    still pass if the from_sql edit were reverted.
    """

    def _conn(self):
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("create table shots_type (shot int, shot_type text)")
        conn.executemany(
            "insert into shots_type values (?,?)",
            [(101, "plasma"), (102, "plasma"), (103, "calibration")],
        )
        conn.commit()
        return conn

    _QUERY = "select shot from shots_type where shot_type = ?"

    def test_from_sql_records_the_query(self):
        pipeline = Pipeline.from_sql(self._conn(), self._QUERY, "plasma")
        self.assertEqual(pipeline._source_spec().query, self._QUERY)

    def test_from_sql_records_the_params(self):
        pipeline = Pipeline.from_sql(self._conn(), self._QUERY, "plasma")
        self.assertEqual(pipeline._source_spec().params, ("plasma",))

    def test_from_sql_source_kind_is_sql(self):
        pipeline = Pipeline.from_sql(self._conn(), self._QUERY, "plasma")
        self.assertEqual(pipeline._source_spec().kind, "sql")

    def test_from_sql_still_returns_a_working_pipeline(self):
        pipeline = Pipeline.from_sql(self._conn(), self._QUERY, "plasma")
        self.assertEqual(len(pipeline.compute_serial()), 2)

    def test_different_queries_have_different_input_identities(self):
        a = Pipeline.from_sql(self._conn(), self._QUERY, "plasma")
        b = Pipeline.from_sql(self._conn(), self._QUERY, "calibration")
        self.assertNotEqual(
            a._run_context(SerialRecordSet, None).input_identity(),
            b._run_context(SerialRecordSet, None).input_identity(),
        )
