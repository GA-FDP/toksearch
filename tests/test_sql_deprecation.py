# Copyright 2024 General Atomics
# Licensed under the Apache License, Version 2.0.

"""Tests for the deprecation shim at toksearch.sql.mssql.connect_d3drdb.

The function is preserved for back-compat after the migration to
toksearch_d3d.sql.connect_d3drdb. Calling it must:
  - emit a DeprecationWarning pointing at the new path
  - lazily delegate to toksearch_d3d.sql.connect_d3drdb
  - forward all kwargs unchanged
  - raise a helpful ImportError if toksearch_d3d isn't installed
"""

import sys
import unittest
import warnings
from unittest import mock


class TestConnectD3DRDBShim(unittest.TestCase):
    def test_emits_deprecation_warning(self):
        from toksearch.sql.mssql import connect_d3drdb
        with mock.patch.dict(
            sys.modules,
            {"toksearch_d3d.sql": mock.MagicMock(
                connect_d3drdb=mock.MagicMock(return_value=mock.sentinel.conn)
            )},
        ):
            with self.assertWarns(DeprecationWarning) as ctx:
                result = connect_d3drdb()
        self.assertIs(result, mock.sentinel.conn)
        self.assertIn(
            "toksearch_d3d.sql.connect_d3drdb", str(ctx.warning),
        )

    def test_kwargs_forwarded_to_impl(self):
        from toksearch.sql.mssql import connect_d3drdb
        impl = mock.MagicMock(return_value=mock.sentinel.conn)
        with mock.patch.dict(
            sys.modules,
            {"toksearch_d3d.sql": mock.MagicMock(connect_d3drdb=impl)},
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                connect_d3drdb(db="code_rundb", host="h", port=9999)
        impl.assert_called_once_with(db="code_rundb", host="h", port=9999)

    def test_helpful_error_if_toksearch_d3d_missing(self):
        from toksearch.sql.mssql import connect_d3drdb
        # Force the import inside the shim to fail.
        # Insert a None entry so `import toksearch_d3d.sql` hits ImportError.
        with mock.patch.dict(sys.modules, {"toksearch_d3d.sql": None}):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                with self.assertRaisesRegex(
                    ImportError, "toksearch_d3d.*conda install"
                ):
                    connect_d3drdb()
