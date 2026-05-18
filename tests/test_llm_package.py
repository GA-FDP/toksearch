# Copyright 2024 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Smoke test: the toksearch.llm package and its backends subpackage import."""

import unittest


class TestPackageImports(unittest.TestCase):
    def test_import_toksearch_llm(self):
        import toksearch.llm  # noqa: F401

    def test_import_toksearch_llm_backends(self):
        import toksearch.llm.backends  # noqa: F401


if __name__ == "__main__":
    unittest.main()
