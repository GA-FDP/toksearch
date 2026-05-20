# Copyright 2024 General Atomics
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
"""Tests for toksearch.llm.gui.figure_capture."""

import unittest
from unittest import mock

import matplotlib
matplotlib.use("Agg")  # headless backend for CI


class TestMatplotlibCapture(unittest.TestCase):
    def test_wrap_emits_each_figure_then_closes(self):
        from toksearch.llm.gui.figure_capture import wrap_run_python_handler
        from toksearch.llm.tools import ToolOutput

        emitted = []
        on_figure = lambda kind, payload: emitted.append((kind, payload))

        # The inner handler is what the spec replaces -- here we fake
        # it to just create two matplotlib figures and return an OK
        # ToolOutput. We do NOT close the figures; the wrapper must.
        def inner(args, session):
            import matplotlib.pyplot as plt
            plt.figure()
            plt.figure()
            return ToolOutput(text="ok", is_error=False)

        wrapped = wrap_run_python_handler(inner, on_figure)
        result = wrapped({"code": ""}, mock.Mock())

        self.assertEqual(result.text, "ok")
        self.assertEqual(len(emitted), 2)
        for kind, fig in emitted:
            self.assertEqual(kind, "matplotlib")
            self.assertEqual(fig.__class__.__name__, "Figure")

        # After the call, pyplot's registry must be empty.
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
