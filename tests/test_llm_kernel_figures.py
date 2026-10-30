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

"""Tests for saving captured figures into a conversation's figures folder."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import plotly.graph_objects as go

from toksearch.llm.kernel.figures import save_figure


class TestSaveFigure(unittest.TestCase):
    def test_matplotlib_saves_png_only(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot([1, 2, 3])
        with TemporaryDirectory() as tmp:
            rec = save_figure("matplotlib", fig, Path(tmp) / "figs", "c1-1")
            self.assertEqual(rec["kind"], "matplotlib")
            self.assertTrue(Path(rec["png_path"]).read_bytes()
                            .startswith(b"\x89PNG"))
            self.assertIsNone(rec["html_path"])
        plt.close("all")

    def test_plotly_shares_one_plotly_js(self):
        with TemporaryDirectory() as tmp:
            figs = Path(tmp) / "figures"
            a = save_figure("plotly", go.Figure(go.Scatter(y=[1, 2])).to_dict(),
                            figs, "c1-1")
            b = save_figure("plotly", go.Figure(go.Scatter(y=[3, 4])).to_dict(),
                            figs, "c2-1")
            self.assertEqual(sorted(p.name for p in figs.glob("*.js")),
                             ["plotly.min.js"])
            html = Path(a["html_path"]).read_text()
            self.assertIn('src="plotly.min.js"', html)
            self.assertLess(len(html), 100_000)
            self.assertTrue(Path(b["png_path"]).exists())
            self.assertIsNone(b["png_error"])

    def test_plotly_png_failure_keeps_html(self):
        fig_dict = go.Figure(go.Scatter(y=[1, 2])).to_dict()
        with TemporaryDirectory() as tmp:
            with mock.patch.object(go.Figure, "write_image",
                                   side_effect=RuntimeError("no Chrome")):
                rec = save_figure("plotly", fig_dict, Path(tmp), "c1-1")
            self.assertIsNone(rec["png_path"])
            self.assertIn("RuntimeError: no Chrome", rec["png_error"])
            self.assertTrue(Path(rec["html_path"]).exists())


if __name__ == "__main__":
    unittest.main()
