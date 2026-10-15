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

"""Save one captured figure into a conversation's figures folder.

plotly HTML uses include_plotlyjs="directory": plotly.min.js is written once
beside the HTML files and every figure references it, so a conversation with
many plots carries one copy and still works offline.
"""

from pathlib import Path


def save_figure(kind: str, payload, out_dir, stem: str) -> dict:
    """Save a figure; return ``kind``, ``png_path``, ``html_path``, ``png_error``.

    ``kind`` is ``"matplotlib"`` (payload: a Figure) or ``"plotly"``
    (payload: a figure dict). Paths are str or None.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "png_path": None, "html_path": None,
           "png_error": None}
    if kind == "matplotlib":
        png = out_dir / f"{stem}.png"
        payload.savefig(png, dpi=110, bbox_inches="tight")
        rec["png_path"] = str(png)
        return rec

    import plotly.graph_objects as go
    fig = go.Figure(payload)
    html = out_dir / f"{stem}.html"
    fig.write_html(str(html), include_plotlyjs="directory")
    rec["html_path"] = str(html)
    png = out_dir / f"{stem}.png"
    try:
        fig.write_image(str(png), format="png")
        rec["png_path"] = str(png)
    except Exception as e:  # kaleido needs a Chrome; the HTML still has the plot
        rec["png_error"] = f"{type(e).__name__}: {e}"
    return rec
