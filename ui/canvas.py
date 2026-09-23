"""Drawing canvas: Gradio Sketchpad -> Stroke objects.

OWNER: Serena.

THE ONE THING TO KNOW: Gradio's Sketchpad hands you an IMAGE, not a list of
pen paths. That is a real limitation and it shapes the whole recognition
problem - you get a raster of what was drawn, not the ordered points.

Two ways out, decide early and write the decision in docs/decisions.md:

  (a) VECTORISE THE RASTER. Threshold the image, skeletonise it, and trace the
      skeleton into ordered points. Adds an opencv/scikit-image dependency and
      loses stroke order and timing, but it works with stock Gradio and needs no
      frontend code. `strokes_from_image` below is the stub for this.

  (b) A REAL CANVAS via gr.HTML with a small JavaScript <canvas> that posts
      pointer events back. You get true ordered paths with timestamps, which is
      strictly better input for recognition, at the cost of writing ~80 lines of
      JS. Gradio supports this through `gr.HTML` plus a hidden `gr.Textbox` that
      the JS writes JSON into.

Recommendation: ship (a) to get end to end working, then do (b) if there is time.
The ML arm is much more interesting with (b)'s data, so it is worth the swap if
the schedule allows.
"""
from __future__ import annotations

import numpy as np

from sketch2stl.config import MIN_STROKE_PTS
from sketch2stl.types import Stroke


def strokes_from_image(image: np.ndarray) -> list[Stroke]:
    """Rasterised sketchpad output -> ordered strokes. STUB (option (a) above).

    Sketch of the approach:
        1. Take the alpha or ink channel, threshold to a boolean mask.
        2. skimage.morphology.skeletonize -> one-pixel-wide centrelines.
        3. Label connected components; each component is one stroke.
        4. Order each component's pixels by walking from an endpoint (a pixel
           with exactly one neighbour) along the skeleton.
        5. Return Stroke(points=ordered_pixels) - px_to_mm happens later.

    Returns [] on a blank canvas, which the caller must handle.
    """
    raise NotImplementedError("Serena: see ui/canvas.py docstring, option (a)")


def strokes_from_json(payload: str) -> list[Stroke]:
    """JSON from a custom JS canvas -> strokes. STUB (option (b) above).

    Expected shape, if you build the JS side:
        [{"points": [[x, y], ...], "t": [0.0, ...]}, ...]
    """
    import json
    if not payload:
        return []
    out = []
    for item in json.loads(payload):
        pts = np.asarray(item["points"], dtype=np.float64)
        if len(pts) < MIN_STROKE_PTS:
            continue
        t = np.asarray(item["t"], dtype=np.float64) if item.get("t") else None
        out.append(Stroke(points=pts, t=t))
    return out
