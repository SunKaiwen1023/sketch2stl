"""Drawing canvas: Gradio Sketchpad output -> ordered Stroke objects.

OWNER: Serena.

DECISION D5, RESOLVED: option (a), vectorise the raster.

Gradio's Sketchpad hands back an IMAGE, not pen paths. So this thresholds the
ink, thins it to one-pixel-wide centrelines, splits it into connected
components, and walks each component into an ordered path.

What that costs: stroke order and timing are gone. A stroke drawn clockwise and
one drawn anticlockwise look identical, and there is no speed signal. For shape
recognition that is survivable - none of the twelve features in
`recognizer/features.py` uses direction or time.

What it buys: it works with stock Gradio, deploys to Spaces unchanged, and needs
no JavaScript. Revisit option (b) - a real JS canvas posting pointer events - if
recognition accuracy stalls and you have time. The hook is `strokes_from_json`,
still here and still working.
"""
from __future__ import annotations

import numpy as np

from sketch2stl.config import MIN_STROKE_PTS
from sketch2stl.types import Stroke

MIN_COMPONENT_PX = 40          # smaller than this is a stray tap, not a stroke
MAX_STROKES = 12               # a sane cap so a scribble cannot hang the app


def _ink_mask(image) -> np.ndarray | None:
    """Pull a boolean ink mask out of whatever Gradio handed us.

    Sketchpad's shape has changed across Gradio versions - sometimes a plain
    array, sometimes {'composite':..., 'layers':[...], 'background':...}. Handle
    all of them rather than pinning a version.
    """
    if image is None:
        return None

    if isinstance(image, dict):
        layers = image.get("layers") or []
        arr = None
        for layer in layers:                       # prefer the ink layer
            a = np.asarray(layer)
            if a.size and a.ndim == 3 and a.shape[2] == 4 and a[..., 3].max() > 0:
                arr = a
                break
        if arr is None:
            arr = np.asarray(image.get("composite") if image.get("composite") is not None
                             else image.get("background"))
    else:
        arr = np.asarray(image)

    if arr is None or arr.size == 0:
        return None

    if arr.ndim == 3 and arr.shape[2] == 4:
        mask = arr[..., 3] > 16                    # alpha: drawn pixels only
        if mask.any():
            return mask
        arr = arr[..., :3]

    if arr.ndim == 3:
        arr = arr.mean(axis=2)

    # Ink is whichever polarity is rarer - works on both light and dark canvases.
    dark = arr < 128
    return dark if dark.mean() < 0.5 else ~dark


def _order_component(coords: np.ndarray) -> np.ndarray:
    """Walk a skeleton component's pixels into an ordered path.

    Greedy nearest-neighbour from the point furthest from the centroid, which on
    an open stroke is an endpoint and on a closed one is an arbitrary but
    perfectly good starting point.
    """
    pts = coords.astype(np.float64)
    if len(pts) < 3:
        return pts

    start = int(np.argmax(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    remaining = np.ones(len(pts), dtype=bool)
    order = [start]
    remaining[start] = False
    current = pts[start]

    for _ in range(len(pts) - 1):
        idx = np.flatnonzero(remaining)
        d = np.linalg.norm(pts[idx] - current, axis=1)
        j = idx[int(np.argmin(d))]
        # a big jump means the walk finished this branch; stop rather than
        # teleporting across the shape and inventing a chord
        if d.min() > 6.0:
            break
        order.append(j)
        remaining[j] = False
        current = pts[j]

    return pts[order]


def strokes_from_image(image) -> list[Stroke]:
    """Sketchpad output -> ordered strokes in canvas pixel coordinates.

    Returns [] for a blank canvas. The caller must handle that.
    """
    mask = _ink_mask(image)
    if mask is None or not mask.any():
        return []

    try:
        from skimage.measure import label
        from skimage.morphology import skeletonize
    except ImportError as exc:                     # noqa: BLE001
        raise ImportError(
            "scikit-image is needed to read the canvas. `pip install scikit-image`, "
            "or it is already in requirements.txt."
        ) from exc

    skeleton = skeletonize(mask)
    labelled = label(skeleton, connectivity=2)

    strokes: list[Stroke] = []
    sizes = [(i, int((labelled == i).sum())) for i in range(1, labelled.max() + 1)]
    sizes.sort(key=lambda t: -t[1])

    for comp, size in sizes[:MAX_STROKES]:
        if size < MIN_COMPONENT_PX:
            continue
        ys, xs = np.nonzero(labelled == comp)
        path = _order_component(np.column_stack([xs, ys]))   # (x, y) = (col, row)
        if len(path) >= MIN_STROKE_PTS:
            strokes.append(Stroke(points=path))

    return strokes


def strokes_from_json(payload: str) -> list[Stroke]:
    """JSON from a custom JS canvas -> strokes. For option (b), if you build it.

    Expected:  [{"points": [[x, y], ...], "t": [0.0, ...]}, ...]
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
