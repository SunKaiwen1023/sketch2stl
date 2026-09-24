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

from sketch2stl.config import CANVAS_H, CANVAS_W, MIN_STROKE_PTS
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


MAX_STEP_PX = 4.0        # skeleton neighbours are at most sqrt(2) apart
STITCH_PX = 14.0         # how far apart two runs may be and still be one stroke


def _order_component(coords: np.ndarray) -> np.ndarray:
    """Walk a skeleton component's pixels into an ordered path.

    Greedy nearest-neighbour, but in SEGMENTS, then stitched back together.

    The naive version - one greedy walk that stops at the first big jump - throws
    away most of the stroke. Skeletonising a hand-drawn line leaves short spurs
    and the odd two-pixel-wide patch; the walk wanders into a spur, exhausts it,
    finds nothing nearby and quits. Measured on a shaky circle: 584 skeleton
    pixels in, 41 points out, spanning 3.8 mm instead of 30. The circle fit then
    returned r = 1.8 mm for a 15 mm circle, and the app cut a hole the size of a
    pinhead.

    So: when the walk stalls, start a NEW segment from the furthest remaining
    point instead of stopping. Then stitch segments whose ends nearly meet. A
    spur becomes a short segment that gets dropped; a ring split in two by a
    thick patch gets rejoined.
    """
    pts = coords.astype(np.float64)
    if len(pts) < 3:
        return pts

    centroid = pts.mean(axis=0)
    remaining = np.ones(len(pts), dtype=bool)
    segments: list[list[int]] = []

    while remaining.any():
        idx = np.flatnonzero(remaining)
        start = idx[int(np.argmax(np.linalg.norm(pts[idx] - centroid, axis=1)))]
        seg = [int(start)]
        remaining[start] = False
        current = pts[start]

        while True:
            idx = np.flatnonzero(remaining)
            if len(idx) == 0:
                break
            d = np.linalg.norm(pts[idx] - current, axis=1)
            if d.min() > MAX_STEP_PX:
                break
            j = int(idx[int(np.argmin(d))])
            seg.append(j)
            remaining[j] = False
            current = pts[j]

        segments.append(seg)

    if not segments:
        return pts

    segments.sort(key=len, reverse=True)
    path = list(segments.pop(0))

    # stitch: repeatedly attach whichever leftover run comes closest to an end
    changed = True
    while changed and segments:
        changed = False
        head, tail = pts[path[0]], pts[path[-1]]
        best = None
        for k, seg in enumerate(segments):
            if len(seg) < 3:
                continue
            a, b = pts[seg[0]], pts[seg[-1]]
            for dist, where, rev in (
                (np.linalg.norm(tail - a), "end", False),
                (np.linalg.norm(tail - b), "end", True),
                (np.linalg.norm(head - b), "start", False),
                (np.linalg.norm(head - a), "start", True),
            ):
                if dist <= STITCH_PX and (best is None or dist < best[0]):
                    best = (dist, k, where, rev)
        if best is not None:
            _, k, where, rev = best
            seg = segments.pop(k)
            if rev:
                seg = seg[::-1]
            path = (path + seg) if where == "end" else (seg + path)
            changed = True

    return pts[path]


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


def render_recognition(raw_mm, primitive, size=(CANVAS_W, CANVAS_H)):
    """Draw what the user drew (grey) under what we recognised (blue).

    The app already replaces a wobbly stroke with an exact primitive - that is
    what `fit_params` does - but the user never SEES it happen, because the
    canvas keeps showing their original scrawl. This is the missing feedback:
    proof that the snap occurred, and a way to catch a bad recognition before
    committing it to the model.
    """
    from PIL import Image, ImageDraw

    from sketch2stl.strokes import mm_to_px

    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)

    for x in range(0, size[0], 40):
        d.line([(x, 0), (x, size[1])], fill="#f0f0f0")
    for y in range(0, size[1], 40):
        d.line([(0, y), (size[0], y)], fill="#f0f0f0")

    if raw_mm is not None and len(raw_mm) > 1:
        d.line([tuple(p) for p in mm_to_px(raw_mm)], fill="#c9c9c9", width=5, joint="curve")

    if primitive is not None and len(primitive.points) > 1:
        pts = [tuple(p) for p in mm_to_px(primitive.points)]
        d.line(pts, fill="#2a6fdb", width=3, joint="curve")
        for p in pts[::max(1, len(pts) // 24)]:
            d.ellipse([p[0] - 2, p[1] - 2, p[0] + 2, p[1] + 2], fill="#2a6fdb")

    return np.asarray(img)
