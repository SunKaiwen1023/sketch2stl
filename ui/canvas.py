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

    RETURNS None FOR ANYTHING IT CANNOT READ, and never raises. This function is
    the first thing every canvas callback calls, and an exception here does not
    just lose a drawing: Gradio leaves EVERY output of the failed event in its
    pending state, so the canvas is left spinning on a loading animation that
    only a different, successful click can clear. That was the hang. A Sketchpad
    that has been written to programmatically and not drawn on since sends
    {'background': None, 'layers': [], 'composite': None}, which is exactly the
    shape that used to reach `arr < 128` as a 0-d object array holding None.
    """
    if image is None:
        return None

    if isinstance(image, dict):
        layers = image.get("layers") or []
        arr = None
        for layer in layers:                       # prefer the ink layer
            a = np.asarray(layer) if layer is not None else None
            if (a is not None and a.size and a.ndim == 3 and a.shape[2] == 4
                    and a[..., 3].max() > 0):
                arr = a
                break
        if arr is None:
            fallback = image.get("composite")
            if fallback is None:
                fallback = image.get("background")
            arr = np.asarray(fallback) if fallback is not None else None
    else:
        arr = np.asarray(image)

    # `np.asarray(None)` is a 0-d OBJECT array, not None and not empty, so the
    # dtype check is the one that actually catches an all-empty payload.
    if arr is None or arr.size == 0 or arr.ndim < 2 or arr.dtype == object:
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


def ink_mask(image) -> np.ndarray | None:
    """Public name for the canvas -> boolean ink mask step.

    Split out from `strokes_from_image` so the app can subtract the ink it has
    already committed. See `Session.consumed`.
    """
    return _ink_mask(image)


def strokes_from_mask(mask: np.ndarray | None) -> list[Stroke]:
    """A boolean ink mask -> ordered strokes in canvas pixel coordinates."""
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


def strokes_from_image(image) -> list[Stroke]:
    """Sketchpad output -> ordered strokes in canvas pixel coordinates.

    Returns [] for a blank canvas. The caller must handle that.
    """
    return strokes_from_mask(_ink_mask(image))


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


def render_sketch(built, current=(), raw_mm=(), selected: str | None = None,
                  size=(CANVAS_W, CANVAS_H)):
    """The CLEAN sketch: every layer already built, plus what is being read now.

    The canvas has to keep the user's own ink (writing to it hangs the
    Sketchpad), so this is where the snapped version lives:

      grey fill      layers already built (ADD), red outline for CUT layers,
                     hidden layers dashed-light
      orange         the layer selected in the layer panel
      thin grey      the raw ink being read right now
      blue           what it was recognised / cleaned up as - every shape, holes included

    `built` is a list of Features (extrude ones are drawn; revolves live on the
    half canvas). `current` is a list of Profiles.
    """
    from PIL import Image, ImageDraw

    from sketch2stl.strokes import mm_to_px
    from sketch2stl.types import FeatureKind, Op

    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    _draw_grid(d, size)

    def ring(points):
        return [tuple(p) for p in mm_to_px(np.asarray(points, dtype=np.float64))]

    for f in built:
        if getattr(f, "kind", FeatureKind.EXTRUDE) is not FeatureKind.EXTRUDE:
            continue
        outer = ring(f.profile.outer)
        if len(outer) < 3:
            continue
        sel = selected is not None and f.feature_id == selected
        if not f.visible:
            d.line(outer + outer[:1], fill="#d9d9d9", width=1)
            continue
        if f.op is Op.CUT:
            d.polygon(outer, fill="#fff4e6" if sel else "#ffffff")
            d.line(outer + outer[:1], fill="#f59f00" if sel else "#d9480f", width=4 if sel else 2)
        else:
            d.polygon(outer, fill="#ffe8cc" if sel else "#e9ecef")
            for h in f.profile.holes:
                d.polygon(ring(h), fill="#ffffff")
            d.line(outer + outer[:1], fill="#e8590c" if sel else "#868e96", width=3 if sel else 2)
            for h in f.profile.holes:
                hr = ring(h); d.line(hr + hr[:1], fill="#868e96", width=2)

    for r in raw_mm or ():
        if r is not None and len(r) > 1:
            d.line(ring(r), fill="#c9c9c9", width=4, joint="curve")

    for prof in current or ():
        outer = ring(prof.outer)
        d.line(outer + outer[:1], fill="#2a6fdb", width=3, joint="curve")
        for h in prof.holes:
            hr = ring(h); d.line(hr + hr[:1], fill="#2a6fdb", width=3, joint="curve")

    return np.asarray(img)


def render_half(raw_mm, segments, ring_mm, revolving: bool,
                size=(CANVAS_W, CANVAS_H)):
    """The half-mode counterpart of `render_recognition`.

    Half mode used to return nothing at all to the preview panel, so "What did
    I draw?" showed an empty box - the one place in the app where the user got
    no feedback, and the place they most need it, because a half profile is the
    hardest thing to picture the result of.

    Four layers, back to front:

      grey     the raw stroke, as drawn
      blue     the FITTED half - straight where you meant straight, a true arc
               where you meant an arc. This is what actually gets swept.
      dots     the corners the fit found, so an unwanted one is visible
      pale     the resulting silhouette: the full circle a revolve sweeps
               through, or the mirrored outline an extrude produces
    """
    from PIL import Image, ImageDraw

    from sketch2stl.strokes import mm_to_px

    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for x in range(0, size[0], 40):
        d.line([(x, 0), (x, size[1])], fill="#f0f0f0")
    for y in range(0, size[1], 40):
        d.line([(0, y), (size[0], y)], fill="#f0f0f0")

    # The silhouette first, so the fitted line draws over it.
    if ring_mm is not None and len(ring_mm) > 2:
        px = [tuple(p) for p in mm_to_px(np.asarray(ring_mm, dtype=np.float64))]
        if revolving:
            # A revolve sweeps the half all the way round: the widest circle it
            # passes through is the outline mirrored to the far side.
            far = [(2 * AXIS_X_PX - x, y) for x, y in px]
            d.polygon(px + far[::-1], fill="#eef1f8", outline="#ccd4e8")
        else:
            d.polygon(px, fill="#eef1f8", outline="#ccd4e8")

    x = int(AXIS_X_PX)
    for y in range(0, size[1], 16):
        d.line([(x, y), (x, min(y + 9, size[1]))], fill=(120, 140, 200), width=2)

    if raw_mm is not None and len(raw_mm) > 1:
        d.line([tuple(p) for p in mm_to_px(np.asarray(raw_mm, dtype=np.float64))],
               fill="#c9c9c9", width=5, joint="curve")

    for seg in segments or ():
        if len(seg.points) < 2:
            continue
        pts = [tuple(p) for p in mm_to_px(np.asarray(seg.points, dtype=np.float64))]
        # Straight pieces and curved pieces in the same blue, but a LINE is drawn
        # as the two endpoints only - that IS the claim being made about it.
        d.line(pts, fill="#2a6fdb", width=3, joint="curve")
        for end in (pts[0], pts[-1]):
            d.ellipse([end[0] - 4, end[1] - 4, end[0] + 4, end[1] + 4],
                      fill="#ffffff", outline="#2a6fdb", width=2)

    label = ("blue = fitted half · pale = the circle it sweeps"
             if revolving else "blue = fitted half · pale = mirrored outline")
    d.text((8, 6), label, fill="#6b7280")
    return np.asarray(img)


# --------------------------------------------------------------------------- #
# Half + centreline mode
# --------------------------------------------------------------------------- #

AXIS_X_PX = CANVAS_W / 2.0


def centerline_axis():
    """The drawn centreline, as an Axis in millimetres.

    Fixed vertical through the middle of the canvas. Fixed rather than
    user-placed on purpose: a drawn axis would need a second input mode and a
    way to tell axis strokes from outline strokes, and the mode already gives
    us the one thing that matters - the axis is KNOWN rather than inferred.

    IT ALSO CARRIES THE VERTICAL DATUM. The axis runs from the BOTTOM of the
    canvas UPWARD, so a point's distance along it is its height above the
    canvas floor, and up on the screen is up in the model. Both halves of that
    matter: the direction used to run downward, which quietly flipped every
    revolve, and `kernel.half_to_radius_height` used to re-zero each profile to
    its own lowest point, which put every revolve at z=0 no matter where it was
    drawn. Together those made a cut drawn across the middle of a part come out
    at the top of it.
    """
    from sketch2stl.strokes import px_to_mm
    from sketch2stl.types import Axis
    floor = px_to_mm(np.array([[AXIS_X_PX, CANVAS_H]]))[0]     # canvas bottom
    top = px_to_mm(np.array([[AXIS_X_PX, 0.0]]))[0]
    return Axis.from_points(floor, top)


def centerline_background(size=(CANVAS_W, CANVAS_H)):
    """A canvas background with the centreline drawn on it, for half mode."""
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    x = int(AXIS_X_PX)
    for y in range(0, h, 16):                       # dashed
        d.line([(x, y), (x, min(y + 9, h))], fill=(120, 140, 200), width=2)
    d.text((x + 8, 6), "centreline - draw one half against this",
           fill=(120, 140, 200))
    return np.asarray(img)


def blank_background(size=(CANVAS_W, CANVAS_H)):
    from PIL import Image
    return np.asarray(Image.new("RGB", size, "white"))


def ghost_background(profiles, mode: str = "free", size=(CANVAS_W, CANVAS_H)):
    """The canvas background: what is already built, drawn faintly underneath.

    Added 29 Sep. Before this, the canvas was wiped blank after every feature,
    so the second shape was drawn blind - the user had to guess where it would
    land relative to the part they could see on the right. Now the committed
    profiles are shown as a pale outline, so "a hole in the middle of that
    plate" is something you can aim at instead of estimate.

    This also fixes a real bug: returning None to a Gradio Sketchpad leaves it
    spinning on a loading state forever. Returning a real image clears it.
    """
    from PIL import Image, ImageDraw
    from sketch2stl.strokes import mm_to_px

    w, h = size
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)

    for prof in profiles or ():
        try:
            px = mm_to_px(np.asarray(prof.outer, dtype=np.float64))
        except Exception:                                   # noqa: BLE001
            continue
        if len(px) < 3:
            continue
        pts = [(float(x), float(y)) for x, y in px]
        # Pale fill plus a slightly stronger edge: enough to aim at, not enough
        # to be mistaken for ink the recogniser will pick up.
        d.polygon(pts, fill=(238, 240, 246), outline=(198, 205, 222))
        for hole in getattr(prof, "holes", ()):  # inner rings punched back out
            hpx = mm_to_px(np.asarray(hole, dtype=np.float64))
            if len(hpx) >= 3:
                d.polygon([(float(x), float(y)) for x, y in hpx],
                          fill="white", outline=(198, 205, 222))

    if mode == "half":
        x = int(AXIS_X_PX)
        for y in range(0, h, 16):
            d.line([(x, y), (x, min(y + 9, h))], fill=(120, 140, 200), width=2)
        d.text((x + 8, 6), "centreline", fill=(120, 140, 200))

    return np.asarray(img)


# --------------------------------------------------------------------------- #
# The grid, and the fixed canvas backgrounds
#
# EVERY COLOUR IN HERE IS LIGHTER THAN 50% GREY ON PURPOSE. `_ink_mask`
# thresholds at 128, so anything paler than that is invisible to the
# recogniser - the grid can never be mistaken for a stroke. There is a test for
# it, because getting this wrong would make the app hallucinate shapes on an
# empty canvas.
# --------------------------------------------------------------------------- #

GRID_MINOR_MM = 5.0
GRID_MAJOR_MM = 25.0
GRID_MINOR = (238, 241, 246)
GRID_MAJOR = (221, 227, 238)
GRID_LABEL = (176, 185, 202)
AXIS_COLOUR = (150, 165, 210)


def _draw_grid(d, size=(CANVAS_W, CANVAS_H)) -> None:
    """A millimetre grid with a label every major line.

    The canvas has a fixed millimetres-per-pixel scale, so until now the size of
    a drawn shape was pure guesswork - you found out it was 44 mm across after
    you built it. With the grid you can draw a 40 mm circle on purpose.
    """
    from sketch2stl.config import PX_PER_MM
    w, h = size
    minor, major = GRID_MINOR_MM * PX_PER_MM, GRID_MAJOR_MM * PX_PER_MM

    x = 0.0
    while x <= w:
        d.line([(x, 0), (x, h)], fill=GRID_MINOR, width=1)
        x += minor
    y = float(h)
    while y >= 0:                      # from the BOTTOM: y=0 mm is the floor
        d.line([(0, y), (w, y)], fill=GRID_MINOR, width=1)
        y -= minor

    x = 0.0
    while x <= w:
        d.line([(x, 0), (x, h)], fill=GRID_MAJOR, width=1)
        d.text((x + 3, h - 13), f"{x / PX_PER_MM:.0f}", fill=GRID_LABEL)
        x += major
    y = float(h)
    while y >= 0:
        d.line([(0, y), (w, y)], fill=GRID_MAJOR, width=1)
        if y < h - 1:
            d.text((3, y + 2), f"{(h - y) / PX_PER_MM:.0f}", fill=GRID_LABEL)
        y -= major
    # Top-left, clear of the axis numbers, which collide with it along the
    # bottom edge once the last major line is labelled.
    d.text((5, 5), f"mm · grid {GRID_MINOR_MM:.0f}", fill=GRID_LABEL)


def _draw_centreline(d, size=(CANVAS_W, CANVAS_H), label: str = "centreline") -> None:
    w, h = size
    x = int(AXIS_X_PX)
    for y in range(0, h, 16):
        d.line([(x, y), (x, min(y + 9, h))], fill=AXIS_COLOUR, width=2)
    if label:
        d.text((x + 8, 6), label, fill=AXIS_COLOUR)


def grid_background(half: bool = False, size=(CANVAS_W, CANVAS_H)):
    """The canvas image, set ONCE when the app is built and never replaced.

    This is the whole reason the loading hang cannot come back: a Sketchpad that
    is in no event's output list can never be put into a pending state by a
    click. See the note on `Session.consumed` for how the ink is retired
    without writing to the canvas.
    """
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    _draw_grid(d, size)
    if half:
        _draw_centreline(d, size)
    return np.asarray(img)


def render_clicks(points_px, closed: bool, half: bool,
                  size=(CANVAS_W, CANVAS_H)):
    """The click-to-place canvas: vertices dropped so far, joined up.

    Freehand is the wrong tool for a shape made of straight edges - you cannot
    draw a 40 mm line by hand, and the corner detector then has to recover what
    you meant. Clicking the corners states it exactly, and the grid makes the
    dimensions deliberate. Curves still want the pen.
    """
    from PIL import Image, ImageDraw
    w, h = size
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    _draw_grid(d, size)
    if half:
        _draw_centreline(d, size)

    # `or ()` is wrong for a numpy array - truthiness on an array raises - and
    # this is called with both a list of clicks and an (N, 2) array.
    pts = ([] if points_px is None else
           [(float(x), float(y)) for x, y in np.asarray(points_px).reshape(-1, 2)])
    if len(pts) > 1:
        ring = pts + [pts[0]] if closed else pts
        if closed:
            d.polygon(ring, fill=(238, 243, 252), outline=None)
        d.line(ring, fill="#2a6fdb", width=3, joint="curve")
    for i, (x, y) in enumerate(pts):
        r = 5 if i else 7
        d.ellipse([x - r, y - r, x + r, y + r], fill="#ffffff",
                  outline="#2a6fdb", width=3 if i == 0 else 2)

    if not pts:
        d.text((14, 14), "Click a corner to start. Each click adds a point and "
                         "joins it to the last.", fill=(120, 130, 150))
    return np.asarray(img)
