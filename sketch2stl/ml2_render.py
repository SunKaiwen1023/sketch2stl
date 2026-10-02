"""ML2's picture of a half-profile: the SAME render the model was trained on.

OWNER: PK. Vendored from notebooks/ML2_revolve_or_mirror.ipynb (`ml2_halves.py`,
also in the model repo pakiino/sketch2stl-revolve-or-mirror). If the training
render changes, this file must change with it - the model only understands
pictures drawn exactly this way.

Picture (3 x 128 x 128, uint8), centreline always vertical at the left:
    ch0  the outline the person drew (edges lying ON the axis are not drawn)
    ch1  the centreline
    ch2  the filled half
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

SIZE = 128
AXIS_X = 6
PAD = 6


def _frame(shape):
    allp = np.vstack([r for poly in shape for r in poly])
    w = max(float(allp[:, 0].max()), 1e-9)
    y0, y1 = float(allp[:, 1].min()), float(allp[:, 1].max())
    h = max(y1 - y0, 1e-9)
    s = min((SIZE - AXIS_X - PAD) / w, (SIZE - 2 * PAD) / h)
    cy = (y0 + y1) / 2
    return lambda p: np.stack([AXIS_X + p[:, 0] * s, SIZE / 2 - (p[:, 1] - cy) * s], 1)


def render(shape, line_w=2):
    """shape = list of polygons [[exterior, holes...], ...] in the axis frame. Returns uint8 (3, SIZE, SIZE)."""
    f = _frame(shape)
    allx = np.vstack([r for poly in shape for r in poly])[:, 0]
    tol = 1e-3 * max(float(allx.max()), 1e-9)
    c0, c1, c2 = (Image.new("L", (SIZE, SIZE), 0) for _ in range(3))
    d0, d1, d2 = ImageDraw.Draw(c0), ImageDraw.Draw(c1), ImageDraw.Draw(c2)
    d1.line([(AXIS_X, 0), (AXIS_X, SIZE - 1)], fill=255, width=1)
    for poly in shape:
        ext = f(poly[0]); d2.polygon([tuple(p) for p in ext], fill=255)
        for hole in poly[1:]:
            d2.polygon([tuple(p) for p in f(hole)], fill=0)
        for ring in poly:
            q = f(ring)
            for (a, b), (ra, rb) in zip(zip(q[:-1], q[1:]), zip(ring[:-1], ring[1:])):
                if ra[0] < tol and rb[0] < tol:      # this edge lies on the axis: nobody draws it
                    continue
                d0.line([tuple(a), tuple(b)], fill=255, width=line_w)
    return np.stack([np.asarray(c) for c in (c0, c1, c2)])


def shape_from_drawing(half_points, axis_point, axis_dir):
    """The app's half (open polyline, mm) + its Axis -> the same axis-frame shape render() expects."""
    p = np.asarray(half_points, float).reshape(-1, 2)
    d = np.asarray(axis_dir, float); d = d / np.linalg.norm(d)
    rel = p - np.asarray(axis_point, float)
    x = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
    y = rel @ d
    ring = np.stack([x, y], 1)
    # close it the way close_to_axis does: drop to the axis at both ends
    ring = np.vstack([[0.0, ring[0, 1]], ring, [0.0, ring[-1, 1]], [0.0, ring[0, 1]]])
    return [[ring]]
