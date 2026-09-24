"""Rule-based recogniser: fit each primitive type, keep the best fit.

OWNER: Serena.

This is the baseline the ML arm has to beat. It is deliberately simple and
deliberately good - a weak baseline makes the ML result meaningless.

All five primitive types now fit. What remains is polish - see the TODOs.
"""
from __future__ import annotations

import numpy as np

from ..config import (ARC_SEGMENTS, CIRCLE_RESIDUAL_MM, CLOSE_TOL_MM,
                      LINE_RESIDUAL_MM, RECT_FILL_MIN)
from ..strokes import close_ring, is_closed, prepare
from ..types import Primitive, PrimitiveKind, Stroke
from .base import Recognizer


def fit_circle(points: np.ndarray) -> tuple[float, float, float, float]:
    """Algebraic (Kasa) circle fit. Returns (cx, cy, r, mean_radial_residual).

    Solves the linear system for x^2 + y^2 = 2*cx*x + 2*cy*y + (r^2 - cx^2 - cy^2),
    which is exact in one step - no iteration, no initial guess. Good enough for
    hand-drawn input, and fast enough to run on every stroke.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(pts))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = float(np.sqrt(max(c + cx ** 2 + cy ** 2, 0.0)))
    residual = float(np.abs(np.hypot(x - cx, y - cy) - r).mean())
    return float(cx), float(cy), r, residual


def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Total-least-squares line fit. Returns (p_start, p_end, mean_residual).

    Uses the first principal component rather than y = mx + c, so a vertical
    line does not blow up.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centre = pts.mean(axis=0)
    centred = pts - centre
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    t = centred @ direction
    residual = float(np.abs(centred - np.outer(t, direction)).sum(axis=1).mean())
    return centre + t.min() * direction, centre + t.max() * direction, residual


def circle_points(cx: float, cy: float, r: float, n: int = ARC_SEGMENTS) -> np.ndarray:
    """Tessellate a circle into a closed CCW ring."""
    a = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return close_ring(ring)


def fit_rect(points: np.ndarray) -> tuple[np.ndarray, dict, float] | None:
    """Minimum-area enclosing rectangle. Returns (ring, params, fill_ratio).

    `fill_ratio` is the stroke's own enclosed area divided by the rectangle's.
    A real rectangle scores near 1.0; a circle scores ~0.64 (pi/4 of its
    bounding square); a random blob scores lower still. That single number is
    what turns "here is a box around the points" into a decision.
    """
    try:
        from shapely.geometry import MultiPoint, Polygon
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rect = MultiPoint([tuple(p) for p in pts]).minimum_rotated_rectangle
        ring = np.asarray(rect.exterior.coords, dtype=np.float64)
        if len(ring) < 5 or rect.area <= 0:
            return None
        stroke_poly = Polygon(close_ring(pts))
        if not stroke_poly.is_valid:
            stroke_poly = stroke_poly.buffer(0)
        fill = float(stroke_poly.area / rect.area) if rect.area > 0 else 0.0
    except Exception:
        return None

    e0, e1 = ring[1] - ring[0], ring[2] - ring[1]
    w, h = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    if w <= 0 or h <= 0:
        return None
    centre = ring[:4].mean(axis=0)
    return ring, {"cx": float(centre[0]), "cy": float(centre[1]), "w": w, "h": h,
                  "angle": float(np.arctan2(e0[1], e0[0]))}, fill


def fit_arc(points: np.ndarray) -> tuple[np.ndarray, dict, float] | None:
    """Circle-fit an OPEN stroke and keep the swept portion.

    Rejected if the sweep is nearly a full turn - that means the user drew a
    circle and closed it sloppily, and calling it an arc would be pedantic.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    cx, cy, r, res = fit_circle(pts)
    if r <= 0 or res > CIRCLE_RESIDUAL_MM:
        return None

    a0 = float(np.arctan2(pts[0, 1] - cy, pts[0, 0] - cx))
    a1 = float(np.arctan2(pts[-1, 1] - cy, pts[-1, 0] - cx))
    mid = pts[len(pts) // 2]
    am = float(np.arctan2(mid[1] - cy, mid[0] - cx))

    # walk the direction that actually passes through the stroke's midpoint
    def sweep(start, end):
        return (end - start) % (2 * np.pi)

    if sweep(a0, am) > sweep(a0, a1):
        a0, a1 = a1, a0
    span = sweep(a0, a1)
    if span < np.deg2rad(20) or span > np.deg2rad(340):
        return None

    a = np.linspace(a0, a0 + span, 48)
    ring = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return ring, {"cx": cx, "cy": cy, "r": r, "a0": a0, "a1": a0 + span}, res


class RuleRecognizer(Recognizer):
    name = "rules"

    def recognize(self, stroke: Stroke) -> Primitive:
        pts = prepare(stroke)
        if pts is None:
            return Primitive(PrimitiveKind.POLYLINE, np.zeros((0, 2)),
                             confidence=0.0, source="rules")

        closed = is_closed(pts, CLOSE_TOL_MM)

        # --- closed shapes: try a circle first ----------------------------- #
        if closed:
            cx, cy, r, res = fit_circle(pts)
            if res <= CIRCLE_RESIDUAL_MM and r > 0:
                conf = float(np.clip(1.0 - res / CIRCLE_RESIDUAL_MM, 0.0, 1.0))
                return Primitive(
                    kind=PrimitiveKind.CIRCLE,
                    points=circle_points(cx, cy, r),
                    params={"cx": cx, "cy": cy, "r": r},
                    confidence=max(conf, 0.6),
                    source="rules",
                )
            rect = fit_rect(pts)
            if rect is not None:
                ring, params, fill = rect
                # A rectangle fills its own bounding box; a blob does not. The
                # fill ratio is what separates them, and it is scale-free.
                if fill >= RECT_FILL_MIN:
                    conf = float(np.clip((fill - RECT_FILL_MIN) / (1.0 - RECT_FILL_MIN),
                                         0.0, 1.0))
                    return Primitive(
                        kind=PrimitiveKind.RECT,
                        points=ring,
                        params=params,
                        confidence=max(conf, 0.6),
                        source="rules",
                    )

            return Primitive(
                kind=PrimitiveKind.POLYLINE,
                points=close_ring(pts),
                confidence=0.5,
                source="rules",
            )

        # --- open shapes: try a straight line ------------------------------ #
        p0, p1, res = fit_line(pts)
        if res <= LINE_RESIDUAL_MM:
            conf = float(np.clip(1.0 - res / LINE_RESIDUAL_MM, 0.0, 1.0))
            return Primitive(
                kind=PrimitiveKind.LINE,
                points=np.vstack([p0, p1]),
                params={"x1": p0[0], "y1": p0[1], "x2": p1[0], "y2": p1[1]},
                confidence=max(conf, 0.6),
                source="rules",
            )

        arc = fit_arc(pts)
        if arc is not None:
            ring, params, res = arc
            conf = float(np.clip(1.0 - res / CIRCLE_RESIDUAL_MM, 0.0, 1.0))
            return Primitive(
                kind=PrimitiveKind.ARC,
                points=ring,
                params=params,
                confidence=max(conf, 0.6),
                source="rules",
            )

        return Primitive(PrimitiveKind.POLYLINE, pts, confidence=0.4, source="rules")


# --------------------------------------------------------------------------- #
# TODO (Serena), in order of how much they buy you:
#
#  1. AXIS SNAPPING. If a fitted line is within ~5 degrees of horizontal or
#     vertical, snap it. Cheap, and it makes the output look dramatically more
#     intentional - which matters a lot in a demo.
#
#  2. DIMENSION ROUNDING. Snap a fitted radius of 14.8 mm to 15 mm. Same idea:
#     the user is drawing by hand but thinking in round numbers.
# --------------------------------------------------------------------------- #
