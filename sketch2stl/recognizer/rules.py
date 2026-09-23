"""Rule-based recogniser: fit each primitive type, keep the best fit.

OWNER: Serena.

This is the baseline the ML arm has to beat. It is deliberately simple and
deliberately good - a weak baseline makes the ML result meaningless.

WHAT WORKS TODAY: circle fitting and the closed-polyline fallback. That is
enough for the app to run end to end, which is the point of the scaffold.
WHAT IS STUBBED: line, arc and rectangle fitting. See the TODOs.
"""
from __future__ import annotations

import numpy as np

from ..config import (ARC_SEGMENTS, CIRCLE_RESIDUAL_MM, CLOSE_TOL_MM,
                      LINE_RESIDUAL_MM)
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
            # TODO: rectangle fit goes here, before the polyline fallback.
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

        # TODO: arc fit goes here.
        return Primitive(PrimitiveKind.POLYLINE, pts, confidence=0.4, source="rules")


# --------------------------------------------------------------------------- #
# TODO (Serena), in order of how much they buy you:
#
#  1. RECTANGLE. Biggest win - most of the mockup is rectangles. Approach: take
#     the convex hull, find the minimum-area enclosing rectangle (rotating
#     calipers, or shapely's `minimum_rotated_rectangle`), and accept it if the
#     stroke's area is close to the rectangle's area. Fill in params
#     {cx, cy, w, h, angle}.
#
#  2. ARC. Fit a circle to an OPEN stroke, then take a0/a1 from the first and
#     last points. Accept only if the swept angle is under ~340 degrees,
#     otherwise the user meant a circle and closed it sloppily.
#
#  3. AXIS SNAPPING. If a fitted line is within ~5 degrees of horizontal or
#     vertical, snap it. Cheap, and it makes the output look dramatically more
#     intentional - which matters a lot in a demo.
#
#  4. DIMENSION ROUNDING. Snap a fitted radius of 14.8 mm to 15 mm. Same idea:
#     the user is drawing by hand but thinking in round numbers.
# --------------------------------------------------------------------------- #
