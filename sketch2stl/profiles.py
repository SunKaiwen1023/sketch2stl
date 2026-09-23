"""Recognised primitives -> closed, valid, extrudable Profiles.

OWNER: PK.

This is the bridge between the two halves, and it is where most of the pain in a
CAD-ish app lives. A user's strokes are never a clean closed loop: lines
overshoot, corners do not quite meet, a ring self-intersects near the start
point. Shapely's `buffer(0)` fixes a surprising amount of that for free.

WHAT WORKS TODAY: a single closed primitive becomes a Profile, with cleanup.
WHAT IS STUBBED: stitching several open strokes into one loop. See the TODOs.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import LinearRing, Polygon
from shapely.geometry.polygon import orient

from .config import CLOSE_TOL_MM, SIMPLIFY_TOL_MM
from .strokes import close_ring
from .types import Primitive, PrimitiveKind, Profile


def _to_polygon(points: np.ndarray) -> Polygon | None:
    """Points -> a valid shapely Polygon, or None if it cannot be rescued."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    ring = close_ring(pts)
    try:
        poly = Polygon(LinearRing(ring))
    except Exception:
        return None

    if not poly.is_valid:
        # buffer(0) is the standard shapely trick for self-intersecting rings:
        # it re-runs the polygon builder and returns a valid result, which for a
        # bowtie means keeping the larger lobe. Good enough for hand-drawn input.
        poly = poly.buffer(0)

    if poly.is_empty or poly.geom_type == "MultiPolygon":
        if poly.geom_type == "MultiPolygon" and len(poly.geoms):
            poly = max(poly.geoms, key=lambda g: g.area)     # keep the biggest lobe
        else:
            return None

    if poly.area <= 0:
        return None
    return poly


def polygon_to_profile(poly: Polygon, simplify: float = 0.0) -> Profile:
    """Shapely Polygon -> our Profile, oriented CCW outside / CW holes.

    `simplify` defaults to OFF. Simplification is for the raw POLYLINE fallback,
    where the ring is hundreds of noisy hand-drawn points. A fitted circle or
    rectangle is already an exact tessellation, and running Douglas-Peucker over
    it only loses area - a 0.4 mm tolerance on a 10 mm circle costs 2.6 % of its
    area, which then shows up as a hole that is visibly too small in the STL.
    """
    if simplify > 0:
        poly = poly.simplify(simplify, preserve_topology=True)
    poly = orient(poly, sign=1.0)          # CCW exterior, CW interiors
    outer = np.asarray(poly.exterior.coords, dtype=np.float64)
    holes = tuple(np.asarray(r.coords, dtype=np.float64) for r in poly.interiors)
    return Profile(outer=outer, holes=holes)


def profile_from_primitive(prim: Primitive) -> Profile | None:
    """The common case: one closed stroke, one profile.

    Returns None if the primitive is not closeable - an open line on its own
    encloses nothing, so there is nothing to extrude. The UI should tell the
    user that rather than silently doing nothing.

    Only the POLYLINE fallback gets simplified; see polygon_to_profile.
    """
    poly = _to_polygon(prim.points)
    if poly is None:
        return None
    tol = SIMPLIFY_TOL_MM if prim.kind is PrimitiveKind.POLYLINE else 0.0
    return polygon_to_profile(poly, simplify=tol)


def profiles_from_primitives(prims: list[Primitive]) -> list[Profile]:
    """Every primitive that can stand alone as a closed region."""
    out = []
    for p in prims:
        prof = profile_from_primitive(p)
        if prof is not None:
            out.append(prof)
    return out


def profile_area(profile: Profile) -> float:
    """Signed-free area in mm^2, holes subtracted."""
    poly = Polygon(profile.outer, [h for h in profile.holes])
    return float(poly.buffer(0).area)


# --------------------------------------------------------------------------- #
# TODO (PK), in order:
#
#  1. STITCHING. Right now one stroke = one profile, so a rectangle drawn as four
#     separate lines produces nothing. Build a graph of primitive endpoints,
#     join any two within CLOSE_TOL_MM, and walk the graph for cycles. This is
#     the single biggest usability gap in the scaffold.
#
#  2. TRIM / EXTEND. Two lines that cross should meet at the intersection; two
#     that nearly meet should extend to touch. Shapely's `unary_union` on the
#     line segments followed by `polygonize` does most of this in two calls and
#     is worth trying before writing anything by hand.
#
#  3. CONSTRAINT SNAPPING. If two fitted lines are within a few degrees of
#     parallel or perpendicular, make them exactly so. This is what makes output
#     look like CAD instead of like a scan of a napkin.
#
#  4. A guard against absurd profiles - area under ~1 mm^2, or an aspect ratio
#     over ~200:1 - with a clear message rather than a downstream mesh failure.
# --------------------------------------------------------------------------- #
