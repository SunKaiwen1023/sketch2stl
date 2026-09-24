"""Mirror symmetry for hand-drawn profiles: find the fold line, then make both halves match.

OWNER: PK (geometry half). Works on `Profile` only, so it sits between profiles.py and
kernel.py and never touches recognition.

    sym = find_symmetry(profile)          # where is the fold line, and how symmetric is it?
    if sym.is_symmetric:                  # the UI asks: "Looks symmetric (95%). Make it symmetric?"
        profile = symmetrize(profile, sym)

HOW THE FOLD LINE IS FOUND - "fold the paper":
  Try a fold line through the shape's centroid at every angle (1 degree steps, then a
  0.1 degree refinement). Mirror the shape across it and measure the overlap with the
  original as intersection-over-union. The best angle is the axis; then the axis slides
  sideways a little (+-10% of the size), because a lopsided drawing pulls the centroid off
  the true axis. The best overlap is the score.
  Axes within SNAP_DEG of vertical or horizontal snap to it, because that is almost
  always what a person meant.

HOW BOTH SIDES ARE MADE EQUAL:
  keep="average"   average the shape with its mirror image (mean of the two signed-distance
                   fields, then the zero contour). Nobody has to choose a side, and wobbles on
                   either side cancel out. Default.
  keep="positive" / "negative"
                   keep one half exactly as drawn and mirror it. "positive" is the side the
                   axis normal points to: the LEFT half for a vertical axis, the TOP half for
                   a horizontal one.
  Either way the final step is "clip to one half, mirror, union", so the output is exactly
  symmetric, not approximately.

No learning here - deliberately. It is fast, exact and explainable. Whether the person MEANT
the shape to be symmetric is a judgement call, which is why the UI asks rather than acting.

TODO(PK): move SNAP_DEG / MIN_SCORE / RASTER_RES into config.py in their own small PR
(config.py is a shared file - see CONTRIBUTING.md).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from .types import Profile

SNAP_DEG = 5.0          # an axis this close to vertical/horizontal snaps to it
MIN_SCORE = 0.85        # below this IoU we do not offer to symmetrize at all
RASTER_RES = 512        # pixels along the longest side for the signed-distance average


@dataclass(frozen=True)
class Symmetry:
    """A fold line through `origin` in direction `angle` (radians, 0 = horizontal axis)."""
    origin: tuple[float, float]
    angle: float
    score: float            # IoU of the shape with its mirror image, 0..1
    snapped: bool           # True if the axis was snapped to vertical/horizontal

    @property
    def is_symmetric(self) -> bool:
        return self.score >= MIN_SCORE

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle) % 180.0


# --------------------------------------------------------------------------- helpers
def _to_polygon(profile: Profile) -> Polygon:
    poly = Polygon(profile.outer, [h for h in profile.holes])
    return poly if poly.is_valid else poly.buffer(0)

def _largest(geom) -> Polygon:
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    return geom

def _to_profile(poly: Polygon) -> Profile:
    poly = orient(_largest(poly), sign=1.0)          # exterior CCW, holes CW
    outer = np.asarray(poly.exterior.coords, dtype=np.float64)
    holes = tuple(np.asarray(r.coords, dtype=np.float64) for r in poly.interiors if len(r.coords) >= 4)
    return Profile(outer=outer, holes=holes)

def reflect(geom, origin: tuple[float, float], angle: float):
    """Mirror a shapely geometry across the line through `origin` with direction `angle`."""
    c2, s2 = math.cos(2 * angle), math.sin(2 * angle)
    cx, cy = origin
    a, b, d, e = c2, s2, s2, -c2
    return affine_transform(geom, [a, b, d, e, cx - (a * cx + b * cy), cy - (d * cx + e * cy)])

def _iou(p: Polygon, q: Polygon) -> float:
    u = p.union(q).area
    return p.intersection(q).area / u if u > 0 else 0.0


# --------------------------------------------------------------------------- public API
def find_symmetry(profile: Profile, step_deg: float = 1.0) -> Symmetry:
    """Find the best mirror axis. Always returns one; check `.is_symmetric` before using it."""
    poly = _to_polygon(profile)
    c = poly.centroid; origin = (c.x, c.y)
    score = lambda deg, o=None: _iou(poly, reflect(poly, o or origin, math.radians(deg)))

    coarse = np.arange(0.0, 180.0, step_deg)
    best = max(coarse, key=score)
    fine = np.arange(best - step_deg, best + step_deg + 1e-9, step_deg / 10)
    best = float(max(fine, key=score)) % 180.0

    # a lopsided drawing pulls the centroid off the true axis: slide the axis sideways a little
    minx, miny, maxx, maxy = poly.bounds
    size = max(maxx - minx, maxy - miny)
    nrm = np.array([-math.sin(math.radians(best)), math.cos(math.radians(best))])
    shifts = np.linspace(-0.1, 0.1, 41) * size
    shift = max(shifts, key=lambda t: score(best, tuple(np.array(origin) + nrm * t)))
    origin = tuple(float(v) for v in np.array(origin) + nrm * shift)
    best_score = score(best)

    # snap to the nearest of horizontal (0) / vertical (90) if close and nearly as good
    for target in (0.0, 90.0, 180.0):
        if abs(best - target) <= SNAP_DEG:
            s = score(target % 180.0)
            if s >= best_score - 0.01:
                return Symmetry(origin, math.radians(target % 180.0), s, True)
    return Symmetry(origin, math.radians(best), best_score, False)


def symmetrize(profile: Profile, sym: Symmetry | None = None, keep: str = "average",
               force: bool = False) -> Profile:
    """Return an exactly mirror-symmetric version of `profile`.

    Raises ValueError if the shape is not symmetric enough (score < MIN_SCORE) unless
    `force=True` - averaging a clearly lopsided shape with its mirror destroys it.
    """
    if keep not in ("average", "positive", "negative"):
        raise ValueError(f"keep must be 'average', 'positive' or 'negative', got {keep!r}")
    sym = sym or find_symmetry(profile)
    if not sym.is_symmetric and not force:
        raise ValueError(f"shape is only {sym.score:.0%} symmetric (need {MIN_SCORE:.0%}); pass force=True to override")

    poly = _to_polygon(profile)
    base = _sdf_average(poly, reflect(poly, sym.origin, sym.angle)) if keep == "average" else poly
    side = "negative" if keep == "negative" else "positive"
    return _to_profile(_mirror_half(base, sym, side))


def _mirror_half(poly: Polygon, sym: Symmetry, side: str) -> Polygon:
    """Keep one half-plane of `poly` and union it with its own mirror image: exact symmetry."""
    minx, miny, maxx, maxy = poly.bounds
    big = 4 * max(maxx - minx, maxy - miny, 1e-6)
    # half-plane on the chosen side of the axis, as a huge box rotated onto the axis
    n = np.array([-math.sin(sym.angle), math.cos(sym.angle)]) * (1 if side == "positive" else -1)
    d = np.array([math.cos(sym.angle), math.sin(sym.angle)])
    o = np.array(sym.origin)
    corners = [o - d * big, o + d * big, o + d * big + n * big, o - d * big + n * big]
    half = poly.intersection(Polygon(corners))
    whole = half.union(reflect(half, sym.origin, sym.angle))
    whole = whole.buffer(1e-7).buffer(-1e-7)             # seal the seam on the axis
    return _largest(whole)


def _sdf_average(p: Polygon, q: Polygon) -> Polygon:
    """Zero contour of the mean signed-distance field of two shapes (a smooth 'average shape')."""
    from scipy.ndimage import distance_transform_edt
    from skimage.draw import polygon as fill
    from skimage.measure import find_contours

    minx, miny, maxx, maxy = p.union(q).bounds
    span = max(maxx - minx, maxy - miny)
    pad = 0.05 * span
    minx, miny = minx - pad, miny - pad
    h = span * 1.1 / RASTER_RES                            # model units per pixel
    W = int(math.ceil((maxx - minx + pad) / h)) + 2
    H = int(math.ceil((maxy - miny + pad) / h)) + 2

    def raster(poly: Polygon) -> np.ndarray:
        m = np.zeros((H, W), bool)
        for g in (poly.geoms if isinstance(poly, MultiPolygon) else [poly]):
            ext = np.asarray(g.exterior.coords)
            rr, cc = fill((ext[:, 1] - miny) / h, (ext[:, 0] - minx) / h, m.shape); m[rr, cc] = True
            for r in g.interiors:
                hole = np.asarray(r.coords)
                rr, cc = fill((hole[:, 1] - miny) / h, (hole[:, 0] - minx) / h, m.shape); m[rr, cc] = False
        return m

    def sdf(m: np.ndarray) -> np.ndarray:                   # negative inside, positive outside
        return distance_transform_edt(~m) - distance_transform_edt(m)

    avg = (sdf(raster(p)) + sdf(raster(q))) / 2.0
    rings = [np.column_stack([c[:, 1] * h + minx, c[:, 0] * h + miny]) for c in find_contours(avg, 0.0) if len(c) >= 4]
    polys = [Polygon(r).buffer(0) for r in rings]
    polys = [g for g in polys if g.area > 0]
    if not polys:
        return p
    outer = max(polys, key=lambda g: g.area)
    for g in polys:                                        # contours inside the outer one are holes
        if g is not outer and outer.contains(g.representative_point()):
            outer = outer.difference(g)
    return _largest(outer).simplify(h * 0.5, preserve_topology=True)
