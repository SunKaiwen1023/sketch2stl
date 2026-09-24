"""Tests for sketch2stl/symmetry.py - no UI, just Profiles in and out."""
import math

import numpy as np
import pytest
from shapely.affinity import rotate
from shapely.geometry import Point, Polygon, box

from sketch2stl.symmetry import MIN_SCORE, find_symmetry, reflect, symmetrize
from sketch2stl.types import Profile


def profile_of(poly: Polygon) -> Profile:
    return Profile(outer=np.asarray(poly.exterior.coords), holes=tuple(np.asarray(r.coords) for r in poly.interiors))

def poly_of(p: Profile) -> Polygon:
    return Polygon(p.outer, list(p.holes))

def controller(wobble: float = 0.0, seed: int = 0) -> Polygon:
    """A game-controller-like outline, 100 mm wide, symmetric about x = 50, drawn 'by hand'."""
    # grips hang below the bar, so the shape is left-right symmetric but NOT top-bottom symmetric
    body = box(25, 18, 75, 34).union(Point(20, 14).buffer(15)).union(Point(80, 14).buffer(15))
    if not wobble:
        return body
    rng = np.random.default_rng(seed)
    ring = np.asarray(body.exterior.coords)[:-1]
    s = np.linspace(0, 2 * math.pi, len(ring))
    ring = ring + np.column_stack([np.sin(3 * s + 1.0), np.cos(2 * s)]) * wobble + rng.normal(0, wobble / 4, ring.shape)
    return Polygon(ring).buffer(0)

def signed_area(ring: np.ndarray) -> float:
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x[:-1], y[1:]) - np.dot(x[1:], y[:-1]))

def mirror_iou(p: Profile) -> float:
    s = find_symmetry(p); g = poly_of(p); m = reflect(g, s.origin, s.angle)
    return g.intersection(m).area / g.union(m).area


def test_rectangle_is_perfectly_symmetric_on_an_axis():
    s = find_symmetry(profile_of(box(0, 0, 40, 20)))
    assert s.score > 0.999 and s.snapped and s.angle_deg in (0.0, 90.0)

def test_hand_drawn_controller_finds_the_vertical_axis():
    s = find_symmetry(profile_of(controller(wobble=1.2)))
    assert s.is_symmetric and s.score < 0.999        # symmetric, but not perfectly
    assert s.angle_deg == 90.0 and s.snapped
    assert abs(s.origin[0] - 50) < 2

def test_rotated_axis_is_found_and_not_snapped():
    shape = rotate(controller(), 30, origin=(50, 20))
    s = find_symmetry(profile_of(shape))
    assert not s.snapped and abs(s.angle_deg - 120.0) < 1.0   # the vertical axis rotated by 30

def test_lopsided_shape_is_not_offered():
    L = box(0, 0, 60, 10).union(box(0, 0, 10, 50)).union(Point(10, 45).buffer(8))
    s = find_symmetry(profile_of(L))
    assert not s.is_symmetric
    with pytest.raises(ValueError):
        symmetrize(profile_of(L))

@pytest.mark.parametrize("keep", ["average", "positive", "negative"])
def test_symmetrize_makes_it_exact_and_keeps_the_contract(keep):
    src = profile_of(controller(wobble=1.2, seed=3))
    out = symmetrize(src, keep=keep)
    assert mirror_iou(out) > 0.999                           # exactly symmetric now
    assert np.allclose(out.outer[0], out.outer[-1])          # closed ring
    assert signed_area(out.outer) > 0                        # counter-clockwise, per types.Profile
    area_in, area_out = poly_of(src).area, poly_of(out).area
    assert abs(area_out - area_in) / area_in < 0.08          # same part, not a different one
    assert poly_of(out).is_valid

def test_average_keeps_a_centered_hole():
    plate = controller().difference(Point(50, 26).buffer(5))
    src = profile_of(plate)
    out = symmetrize(src)
    assert len(out.holes) == 1
    assert all(signed_area(h) < 0 for h in out.holes)       # holes clockwise
    assert abs(poly_of(out).area - plate.area) / plate.area < 0.05

def test_keep_positive_uses_the_left_half_for_a_vertical_axis():
    # left grip bigger than the right one: keeping the left must give two big grips
    shape = box(25, 18, 75, 34).union(Point(20, 14).buffer(16)).union(Point(80, 14).buffer(14))
    src = profile_of(shape)
    s = find_symmetry(src)
    left = poly_of(symmetrize(src, s, keep="positive"))
    right = poly_of(symmetrize(src, s, keep="negative"))
    assert s.is_symmetric                     # 16 vs 14 mm grips still reads as "meant to be symmetric"
    assert left.area > right.area
