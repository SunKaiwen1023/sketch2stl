"""Corner detection and piecewise fitting - the 29 Sep "star fix".

The bug being tested: a drawn star came out with rounded tips, because the
POLYLINE path resampled to 64 points and smoothed with a window of 5 before
anything looked for corners. Both steps destroy corners by design.
"""
from __future__ import annotations

import numpy as np
import pytest

from sketch2stl.corners import (find_corners, polyline_from_segments,
                                segment_stroke, turning_angles)
from sketch2stl.data.synth import MOUSE, NEAT, TYPICAL, handdraw
from sketch2stl.strokes import prepare_points
from sketch2stl.types import PrimitiveKind


def densify(ring, step=0.6):
    out = []
    for a, b in zip(ring[:-1], ring[1:]):
        k = max(2, int(np.linalg.norm(b - a) / step))
        out.append(np.linspace(a, b, k, endpoint=False))
    out.append(ring[-1:])
    return np.vstack(out)


def star(n=5, R=40.0, r=16.0, cx=80.0, cy=60.0):
    a = np.linspace(0, 2 * np.pi, 2 * n, endpoint=False) - np.pi / 2
    rad = np.where(np.arange(2 * n) % 2 == 0, R, r)
    p = np.column_stack([cx + rad * np.cos(a), cy + rad * np.sin(a)])
    return densify(np.vstack([p, p[:1]]))


def regular(n, R=35.0, cx=60.0, cy=45.0):
    a = np.linspace(0, 2 * np.pi, n + 1) - np.pi / 2
    return densify(np.column_stack([cx + R * np.cos(a), cy + R * np.sin(a)]))


def circle(r=30.0, cx=60.0, cy=60.0, n=240):
    a = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])


def ell(cx=20.0, cy=20.0, L=40.0):
    return np.vstack([
        np.column_stack([np.linspace(cx, cx + L, 60), np.full(60, cy)]),
        np.column_stack([np.full(60, cx + L), np.linspace(cy, cy + L, 60)]),
    ])


# --------------------------------------------------------------------------- #
# turning angle
# --------------------------------------------------------------------------- #

def test_straight_line_never_turns():
    line = np.column_stack([np.linspace(0, 60, 80), np.full(80, 30.0)])
    assert turning_angles(line).max() < 1.0


def test_star_tip_and_inner_angles_are_recovered():
    """A 5-point star turns 144 deg at a tip and 72 deg at an inner vertex."""
    pts = star()
    a = turning_angles(pts, closed=True)
    assert a.max() > 130, "tip angle lost"
    # Both families of corner must be present, or only the tips get found.
    assert ((a > 55) & (a < 90)).any(), "inner vertices lost"


# --------------------------------------------------------------------------- #
# corner counts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("shape,closed,want", [
    (star(), True, 10),
    (regular(3), True, 3),
    (regular(4), True, 4),
    (regular(6), True, 6),
    (ell(), False, 1),
    (circle(), True, 0),
])
def test_corner_counts_on_clean_strokes(shape, closed, want):
    assert len(find_corners(shape, closed=closed)) == want


@pytest.mark.parametrize("style", [NEAT, TYPICAL])
def test_star_survives_hand_tremor(style):
    """The star is the shape the prof flagged. It has to work with a real hand."""
    pts = handdraw(star(), style, np.random.default_rng(11), closed=True)
    assert len(find_corners(pts, closed=True)) == 10


def test_circle_has_no_corners_at_moderate_noise():
    pts = handdraw(circle(), TYPICAL, np.random.default_rng(4), closed=True)
    assert find_corners(pts, closed=True) == []


def test_closed_ring_finds_a_corner_sitting_on_the_seam():
    """Regression: the tangent window could not wrap, so index 0 was never
    scored and every closed polygon came back one corner short."""
    sq = regular(4)                       # first point IS a vertex
    assert len(find_corners(sq, closed=True)) == 4


# --------------------------------------------------------------------------- #
# segmentation and the fix itself
# --------------------------------------------------------------------------- #

def test_star_becomes_ten_straight_flanks():
    segs, corners = segment_stroke(star(), closed=True)
    assert len(corners) == 10
    assert len(segs) == 10
    assert all(s.kind is PrimitiveKind.LINE for s in segs)


def test_star_tips_keep_their_radius():
    """The whole point. The old path rounded the tips; this one must not."""
    centre = np.array([80.0, 60.0])
    segs, _ = segment_stroke(star(), closed=True)
    ring = polyline_from_segments(segs, closed=True)
    radius = np.linalg.norm(ring - centre, axis=1)
    assert radius.max() == pytest.approx(40.0, abs=0.5), "tip was cut"
    assert radius.min() == pytest.approx(16.0, abs=0.5), "inner vertex was cut"


def test_new_path_is_straight_flanks_not_a_smoothed_blob():
    """Documents the bug so nobody reintroduces the old behaviour.

    The difference is structural rather than a single angle. The old path is
    64 resampled, smoothed points with no notion of an edge - the corner is
    spread over roughly twenty of them, which is what a rounded tip IS. The new
    one is ten straight segments meeting at ten vertices.
    """
    pts = handdraw(star(), TYPICAL, np.random.default_rng(3), closed=True)

    old = prepare_points(pts)                        # resample 64 + smooth 5
    segs, _ = segment_stroke(pts, closed=True)
    new = polyline_from_segments(segs, closed=True)

    assert len(old) == 64
    assert len(new) <= 15, "the fitted ring should be vertices, not samples"

    # Every flank is genuinely a straight line, which the smoothed path has no
    # way to claim.
    assert all(s.kind is PrimitiveKind.LINE for s in segs)
    assert max(s.residual for s in segs) < 1.0

    # And the corner is concentrated, not smeared across many samples.
    spread_old = int((turning_angles(old, closed=True) > 80).sum())
    spread_new = int((turning_angles(new, closed=True) > 80).sum())
    assert spread_new < spread_old


def test_slot_is_split_at_its_tangent_junctions():
    """A slot's line-arc junctions turn through ZERO degrees, so no corner
    detector can find them. The recursive fit-and-split has to."""
    cx, cy, L, r = 60.0, 40.0, 40.0, 12.0
    th = np.linspace(-np.pi / 2, np.pi / 2, 40)
    right = np.column_stack([cx + L / 2 + r * np.cos(th), cy + r * np.sin(th)])
    th2 = np.linspace(np.pi / 2, 3 * np.pi / 2, 40)
    left = np.column_stack([cx - L / 2 + r * np.cos(th2), cy + r * np.sin(th2)])
    top = np.column_stack([np.linspace(cx + L / 2, cx - L / 2, 30), np.full(30, cy + r)])
    bot = np.column_stack([np.linspace(cx - L / 2, cx + L / 2, 30), np.full(30, cy - r)])
    ring = np.vstack([bot, right, top, left, bot[:1]])

    assert find_corners(ring, closed=True) == []      # genuinely no corners
    segs, _ = segment_stroke(ring, closed=True)
    kinds = {s.kind for s in segs}
    assert PrimitiveKind.ARC in kinds, "the round ends were not found"
    assert PrimitiveKind.LINE in kinds, "the straight flanks were not found"


def test_a_seam_is_treated_as_a_free_corner():
    """Where two strokes were joined, a person almost always lifted the pen at
    a corner. That index is passed in and must be honoured."""
    line = np.column_stack([np.linspace(0, 60, 100), np.full(100, 30.0)])
    segs, corners = segment_stroke(line, seams=(50,))
    assert 50 in corners
    assert len(segs) == 2


def test_short_stroke_is_handled_not_crashed():
    assert segment_stroke(np.array([[0.0, 0.0], [1.0, 1.0]])) == ([], [])
