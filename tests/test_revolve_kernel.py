"""Revolve, measured against shapes whose volume we can work out by hand.

The contract tests (test_revolve_contract.py) say what a revolve IS. These say
whether ours is right, which is a different question and the one the prof will
ask when he sees a ball on the screen.

Tolerances are loose on purpose: the profile is a polyline and the sweep is
ARC_SEGMENTS facets, so a revolved circle is a faceted ball that is always a
little under the true volume. Anything within about 1% is the discretisation,
not a bug.
"""
from __future__ import annotations

import numpy as np
import pytest

from sketch2stl.kernel import _clean_revolve_profile, build, revolve
from sketch2stl.profiles import close_to_axis, profile_from_points
from sketch2stl.types import Axis, Document, Feature, FeatureKind, Op

Y_AXIS = Axis.from_points([0.0, 0.0], [0.0, 1.0])


def half_arch(r=25.0, n=200, x0=0.0):
    """Half a circle standing on the axis - revolved, a ball."""
    th = np.linspace(-np.pi / 2, np.pi / 2, n)
    return np.column_stack([x0 + r * np.cos(th), r * np.sin(th)])


def half_rect(r=20.0, h=40.0, n=60):
    """Half a rectangle against the axis - revolved, a cylinder."""
    return np.vstack([
        np.column_stack([np.full(n, 1e-9), np.linspace(0, h, n)]),
        np.column_stack([np.linspace(0, r, n), np.full(n, h)]),
        np.column_stack([np.full(n, r), np.linspace(h, 0, n)]),
    ])


# --------------------------------------------------------------------------- #
# volumes
# --------------------------------------------------------------------------- #

def test_a_half_arch_revolves_into_a_ball():
    r = 25.0
    mesh = revolve(half_arch(r), Y_AXIS)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(4 / 3 * np.pi * r ** 3, rel=0.02)


def test_a_half_rectangle_revolves_into_a_cylinder():
    r, h = 20.0, 40.0
    mesh = revolve(half_rect(r, h), Y_AXIS)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(np.pi * r ** 2 * h, rel=0.02)


def test_a_stepped_half_keeps_its_step():
    """Two stacked cylinders. If the shoulder were averaged into a chamfer the
    volume would come out between the two, so this measures the fix."""
    pts = np.vstack([
        np.column_stack([np.linspace(0, 20, 40), np.zeros(40)]),      # base
        np.column_stack([np.full(40, 20.0), np.linspace(0, 10, 40)]),  # wall 1
        np.column_stack([np.linspace(20, 8, 40), np.full(40, 10.0)]),  # shoulder
        np.column_stack([np.full(40, 8.0), np.linspace(10, 30, 40)]),  # wall 2
        np.column_stack([np.linspace(8, 0, 40), np.full(40, 30.0)]),   # top
    ])
    mesh = revolve(pts, Y_AXIS)
    expect = np.pi * 20 ** 2 * 10 + np.pi * 8 ** 2 * 20
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(expect, rel=0.03)


def test_a_partial_revolve_sweeps_less_than_a_full_one():
    full = revolve(half_rect(), Y_AXIS)
    half = revolve(half_rect(), Y_AXIS, angle=np.pi)
    assert half.volume == pytest.approx(full.volume / 2, rel=0.05)


# --------------------------------------------------------------------------- #
# the sampling fix
# --------------------------------------------------------------------------- #

def test_a_densely_sampled_half_still_comes_out_watertight():
    """Regression, and the reason `_clean_revolve_profile` exists.

    A half traced off the canvas skeletoniser arrives as several hundred points.
    Where two of them land at the same height the sweep produced a zero-height
    band of degenerate triangles, and the ball came out NOT WATERTIGHT with
    36,000 faces. Nothing about the maths was wrong - it was the sampling.
    """
    dense = half_arch(25.0, n=900)
    dense = dense + np.random.default_rng(0).normal(0, 0.05, dense.shape)
    mesh = revolve(dense, Y_AXIS)
    assert mesh.is_watertight
    assert len(mesh.faces) < 12_000, "profile was not thinned"


def test_a_tube_keeps_its_bore():
    """The case that killed the height-sorted implementation.

    A tube's half is a rectangle standing away from the axis, so there are TWO
    radii at every height and the profile is not a function of height at all.
    Sorted by height it produced a watertight, plausible-looking solid with a
    fifth of the right volume, which is the worst kind of wrong.
    """
    r_in, r_out, h = 8.0, 20.0, 40.0
    ring = np.array([[r_in, 0.0], [r_out, 0.0], [r_out, h], [r_in, h], [r_in, 0.0]])
    mesh = revolve(ring, Y_AXIS)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(np.pi * (r_out ** 2 - r_in ** 2) * h, rel=0.02)


def test_a_cylinder_is_two_points_off_the_axis_and_that_is_allowed():
    """A cylinder's half touches the axis at both ends, so only two of its
    corners have any radius. An earlier guard demanded three and rejected the
    most ordinary turned part there is."""
    mesh = revolve(close_to_axis(np.array([[0.0, 0.0], [15.0, 0.0],
                                           [15.0, 30.0], [0.0, 30.0]]), Y_AXIS),
                   Y_AXIS)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(np.pi * 15 ** 2 * 30, rel=0.02)


def test_a_cone_revolves():
    ring = np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 50.0], [0.0, 0.0]])
    mesh = revolve(ring, Y_AXIS)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(np.pi * 20 ** 2 * 50 / 3, rel=0.03)


def test_the_normals_point_outwards_whichever_way_the_ring_was_drawn():
    """A user drawing a half clockwise instead of anticlockwise should not get
    an inside-out solid, which is watertight but has negative volume and slices
    into nothing."""
    ring = np.array([[5.0, 0.0], [20.0, 0.0], [20.0, 30.0], [5.0, 30.0], [5.0, 0.0]])
    for direction in (ring, ring[::-1]):
        mesh = revolve(direction, Y_AXIS)
        assert mesh.volume > 0
        assert mesh.is_watertight


def test_cleaning_does_not_flatten_the_whole_outline():
    """The first attempt chained points by the GAP to their neighbour, which
    merges a finely sampled outline into a single band - every neighbour is
    closer than the threshold, so the chain never breaks."""
    rh = np.column_stack([np.abs(np.sin(np.linspace(0, np.pi, 300))) * 20,
                          np.linspace(0, 50, 300)])
    cleaned = _clean_revolve_profile(rh)
    assert len(cleaned) > 5


# --------------------------------------------------------------------------- #
# through the document
# --------------------------------------------------------------------------- #

def test_a_revolve_feature_builds_through_the_document():
    ring = close_to_axis(half_arch(25.0), Y_AXIS)
    prof = profile_from_points(ring)
    assert prof is not None
    doc = Document()
    doc.add(Feature("a", "Ball", Op.ADD, prof, depth=0.0,
                    kind=FeatureKind.REVOLVE, axis=Y_AXIS))
    mesh = build(doc)
    assert mesh is not None and mesh.is_watertight
    assert mesh.volume == pytest.approx(4 / 3 * np.pi * 25 ** 3, rel=0.03)


def test_the_axis_return_leg_does_not_fold_the_surface():
    """`close_to_axis` adds a leg of radius-0 points so the half satisfies
    Profile's closed-region contract. Sorted by height those would interleave
    with the real outline; revolve() has to drop them."""
    ring = close_to_axis(half_rect(), Y_AXIS)
    assert (np.abs(Y_AXIS.signed_distance(ring)) < 1e-6).sum() >= 2
    mesh = revolve(ring, Y_AXIS)
    assert mesh.is_watertight
    assert mesh.volume == pytest.approx(np.pi * 20 ** 2 * 40, rel=0.03)


def test_a_half_sitting_on_the_centreline_is_refused_readably():
    from sketch2stl.kernel import KernelError
    flat = np.column_stack([np.zeros(20), np.linspace(0, 30, 20)])
    with pytest.raises(KernelError, match="no radius"):
        revolve(flat, Y_AXIS)


def test_a_revolve_can_cut():
    """A groove is a revolve-and-cut. Op and FeatureKind are independent."""
    from sketch2stl.kernel import extrude
    from sketch2stl.types import Profile
    block = Profile(np.array([[-30.0, 0], [30, 0], [30, 60], [-30, 60], [-30, 0]]))
    doc = Document()
    doc.add(Feature("a", "Block", Op.ADD, block, depth=60.0))
    doc.add(Feature("b", "Bore", Op.CUT,
                    profile_from_points(close_to_axis(half_rect(10.0, 80.0), Y_AXIS)),
                    depth=0.0, kind=FeatureKind.REVOLVE, axis=Y_AXIS))
    solid = build(doc)
    plain = extrude(block, 60.0)
    assert solid.volume < plain.volume
