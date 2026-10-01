"""The 29 Sep contract revision: revolve, axes, and half profiles.

These tests are the specification PK's kernel work has to satisfy. They pass
without any revolve implementation, because they only exercise the contract.
"""
from __future__ import annotations

import numpy as np
import pytest

from sketch2stl.profiles import fold_to_one_side, mirror_half, profile_from_points
from sketch2stl.types import Axis, Feature, FeatureKind, Op, Profile

Y_AXIS = Axis.from_points([0.0, 0.0], [0.0, 1.0])
HALF = np.array([[10.0, 0], [40, 0], [40, 30], [10, 30], [10, 0]])


def test_axis_normalises_its_direction():
    a = Axis.from_points([0, 0], [0, 7])
    assert np.linalg.norm(a.direction) == pytest.approx(1.0)


def test_axis_rejects_a_zero_direction():
    with pytest.raises(ValueError):
        Axis(np.zeros(2), np.zeros(2))


def test_signed_distance_has_opposite_signs_on_opposite_sides():
    d = Y_AXIS.signed_distance(np.array([[10.0, 0], [-10.0, 0]]))
    assert d[0] * d[1] < 0


def test_crosses_detects_a_profile_straddling_the_axis():
    assert Y_AXIS.crosses(np.array([[-5.0, 0], [5.0, 0]]))
    assert not Y_AXIS.crosses(HALF)


# --------------------------------------------------------------------------- #
# Feature
# --------------------------------------------------------------------------- #

def test_an_old_style_feature_is_still_an_extrude():
    """Backward compatibility: nothing in PK's existing code should break."""
    f = Feature("f", "Old", Op.ADD, Profile(HALF), depth=10.0)
    assert f.kind is FeatureKind.EXTRUDE
    assert f.axis is None


def test_a_revolve_needs_an_axis():
    with pytest.raises(ValueError, match="needs an axis"):
        Feature("f", "R", Op.ADD, Profile(HALF), 0.0, kind=FeatureKind.REVOLVE)


def test_a_revolve_profile_may_not_cross_its_own_axis():
    """Swept through itself, such a profile makes a self-intersecting solid.
    Catching it at the contract beats failing deep inside trimesh."""
    straddling = np.array([[-10.0, 0], [40, 0], [40, 30], [-10, 30], [-10, 0]])
    with pytest.raises(ValueError, match="crosses its axis"):
        Feature("f", "R", Op.ADD, Profile(straddling), 0.0,
                kind=FeatureKind.REVOLVE, axis=Y_AXIS)


def test_a_valid_revolve_is_accepted():
    f = Feature("f", "R", Op.ADD, Profile(HALF), 0.0,
                kind=FeatureKind.REVOLVE, axis=Y_AXIS)
    assert f.angle == pytest.approx(2 * np.pi)


@pytest.mark.parametrize("angle", [0.0, -1.0, 7.0])
def test_a_revolve_angle_outside_zero_to_two_pi_is_rejected(angle):
    with pytest.raises(ValueError, match="revolve angle"):
        Feature("f", "R", Op.ADD, Profile(HALF), 0.0, kind=FeatureKind.REVOLVE,
                axis=Y_AXIS, angle=angle)


def test_mirror_extrude_also_needs_an_axis():
    with pytest.raises(ValueError, match="needs an axis"):
        Feature("f", "M", Op.ADD, Profile(HALF), 10.0,
                kind=FeatureKind.MIRROR_EXTRUDE)


# --------------------------------------------------------------------------- #
# half profiles
# --------------------------------------------------------------------------- #

def test_mirroring_a_half_doubles_its_width_and_closes_it():
    ring = mirror_half(HALF[:-1], Y_AXIS)
    assert np.allclose(ring[0], ring[-1]), "ring not closed"
    assert ring[:, 0].min() == pytest.approx(-40.0)
    assert ring[:, 0].max() == pytest.approx(40.0)


def test_a_mirrored_half_is_symmetric_about_the_axis():
    """Checked on the bounding box, not the point sum: the ring keeps one
    unpaired vertex where it starts, so the coordinates do not cancel."""
    ring = mirror_half(HALF[:-1], Y_AXIS)
    d = Y_AXIS.signed_distance(ring)
    assert abs(d.max() + d.min()) < 1e-6


def test_mirroring_a_half_that_stands_off_the_axis_spans_it():
    """Worth being explicit about, because it surprised me.

    HALF runs from x=10 to x=40, so it does not touch the axis. Its mirror runs
    from -40 to -10, and the ring that joins the two closes ACROSS the axis -
    giving one solid 80 wide, not two separate blocks 30 wide. Revolved, that
    same profile would be a tube. If a user wants two separate walls they draw
    two shapes.
    """
    ring = mirror_half(HALF[:-1], Y_AXIS)
    prof = profile_from_points(ring)
    assert prof is not None
    from sketch2stl.profiles import profile_area
    assert profile_area(prof) == pytest.approx(80 * 30, rel=0.05)


def test_overshooting_the_centreline_is_folded_back_not_rejected():
    """People overshoot the axis by a millimetre. Refusing the whole drawing
    for that would be unkind, and the Feature contract rejects a crossing
    profile, so fold instead of fail."""
    stray = np.array([[-2.0, 0], [40, 0], [40, 30], [-2.0, 30]])
    folded = fold_to_one_side(stray, Y_AXIS)
    assert folded[:, 0].min() == pytest.approx(0.0)
    assert not Y_AXIS.crosses(folded)


def test_mirror_half_needs_enough_points():
    with pytest.raises(ValueError):
        mirror_half(np.array([[1.0, 1.0]]), Y_AXIS)
