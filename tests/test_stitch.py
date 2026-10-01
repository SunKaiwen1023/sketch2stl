"""Multi-stroke contours - the "sketch contour" work from the 29 Sep review.

The app assumed one shape = one stroke. Nobody draws that way: a rectangle is
four strokes, a big outline gets interrupted, a person lifts the pen to think.
"""
from __future__ import annotations

import numpy as np
import pytest

from sketch2stl.config import CIRCLE_RESIDUAL_MM, CLOSE_TOL_MM, RECT_FILL_MIN
from sketch2stl.corners import find_corners, segment_stroke
from sketch2stl.recognizer.rules import fit_circle, fit_rect
from sketch2stl.stitch import Contour, join_tolerance, smooth_seams, stitch
from sketch2stl.strokes import is_closed, prepare_points


def seg(a, b, n=40, noise=0.0, rng=None):
    p = np.linspace(np.asarray(a, float), np.asarray(b, float), n)
    return p + rng.normal(0, noise, p.shape) if noise and rng is not None else p


def test_single_stroke_passes_through_unchanged():
    p = seg([0, 0], [50, 0])
    c = stitch([p])
    assert c.n_strokes == 1
    assert c.seams == ()
    assert np.allclose(c.points, p)


def test_four_strokes_become_one_closed_rectangle():
    corners = [(20, 20), (90, 20), (90, 65), (20, 65)]
    strokes = [seg(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    c = stitch(strokes)
    assert c.n_strokes == 4
    assert c.closed
    assert len(c.seams) >= 3


def test_stroke_order_and_direction_do_not_matter():
    """People draw edges in any order and in either direction."""
    corners = [(20, 20), (90, 20), (90, 65), (20, 65)]
    strokes = [seg(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    rng = np.random.default_rng(2)
    shuffled = [s[::-1] if rng.random() < 0.5 else s
                for s in (strokes[i] for i in rng.permutation(4))]
    c = stitch(shuffled)
    assert c.n_strokes == 4
    assert c.closed


def test_a_four_stroke_rectangle_still_reads_as_a_rectangle():
    """The point of stitching: it must not turn every shape into a polyline."""
    corners = [(20, 20), (90, 20), (90, 65), (20, 65)]
    rng = np.random.default_rng(7)
    strokes = [seg(corners[i], corners[(i + 1) % 4], noise=0.5, rng=rng)
               for i in range(4)]
    c = stitch(strokes)
    fitted = fit_rect(prepare_points(c.points))
    assert fitted is not None
    assert fitted[2] >= RECT_FILL_MIN


def test_a_circle_drawn_in_two_halves_still_reads_as_a_circle():
    a = np.linspace(0, np.pi, 80)
    b = np.linspace(np.pi, 2 * np.pi, 80)
    top = np.column_stack([60 + 30 * np.cos(a), 60 + 30 * np.sin(a)])
    bottom = np.column_stack([60 + 30 * np.cos(b), 60 + 30 * np.sin(b)])
    c = stitch([bottom, top])              # deliberately out of order
    assert c.closed
    _, _, r, res = fit_circle(prepare_points(smooth_seams(c)))
    assert res < CIRCLE_RESIDUAL_MM
    assert r == pytest.approx(30.0, abs=2.0)


def test_seam_smoothing_stops_a_join_reading_as_a_corner():
    a = np.linspace(0, np.pi, 80)
    b = np.linspace(np.pi, 2 * np.pi, 80)
    top = np.column_stack([60 + 30 * np.cos(a), 60 + 30 * np.sin(a)])
    bottom = np.column_stack([60 + 30 * np.cos(b), 60 + 30 * np.sin(b)])
    c = stitch([top, bottom])
    assert find_corners(smooth_seams(c), closed=True) == []


def test_distant_strokes_are_not_joined():
    """Two separate shapes on the canvas must stay separate."""
    a = seg([10, 10], [40, 10])
    b = seg([110, 60], [140, 60])
    c = stitch([a, b])
    assert c.n_strokes == 1


def test_join_tolerance_scales_with_the_drawing():
    small = [seg([0, 0], [10, 0]), seg([10, 0], [10, 10])]
    big = [seg([0, 0], [300, 0]), seg([300, 0], [300, 300])]
    assert join_tolerance(big) > join_tolerance(small)


def test_an_L_in_two_strokes_keeps_its_corner():
    a = seg([20, 20], [70, 20], 50)
    b = seg([70, 20], [70, 60], 50)
    c = stitch([a, b])
    segs, corners = segment_stroke(c.points, seams=c.seams, closed=c.closed)
    assert len(segs) == 2
    assert corners


def test_empty_input_is_handled():
    c = stitch([])
    assert c.n_strokes == 0
    assert len(c.points) == 0


def test_contour_rejects_a_bad_shape():
    with pytest.raises(ValueError):
        Contour(np.zeros((5, 3)), False)
