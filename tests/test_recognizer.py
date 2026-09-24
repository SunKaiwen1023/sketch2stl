import numpy as np
import pytest

from sketch2stl.recognizer.rules import RuleRecognizer, fit_circle, fit_line
from sketch2stl.strokes import mm_to_px
from sketch2stl.types import PrimitiveKind, Stroke

rng = np.random.default_rng(20260922)


def noisy_circle(cx, cy, r, n=120, sigma=0.3):
    a = np.linspace(0, 2 * np.pi, n)
    return np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)]) + rng.normal(0, sigma, (n, 2))


def test_fit_circle_recovers_parameters():
    cx, cy, r, res = fit_circle(noisy_circle(80, 60, 15))
    assert abs(cx - 80) < 0.5 and abs(cy - 60) < 0.5
    assert abs(r - 15) < 0.5
    assert res < 1.0


def test_fit_line_handles_vertical():
    pts = np.column_stack([np.full(40, 7.0), np.linspace(0, 20, 40)])
    p0, p1, res = fit_line(pts)
    assert res < 1e-6
    assert abs(p0[0] - 7.0) < 1e-6


def test_recognizes_a_hand_drawn_circle():
    prim = RuleRecognizer().recognize(Stroke(points=mm_to_px(noisy_circle(80, 60, 15))))
    assert prim.kind is PrimitiveKind.CIRCLE
    assert abs(prim.params["r"] - 15) < 1.0
    assert prim.confidence > 0.5


def test_recognizes_a_hand_drawn_line():
    pts = np.column_stack([np.linspace(20, 100, 60), np.full(60, 40.0)])
    pts = pts + rng.normal(0, 0.2, pts.shape)
    prim = RuleRecognizer().recognize(Stroke(points=mm_to_px(pts)))
    assert prim.kind is PrimitiveKind.LINE


def test_never_raises_on_junk():
    for pts in [np.zeros((0, 2)), np.zeros((2, 2)), np.full((5, 2), 3.0)]:
        prim = RuleRecognizer().recognize(Stroke(points=pts))
        assert prim is not None



def test_recognizes_a_rectangle():
    pts = []
    for (x0, y0), (x1, y1) in [((40, 30), (120, 30)), ((120, 30), (120, 90)),
                               ((120, 90), (40, 90)), ((40, 90), (40, 30))]:
        for s in np.linspace(0, 1, 50, endpoint=False):
            pts.append([x0 + (x1 - x0) * s, y0 + (y1 - y0) * s])
    prim = RuleRecognizer().recognize(Stroke(points=mm_to_px(np.asarray(pts))))
    assert prim.kind is PrimitiveKind.RECT
