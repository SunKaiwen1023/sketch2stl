import numpy as np
import pytest

from sketch2stl.data.synth import NEAT, SHAKY, TYPICAL, deviation, handdraw
from sketch2stl.recognizer import RuleRecognizer
from sketch2stl.strokes import mm_to_px
from sketch2stl.types import PrimitiveKind, Stroke


def circle(r=15.0, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([80 + r * np.cos(a), 60 + r * np.sin(a)])
    return np.vstack([p, p[:1]])


def line(n=32):
    return np.column_stack([np.linspace(20, 100, n), np.full(n, 40.0)])


def test_handdraw_stays_near_the_original():
    dev = deviation(handdraw(circle(), TYPICAL, np.random.default_rng(0)), circle())
    assert 0.05 < dev < 3.0, f"{dev} mm is not a plausible hand"


def test_shakier_style_deviates_more():
    rng = np.random.default_rng(1)
    neat = np.mean([deviation(handdraw(circle(), NEAT, rng), circle()) for _ in range(8)])
    shaky = np.mean([deviation(handdraw(circle(), SHAKY, rng), circle()) for _ in range(8)])
    assert shaky > neat * 1.5


def test_deviation_is_measured_against_the_curve_not_its_endpoints():
    # a straight line arrives as 2 points; naively comparing against those two
    # reports the distance to the ENDS and inflates the number ~10x
    assert deviation(line(), np.array([[20.0, 40.0], [100.0, 40.0]])) < 0.5


def test_synthetic_strokes_are_still_recognisable():
    """If the rules arm cannot read them, the distortion model is too aggressive."""
    r, rng = RuleRecognizer(), np.random.default_rng(3)
    hits = sum(r.recognize(Stroke(points=mm_to_px(handdraw(circle(), TYPICAL, rng)))).kind
               is PrimitiveKind.CIRCLE for _ in range(20))
    assert hits >= 15, f"only {hits}/20 synthetic circles were readable"


def test_closed_shapes_stay_roughly_closed():
    rng = np.random.default_rng(4)
    gaps = [np.linalg.norm(s[0] - s[-1]) for s in
            (handdraw(circle(), TYPICAL, rng, closed=True) for _ in range(20))]
    assert np.median(gaps) < 4.0


def test_deviation_no_longer_scales_with_part_size():
    """A 2 m beam and a 3 mm pin are both drawn at canvas size by a real hand.

    Before to_canvas_scale existed, deviation ran 0.38 mm on a 4 mm circle and
    9.10 mm on a 2000 mm one - a 24x spread. The classifier would have learned
    "big shapes are clean", which is a pure artefact of the synthesis.
    """
    from sketch2stl.data.synth import to_canvas_scale

    rng = np.random.default_rng(0)
    devs = []
    for r in (2.0, 10.0, 50.0, 250.0, 1000.0):
        base = to_canvas_scale(circle(r), rng)
        devs.append(np.mean([deviation(handdraw(base, TYPICAL, rng), base)
                             for _ in range(5)]))
    devs = np.array(devs)
    assert devs.max() / devs.min() < 2.0, f"still size-dependent: {devs.round(3)}"
    assert devs.max() < 1.5, f"too shaky for a hand: {devs.round(3)}"


def test_canvas_scale_lands_inside_the_sheet():
    from sketch2stl.config import CANVAS_H, CANVAS_W, PX_PER_MM
    from sketch2stl.data.synth import to_canvas_scale

    rng = np.random.default_rng(1)
    for r in (2.0, 1000.0):
        out = to_canvas_scale(circle(r), rng)
        lo, hi = out.min(axis=0), out.max(axis=0)
        assert lo[0] > -1 and lo[1] > -1
        assert hi[0] < CANVAS_W / PX_PER_MM + 1 and hi[1] < CANVAS_H / PX_PER_MM + 1


def test_canvas_scale_preserves_shape():
    """Rescaling must not distort - that is handdraw's job."""
    from sketch2stl.data.synth import to_canvas_scale

    src = circle(37.0)
    out = to_canvas_scale(src, np.random.default_rng(2))
    c_src, c_out = src.mean(axis=0), out.mean(axis=0)
    r_src = np.linalg.norm(src - c_src, axis=1)
    r_out = np.linalg.norm(out - c_out, axis=1)
    assert r_src.std() / r_src.mean() == pytest.approx(r_out.std() / r_out.mean(), abs=1e-9)
