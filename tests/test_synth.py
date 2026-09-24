import numpy as np

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
