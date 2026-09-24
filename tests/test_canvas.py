"""Canvas tracing: raster ink -> ordered stroke.

The regression this guards is a bad one. The original tracer was a single greedy
nearest-neighbour walk that stopped at the first gap. On a shaky circle it
wandered into a skeleton spur, gave up, and returned 41 of 584 pixels - a stroke
spanning 3.8 mm instead of 30. The circle fit then reported r = 1.8 mm for a
15 mm circle, and the app cut a pinhole. Nothing raised; the number was just
wrong.
"""
import numpy as np
import pytest

from sketch2stl.recognizer import RuleRecognizer
from sketch2stl.recognizer.rules import fit_circle
from sketch2stl.strokes import prepare
from sketch2stl.types import PrimitiveKind
from ui.canvas import strokes_from_image

PIL = pytest.importorskip("PIL")
pytest.importorskip("skimage")
from PIL import Image, ImageDraw  # noqa: E402


def canvas(draw_fn):
    img = Image.new("RGB", (640, 480), "white")
    draw_fn(ImageDraw.Draw(img))
    return np.asarray(img)


def circle_img(noise=0.0, seed=4, width=2):
    rng = np.random.default_rng(seed)

    def d(dr):
        a = np.linspace(0, 2 * np.pi, 160)
        p = [(320 + 60 * np.cos(t) + rng.normal(0, noise),
              240 + 60 * np.sin(t) + rng.normal(0, noise)) for t in a]
        dr.line(p + [p[0]], fill="black", width=width, joint="curve")
    return canvas(d)


def rect_img(noise=0.0, seed=3, width=2):
    rng = np.random.default_rng(seed)

    def d(dr):
        p = []
        for (x0, y0), (x1, y1) in [((160, 120), (480, 120)), ((480, 120), (480, 360)),
                                   ((480, 360), (160, 360)), ((160, 360), (160, 120))]:
            for t in np.linspace(0, 1, 60, endpoint=False):
                p.append((x0 + (x1 - x0) * t + rng.normal(0, noise),
                          y0 + (y1 - y0) * t + rng.normal(0, noise)))
        dr.line(p + [p[0]], fill="black", width=width, joint="curve")
    return canvas(d)


@pytest.mark.parametrize("noise", [0.0, 3.0])
def test_traced_circle_keeps_its_size(noise):
    """The bug: 584 skeleton pixels traced to 41 points spanning 3.8 mm."""
    strokes = strokes_from_image(circle_img(noise))
    assert len(strokes) == 1
    mm = prepare(strokes[0])
    span = mm.max(axis=0) - mm.min(axis=0)
    assert 26 < span[0] < 36 and 26 < span[1] < 36, f"span {span} mm, expected ~30x30"
    _, _, r, _ = fit_circle(mm)
    assert 13 < r < 17, f"fitted r={r:.1f} mm, expected ~15"


@pytest.mark.parametrize("noise", [0.0, 3.0])
def test_traced_rectangle_keeps_its_size(noise):
    mm = prepare(strokes_from_image(rect_img(noise))[0])
    span = mm.max(axis=0) - mm.min(axis=0)
    assert 74 < span[0] < 88 and 54 < span[1] < 68, f"span {span} mm, expected ~80x60"


def test_shaky_shapes_still_recognise():
    r = RuleRecognizer()
    assert r.recognize(strokes_from_image(circle_img(3.0))[0]).kind is PrimitiveKind.CIRCLE
    assert r.recognize(strokes_from_image(rect_img(3.0))[0]).kind is PrimitiveKind.RECT


def test_blank_canvas_returns_nothing():
    assert strokes_from_image(canvas(lambda d: None)) == []
    assert strokes_from_image(None) == []


def test_two_shapes_are_two_strokes():
    def d(dr):
        dr.ellipse([60, 60, 200, 200], outline="black", width=2)
        dr.rectangle([380, 260, 560, 420], outline="black", width=2)
    assert len(strokes_from_image(canvas(d))) == 2
