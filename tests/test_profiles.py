import numpy as np

from sketch2stl.profiles import profile_area, profile_from_primitive
from sketch2stl.types import Primitive, PrimitiveKind


def ring(cx, cy, r, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.vstack([p, p[:1]])


def test_closed_circle_becomes_a_profile():
    prim = Primitive(PrimitiveKind.CIRCLE, ring(0, 0, 10))
    prof = profile_from_primitive(prim)
    assert prof is not None
    assert abs(profile_area(prof) - np.pi * 100) / (np.pi * 100) < 0.02


def test_open_line_is_not_a_profile():
    prim = Primitive(PrimitiveKind.LINE, np.array([[0.0, 0.0], [10.0, 0.0]]))
    assert profile_from_primitive(prim) is None


def test_self_intersecting_ring_is_repaired():
    bowtie = np.array([[0, 0], [10, 10], [10, 0], [0, 10], [0, 0]], dtype=float)
    prof = profile_from_primitive(Primitive(PrimitiveKind.POLYLINE, bowtie))
    assert prof is not None and profile_area(prof) > 0


def test_exterior_is_counter_clockwise():
    prof = profile_from_primitive(Primitive(PrimitiveKind.CIRCLE, ring(0, 0, 5)[::-1]))
    x, y = prof.outer[:, 0], prof.outer[:, 1]
    signed = 0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
    assert signed > 0
