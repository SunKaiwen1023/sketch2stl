import numpy as np
import pytest

from sketch2stl.strokes import (arc_length, close_ring, is_closed, mm_to_px,
                                normalize, px_to_mm, resample)


def test_px_mm_roundtrip():
    pts = np.array([[0.0, 0.0], [100.0, 50.0], [639.0, 479.0]])
    assert np.allclose(mm_to_px(px_to_mm(pts)), pts)


def test_resample_fixed_length():
    line = np.column_stack([np.linspace(0, 10, 7), np.zeros(7)])
    out = resample(line, 64)
    assert out.shape == (64, 2)
    # evenly spaced along arc length
    d = np.diff(arc_length(out))
    assert np.allclose(d, d[0])


def test_resample_degenerate():
    assert resample(np.array([[1.0, 1.0]]), 8).shape == (8, 2)


def test_normalize_is_scale_invariant():
    a = np.column_stack([np.cos(np.linspace(0, 6.28, 40)), np.sin(np.linspace(0, 6.28, 40))])
    n1, _ = normalize(a)
    n2, _ = normalize(a * 37.0 + 500.0)
    assert np.allclose(n1, n2, atol=1e-9)


def test_is_closed():
    ring = np.array([[0, 0], [1, 0], [1, 1], [0, 0]], dtype=float)
    assert is_closed(ring, tol=0.01)
    assert not is_closed(np.array([[0, 0], [5, 0]], dtype=float), tol=0.01)


def test_close_ring_is_idempotent():
    r = close_ring(np.array([[0, 0], [1, 0], [1, 1]], dtype=float))
    assert np.allclose(r[0], r[-1])
    assert len(close_ring(r)) == len(r)
