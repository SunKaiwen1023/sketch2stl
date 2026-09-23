import numpy as np
import pytest

from sketch2stl.kernel import KernelError, build, extrude, stats
from sketch2stl.types import Document, Feature, Op, Profile

SQUARE = Profile(outer=np.array([[0, 0], [80, 0], [80, 60], [0, 60], [0, 0]], dtype=float))


def circle_profile(cx, cy, r, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ring = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return Profile(outer=np.vstack([ring, ring[:1]]))


def feat(name, op, profile, depth, z=0.0):
    return Feature(feature_id=name, name=name, op=op, profile=profile, depth=depth, z_base=z)


def test_extrude_volume_is_correct():
    m = extrude(SQUARE, 20.0)
    assert m.is_watertight
    assert abs(m.volume - 80 * 60 * 20) < 1.0


def test_extrude_rejects_zero_depth():
    with pytest.raises(KernelError):
        extrude(SQUARE, 0.0)


def test_cut_removes_the_right_volume():
    doc = Document(features=[
        feat("plate", Op.ADD, SQUARE, 20.0),
        feat("hole", Op.CUT, circle_profile(40, 30, 15), 20.0),
    ])
    mesh = build(doc)
    expected = 80 * 60 * 20 - np.pi * 15 ** 2 * 20
    assert mesh.is_watertight
    assert abs(mesh.volume - expected) / expected < 0.01     # 1% for tessellation


def test_cut_before_any_body_is_a_clear_error():
    doc = Document(features=[feat("hole", Op.CUT, circle_profile(40, 30, 5), 10.0)])
    with pytest.raises(KernelError, match="nothing to cut"):
        build(doc)


def test_empty_document_builds_nothing():
    assert build(Document()) is None


def test_hidden_features_are_skipped():
    f = feat("plate", Op.ADD, SQUARE, 20.0)
    doc = Document(features=[Feature(**{**f.__dict__, "visible": False})])
    assert build(doc) is None


def test_build_is_deterministic():
    doc = Document(features=[feat("plate", Op.ADD, SQUARE, 20.0),
                             feat("hole", Op.CUT, circle_profile(40, 30, 15), 20.0)])
    assert abs(build(doc).volume - build(doc).volume) < 1e-6


def test_stats_reports_size():
    s = stats(extrude(SQUARE, 20.0))
    assert s["watertight"] is True
    assert abs(s["size_mm"][0] - 80) < 0.01
