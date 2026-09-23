import numpy as np

from sketch2stl.exporter import check, export_stl
from sketch2stl.kernel import extrude
from sketch2stl.types import Profile

SQUARE = Profile(outer=np.array([[0, 0], [80, 0], [80, 60], [0, 60], [0, 0]], dtype=float))
HUGE = Profile(outer=np.array([[0, 0], [500, 0], [500, 500], [0, 500], [0, 0]], dtype=float))


def test_a_normal_plate_has_no_problems():
    assert check(extrude(SQUARE, 20.0)) == []


def test_oversized_part_is_flagged():
    problems = check(extrude(HUGE, 20.0))
    assert any("build volume" in p for p in problems)


def test_thin_part_is_flagged():
    problems = check(extrude(SQUARE, 0.5))
    assert any("thinnest" in p for p in problems)


def test_export_writes_a_file(tmp_path):
    out = tmp_path / "part.stl"
    export_stl(extrude(SQUARE, 20.0), out)
    assert out.exists() and out.stat().st_size > 100
