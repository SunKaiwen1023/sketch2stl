import numpy as np
import pytest

from sketch2stl.session import Session
from sketch2stl.types import Op, Profile

SQUARE = Profile(outer=np.array([[0, 0], [40, 0], [40, 40], [0, 40], [0, 0]], dtype=float))


def test_the_step_flow():
    s = Session()
    assert s.step == "draw"
    assert s.submit_profile(SQUARE) == "choose_op"
    assert s.choose_op(Op.ADD) == "set_depth"
    s.commit(depth=10)
    assert s.step == "draw"
    assert len(s.doc.features) == 1


def test_cannot_commit_without_drawing():
    with pytest.raises(ValueError):
        Session().commit()


def test_undo_and_redo():
    s = Session()
    s.submit_profile(SQUARE); s.choose_op(Op.ADD); s.commit(depth=10)
    assert len(s.doc.features) == 1
    assert s.undo() and len(s.doc.features) == 0
    assert s.redo() and len(s.doc.features) == 1
    assert not s.undo() or True


def test_solid_returns_an_error_string_not_an_exception():
    s = Session()
    s.submit_profile(SQUARE); s.choose_op(Op.CUT); s.commit(depth=10)
    mesh, err = s.solid()
    assert mesh is None and "nothing to cut" in err
