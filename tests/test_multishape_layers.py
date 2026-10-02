"""1 Oct: several shapes in one go, holes, Make symmetric, ML2 deciding, and the
Figma-style layer panel. Driven with fake canvases, no browser."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gradio")

import app as A                                           # noqa: E402
from sketch2stl import suggest                            # noqa: E402
from sketch2stl.config import CANVAS_H, CANVAS_W          # noqa: E402
from sketch2stl.profiles import nest_profiles, profile_from_points  # noqa: E402
from sketch2stl.session import Session                    # noqa: E402
from sketch2stl.stitch import stitch_all                  # noqa: E402
from sketch2stl.suggest import Suggestion                 # noqa: E402
from sketch2stl.types import FeatureKind, Op              # noqa: E402
from tests.test_app_wiring import BLANK, pad              # noqa: E402


def seg(a, b, n=40):
    return np.linspace(np.asarray(a, float), np.asarray(b, float), n)


def circ(c, r, n=80, gap=0.02):
    t = np.linspace(0, 2 * np.pi * (1 - gap), n)
    return np.c_[c[0] + r * np.cos(t), c[1] + r * np.sin(t)]


def square(cx, cy, s):
    return np.array([[cx - s, cy - s], [cx + s, cy - s], [cx + s, cy + s],
                     [cx - s, cy + s], [cx - s, cy - s]], float)


# --------------------------------------------------------------- stitching
def test_stitch_all_keeps_every_shape_and_any_stroke_order():
    r = [seg((100, 100), (400, 100)), seg((400, 100), (400, 300)),
         seg((400, 300), (100, 300)), seg((100, 300), (100, 100))]
    strokes = [circ((180, 200), 40), r[2], r[0], circ((320, 200), 35), r[1][::-1], r[3]]
    shapes = stitch_all(strokes)
    assert sorted(c.n_strokes for c in shapes) == [1, 1, 4]
    assert all(c.closed for c in shapes)


def test_a_closed_shape_does_not_swallow_its_neighbour():
    shapes = stitch_all([circ((100, 100), 30), circ((175, 100), 30)])
    assert len(shapes) == 2


# --------------------------------------------------------------- nesting
def test_outlines_inside_an_outline_become_holes():
    plate = profile_from_points(square(50, 50, 40))
    h1, h2 = profile_from_points(circ((30, 50), 8)), profile_from_points(circ((70, 50), 8))
    out = nest_profiles([h1, plate, h2])
    assert len(out) == 1 and len(out[0].holes) == 2


def test_separate_outlines_stay_separate():
    a, b = profile_from_points(square(20, 20, 10)), profile_from_points(square(80, 20, 10))
    assert len(nest_profiles([a, b])) == 2


def test_island_inside_a_hole_is_material_again():
    out = nest_profiles([profile_from_points(square(50, 50, 40)),
                         profile_from_points(square(50, 50, 25)),
                         profile_from_points(square(50, 50, 10))])
    assert len(out) == 2


# --------------------------------------------------------------- the app
@pytest.fixture
def plate_with_holes_pad():
    def draw(d):
        d.rectangle([140, 120, 500, 360], outline=(0, 0, 0, 255), width=3)
        d.ellipse([200, 200, 280, 280], outline=(0, 0, 0, 255), width=3)
        d.ellipse([360, 200, 440, 280], outline=(0, 0, 0, 255), width=3)
    return pad(draw)


def test_plate_and_two_holes_drawn_together_build_one_holed_plate(plate_with_holes_pad):
    s = Session()
    out = A.on_add(plate_with_holes_pad, BLANK, 10.0, 0.0, s)
    assert len(s.doc.features) == 1, out[2]
    assert len(s.doc.features[0].profile.holes) == 2
    assert "2 hole" in out[2]


def test_preview_reads_every_shape(plate_with_holes_pad):
    _, note, shot, _ = A.on_preview(plate_with_holes_pad, BLANK, Session())
    assert note.count("(") >= 3            # three recognised shapes listed
    assert shot.shape == (CANVAS_H, CANVAS_W, 3)


def test_two_separate_shapes_are_two_layers_but_one_undo():
    def draw(d):
        d.ellipse([60, 60, 200, 200], outline=(0, 0, 0, 255), width=3)
        d.ellipse([400, 200, 540, 340], outline=(0, 0, 0, 255), width=3)
    s = Session()
    A.on_add(pad(draw), BLANK, 10.0, 0.0, s)
    assert len(s.doc.features) == 2
    s.undo()
    assert len(s.doc.features) == 0


# --------------------------------------------------------------- make symmetric
def test_make_symmetric_straightens_a_lopsided_drawing():
    # a T-ish outline whose stem is 2 px off-centre
    pts = [(150, 380), (490, 380), (490, 240), (400, 240), (400, 120), (250, 120),
           (250, 236), (150, 236), (150, 380)]
    p = pad(lambda d: d.line(pts, fill=(0, 0, 0, 255), width=3))
    _, note_off, _, _ = A.on_preview(p, BLANK, Session())
    _, note_on, _, _ = A.on_preview(p, BLANK, Session(), make_sym=True)
    assert "symmetric" in note_off
    assert "made symmetric" in note_on


# --------------------------------------------------------------- ML2 decides
def test_auto_lets_ml2_decide(arch_pad, monkeypatch):
    monkeypatch.setattr(A, "suggest_kind",
                        lambda half, axis: Suggestion("extrude", 0.9, "stub", source="ml"))
    s = Session()
    out = A.on_add(BLANK, arch_pad, 10.0, 0.0, s, A.HALF_MODE, A.FREEHAND, A.AUTO_AS)
    assert s.doc.features[0].kind is FeatureKind.EXTRUDE
    assert "ML2 chose" in out[2]

    monkeypatch.setattr(A, "suggest_kind",
                        lambda half, axis: Suggestion("revolve", 0.9, "stub", source="ml"))
    s2 = Session()
    A.on_add(BLANK, arch_pad, 10.0, 0.0, s2, A.HALF_MODE, A.FREEHAND, A.AUTO_AS)
    assert s2.doc.features[0].kind is FeatureKind.REVOLVE


def test_an_explicit_choice_overrides_ml2(arch_pad, monkeypatch):
    monkeypatch.setattr(A, "suggest_kind",
                        lambda half, axis: Suggestion("extrude", 0.99, "stub", source="ml"))
    s = Session()
    A.on_add(BLANK, arch_pad, 10.0, 0.0, s, A.HALF_MODE, A.FREEHAND, A.REVOLVE_AS)
    assert s.doc.features[0].kind is FeatureKind.REVOLVE


@pytest.fixture
def arch_pad():
    th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    x, r = CANVAS_W / 2, 100
    pts = [(x + r * np.cos(t), CANVAS_H / 2 - r * np.sin(t)) for t in th]
    return pad(lambda d: d.line(pts, fill=(0, 0, 0, 255), width=3))


def test_ml2_model_is_used_when_present(monkeypatch):
    """The real suggest_kind routes through the ONNX model if one is loaded."""
    class FakeSess:
        def run(self, _, feeds):
            assert feeds["image"].shape == (1, 3, 128, 128)
            return [np.array([[0.2, 0.8]], np.float32)]
    m = np.zeros((3, 1, 1), np.float32); sd = np.ones((3, 1, 1), np.float32)
    monkeypatch.setattr(suggest, "load_kind_model", lambda *a, **k: (FakeSess(), m, sd))
    from sketch2stl.types import Axis
    half = np.c_[np.full(30, 20.0), np.linspace(0, 60, 30)]
    sug = suggest.suggest_kind(half, Axis.from_points([0, 0], [0, 1]))
    assert sug.source == "ml" and sug.value == "revolve" and abs(sug.confidence - 0.8) < 1e-6


# --------------------------------------------------------------- layer panel
def _two_layers():
    s = Session()
    A.on_add(pad(lambda d: d.rectangle([100, 100, 500, 380], outline=(0, 0, 0, 255), width=3)),
             BLANK, 10.0, 0.0, s)
    p2 = pad(lambda d: (d.rectangle([100, 100, 500, 380], outline=(0, 0, 0, 255), width=3),
                        d.ellipse([260, 200, 340, 280], outline=(0, 0, 0, 255), width=3)))
    A.on_cut(p2, BLANK, 20.0, 0.0, s)
    return s


def test_layer_edit_changes_the_part_and_can_be_undone():
    s = _two_layers()
    assert [f.op for f in s.doc.features] == [Op.ADD, Op.CUT]
    base = s.doc.features[0]
    out = A.on_layer_apply(base.feature_id, "Base plate", "Add", 25.0, 0.0, True, s)
    assert s.doc.features[0].depth == 25.0 and s.doc.features[0].name == "Base plate"
    assert "updated" in out[2].lower()
    s.undo()
    assert s.doc.features[0].depth == 10.0


def test_layer_hide_delete_and_reorder():
    s = _two_layers()
    cut = s.doc.features[1]
    A.on_layer_apply(cut.feature_id, cut.name, "Cut", cut.depth, 0.0, False, s)
    assert s.doc.features[1].visible is False
    A.on_layer_move(cut.feature_id, -1, s)
    assert s.doc.features[0].feature_id == cut.feature_id
    A.on_layer_delete(cut.feature_id, s)
    assert len(s.doc.features) == 1


def test_layer_pick_fills_the_properties():
    s = _two_layers()
    f = s.doc.features[0]
    name, op, depth, z, vis, img = A.on_layer_pick(f.feature_id, s)
    assert (name, op, depth, vis) == (f.name, "Add", 10.0, True)
    assert img.shape == (CANVAS_H, CANVAS_W, 3)
