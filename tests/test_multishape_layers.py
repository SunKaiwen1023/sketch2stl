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


# --------------------------------------------------------------- layer list
def test_layer_list_shows_what_was_built():
    s = Session()
    A.on_add(pad(lambda d: d.rectangle([100, 100, 500, 380], outline=(0, 0, 0, 255), width=3)),
             BLANK, 10.0, 0.0, s)
    html = A._layer_html(s)
    assert "Add 1" in html and "add" in html and "10 mm" in html
    assert "Nothing built yet" in A._layer_html(Session())


def test_session_layer_edits_are_single_undo_steps():
    s = Session()
    A.on_add(pad(lambda d: d.rectangle([100, 100, 500, 380], outline=(0, 0, 0, 255), width=3)),
             BLANK, 10.0, 0.0, s)
    fid = s.doc.features[0].feature_id
    s.edit_feature(fid, depth=25.0)
    assert s.doc.features[0].depth == 25.0
    s.undo()
    assert s.doc.features[0].depth == 10.0


def test_hybrid_takes_the_circle_fit_when_ml_says_polyline():
    from sketch2stl.recognizer.hybrid import HybridRecognizer
    from sketch2stl.types import Primitive, PrimitiveKind, Stroke

    class Always:
        def __init__(self, kind): self.kind = kind
        def recognize(self, stroke):
            return Primitive(kind=self.kind, params={}, points=np.zeros((4, 2)), confidence=0.9)
    ring = np.c_[320 + 80 * np.cos(np.linspace(0, 2 * np.pi, 80)),
                 240 + 80 * np.sin(np.linspace(0, 2 * np.pi, 80))]
    h = HybridRecognizer(Always(PrimitiveKind.POLYLINE), Always(PrimitiveKind.CIRCLE))
    assert h.recognize(Stroke(points=ring)).kind is PrimitiveKind.CIRCLE
    quad = np.vstack([np.linspace(a, b, 20) for a, b in
                      [((200, 150), (420, 160)), ((420, 160), (400, 330)),
                       ((400, 330), (210, 320)), ((210, 320), (200, 150))]])
    assert h.recognize(Stroke(points=quad)).kind is PrimitiveKind.POLYLINE   # 4 corners: not a circle
    h = HybridRecognizer(Always(PrimitiveKind.ARC), Always(PrimitiveKind.CIRCLE))
    assert h.recognize(Stroke(points=np.zeros((10, 2)))).kind is PrimitiveKind.ARC


# --------------------------------------------------------------- stacking
def test_build_on_top_stacks_adds_and_cuts_down_from_the_top():
    def plate(d): d.rectangle([140, 120, 500, 360], outline=(0, 0, 0, 255), width=3)
    def boss(d): plate(d); d.ellipse([260, 180, 380, 300], outline=(0, 0, 0, 255), width=3)
    def pocket(d): boss(d); d.rectangle([160, 140, 220, 200], outline=(0, 0, 0, 255), width=3)
    def beside(d): pocket(d); d.ellipse([540, 60, 600, 120], outline=(0, 0, 0, 255), width=3)
    s = Session()
    A.on_add(pad(plate), BLANK, 10.0, 0.0, s, on_top=True)
    A.on_add(pad(boss), BLANK, 5.0, 0.0, s, on_top=True)            # circle ON the plate
    assert s.doc.features[1].z_base == 10.0
    A.on_cut(pad(pocket), BLANK, 4.0, 0.0, s, on_top=True)          # pocket INTO the plate
    assert s.doc.features[2].z_base == 6.0
    A.on_add(pad(beside), BLANK, 8.0, 0.0, s, on_top=True)          # off the part: on the table
    assert s.doc.features[3].z_base == 0.0
    mesh, _ = s.solid()
    assert abs(mesh.bounds[1][2] - 15.0) < 1e-6


def test_unticked_uses_start_z():
    def plate(d): d.rectangle([140, 120, 500, 360], outline=(0, 0, 0, 255), width=3)
    def boss(d): plate(d); d.ellipse([260, 180, 380, 300], outline=(0, 0, 0, 255), width=3)
    s = Session()
    A.on_add(pad(plate), BLANK, 10.0, 0.0, s, on_top=False)
    A.on_add(pad(boss), BLANK, 5.0, 3.0, s, on_top=False)
    assert s.doc.features[1].z_base == 3.0


# --------------------------------------------------------------- layer editing
class _Sel:
    def __init__(self, row): self.index = [row, 1]


def _two_layer_session():
    def plate(d): d.rectangle([140, 120, 500, 360], outline=(0, 0, 0, 255), width=3)
    def hole(d): plate(d); d.ellipse([260, 180, 380, 300], outline=(0, 0, 0, 255), width=3)
    s = Session()
    A.on_add(pad(plate), BLANK, 10.0, 0.0, s, on_top=True)
    A.on_cut(pad(hole), BLANK, 10.0, 0.0, s, on_top=True)
    return s


def test_layer_table_is_newest_on_top_and_click_selects_that_layer():
    s = _two_layer_session()
    t = A._layer_table(s)
    assert list(t["Layer"]) == ["Cut 1", "Add 1"] and list(t["Op"]) == ["cut", "add"]
    sel, name, op, depth, z, vis, note, _ = A.on_select_layer(s, _Sel(0))
    assert name == "Cut 1" and op == A.OP_CUT and sel == s.doc.features[1].feature_id
    assert A.on_select_layer(s, _Sel(1))[1] == "Add 1"


def test_swap_cut_to_add_rename_and_change_numbers():
    s = _two_layer_session()
    v0 = s.solid()[0].volume
    sel = s.doc.features[1].feature_id
    A.on_layer_apply(sel, "Cut 1", A.OP_ADD, 5.0, 10.0, True, s)
    assert s.doc.features[1].name == "Add 2"          # auto names follow the operation
    s.undo()
    A.on_layer_apply(sel, "Boss", A.OP_ADD, 5.0, 10.0, True, s)
    f = s.doc.features[1]
    assert (f.name, f.op, f.depth, f.z_base) == ("Boss", Op.ADD, 5.0, 10.0)
    assert s.solid()[0].volume > v0
    s.undo()
    assert s.doc.features[1].op is Op.CUT


def test_an_edit_that_breaks_the_part_is_rolled_back():
    s = _two_layer_session()
    out = A.on_layer_apply(s.doc.features[0].feature_id, "", A.OP_CUT, 10.0, 0.0, True, s)
    assert s.doc.features[0].op is Op.ADD          # a part made only of cuts cannot be built
    assert "Not changed" in out[2]


def test_delete_and_move_layers():
    s = _two_layer_session()
    cut = s.doc.features[1].feature_id
    out = A.on_layer_down(cut, s)                   # a cut below everything has nothing to cut
    assert "Not changed" in out[2] and s.doc.features[1].feature_id == cut
    assert "already at the bottom" in A.on_layer_down(s.doc.features[0].feature_id, s)[2]
    s = _two_layer_session()
    cut = s.doc.features[1].feature_id
    A.on_layer_delete(cut, s)
    assert [f.name for f in s.doc.features] == ["Add 1"]
    assert "Click a layer" in A.on_layer_delete(None, s)[2]
