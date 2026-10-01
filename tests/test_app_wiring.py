"""The app callbacks, driven with a fake canvas instead of a browser.

Importing `app` builds the Blocks but never launches, so this is fast and needs
no server. It is worth having because most of what broke in this app was
wiring, not geometry, and wiring is exactly what a unit test of the geometry
will never catch.

THE CENTRAL INVARIANT here is that the Sketchpads are inputs and never outputs.
That is what makes the loading hang impossible rather than merely unlikely, and
`test_no_event_writes_to_a_sketchpad` is the test that keeps it true.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gradio")

import gradio as gr                                       # noqa: E402

import app as A                                           # noqa: E402
from sketch2stl.config import CANVAS_H, CANVAS_W          # noqa: E402
from sketch2stl.session import Session                    # noqa: E402
from ui.canvas import grid_background                     # noqa: E402


def pad(draw=None):
    """A Sketchpad payload, the way Gradio 6 sends it: fixed grid background,
    the user's ink on a transparent layer above it."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), (0, 0, 0, 0))
    if draw is not None:
        draw(ImageDraw.Draw(img))
    return {"layers": [np.asarray(img)], "composite": None,
            "background": grid_background(False)}


BLANK = pad()


class Click:
    """Stands in for gr.SelectData."""
    def __init__(self, xy):
        self.index = list(xy)


@pytest.fixture
def circle_pad():
    cx, cy, r = CANVAS_W / 2, CANVAS_H / 2, 90
    return pad(lambda d: d.ellipse([cx - r, cy - r, cx + r, cy + r],
                                   outline=(0, 0, 0, 255), width=3))


@pytest.fixture
def arch_pad():
    """Half an arch drawn against the centreline - a ball, revolved."""
    th = np.linspace(-np.pi / 2, np.pi / 2, 200)
    x, r = CANVAS_W / 2, 100
    pts = [(x + r * np.cos(t), CANVAS_H / 2 - r * np.sin(t)) for t in th]
    return pad(lambda d: d.line(pts, fill=(0, 0, 0, 255), width=3))


# --------------------------------------------------------------------------- #
# the canvas is never written to  (the loading hang, structurally)
# --------------------------------------------------------------------------- #

def test_no_event_writes_to_a_sketchpad():
    """THE test. Two earlier fixes for the loading hang each removed a real bug
    and each left the hang in place, because the hang is not about WHAT gets
    written to the canvas - it is about writing to it at all. A component in an
    event's output list is marked pending the moment the click is dispatched.

    So: no Sketchpad may appear in any event's outputs. Ever.
    """
    pads = {id(A.pad_free), id(A.pad_half)}
    for event in A.demo.fns.values():
        written = {id(c) for c in (event.outputs or [])}
        assert not (written & pads), (
            f"{getattr(event.fn, '__name__', event.fn)} writes to a Sketchpad - "
            f"that is what leaves the canvas spinning")


def test_the_sketchpads_are_still_read():
    """The other half of the invariant: inputs only, but genuinely inputs."""
    pads = {id(A.pad_free), id(A.pad_half)}
    assert any(pads & {id(c) for c in (e.inputs or [])}
               for e in A.demo.fns.values())


@pytest.mark.parametrize("call", [
    lambda c, s: A.on_add(c, BLANK, 10.0, 0.0, s),
    lambda c, s: A.on_cut(c, BLANK, 10.0, 0.0, s),
    lambda c, s: A.on_undo(s),
    lambda c, s: A.on_clear(c, BLANK, s),
    lambda c, s: A.on_add(BLANK, BLANK, 10.0, 0.0, s),        # empty canvas
    lambda c, s: A.on_cut(c, BLANK, 10.0, 0.0, Session()),    # nothing to cut
])
def test_every_callback_returns_a_complete_tuple(circle_pad, call):
    """A handler that raises returns nothing, which leaves every output of that
    event pending - including the 3D preview and the layer table."""
    session = Session()
    A.on_add(circle_pad, BLANK, 10.0, 0.0, session)
    out = call(circle_pad, session)
    assert len(out) == 6, "a short tuple leaves the rest of the page pending"
    assert isinstance(out[5], Session), "the session slot must always be filled"
    assert isinstance(out[2], str) and out[2], "the user gets told something"


def test_a_cut_with_nothing_to_cut_from_says_so_and_is_rolled_back(circle_pad):
    session = Session()
    out = A.on_cut(circle_pad, BLANK, 10.0, 0.0, session)
    assert "nothing to cut" in out[2].lower()
    assert session.doc.features == []


# --------------------------------------------------------------------------- #
# ink is retired, not wiped
# --------------------------------------------------------------------------- #

def test_a_shape_is_only_built_once(circle_pad):
    """The canvas keeps the drawing, so the app has to remember what it has
    already used. Otherwise every click would re-add the same circle."""
    session = Session()
    A.on_add(circle_pad, BLANK, 10.0, 0.0, session)
    out = A.on_add(circle_pad, BLANK, 10.0, 0.0, session)
    assert len(session.doc.features) == 1
    assert "nothing new" in out[2].lower()


def test_undo_puts_the_ink_back_in_play(circle_pad):
    """Otherwise undo leaves a drawing on screen that the app refuses to look
    at, which is worse than not having undo."""
    session = Session()
    A.on_add(circle_pad, BLANK, 10.0, 0.0, session)
    A.on_undo(session)
    A.on_add(circle_pad, BLANK, 10.0, 0.0, session)
    assert len(session.doc.features) == 1


def test_erasing_a_shape_makes_it_forgotten(circle_pad):
    """`consumed` can only ever be ink that is still on the canvas, so the
    eraser and the canvas bin icon both just work."""
    session = Session()
    A.on_add(circle_pad, BLANK, 10.0, 0.0, session)
    A.on_add(BLANK, BLANK, 10.0, 0.0, session)            # canvas wiped by the user
    assert session.consumed.get("ink-free") is None or \
        not session.consumed["ink-free"].any()
    A.on_add(circle_pad, BLANK, 10.0, 0.0, session)       # redrawn
    assert len(session.doc.features) == 2


def test_the_two_canvases_keep_separate_books(circle_pad, arch_pad):
    """Ink retired in free mode must not be retired in half mode."""
    session = Session()
    A.on_add(circle_pad, arch_pad, 10.0, 0.0, session, A.FREE_MODE)
    A.on_add(circle_pad, arch_pad, 10.0, 0.0, session, A.HALF_MODE,
             A.FREEHAND, A.REVOLVE_AS)
    assert len(session.doc.features) == 2


def test_start_over_does_not_touch_the_canvas(circle_pad):
    session = Session()
    A.on_add(circle_pad, BLANK, 10.0, 0.0, session)
    out = A.on_clear(circle_pad, BLANK, session)
    fresh = out[5]
    assert fresh.doc.features == []
    # The drawing is still there, and still ignored.
    assert "nothing new" in A.on_add(circle_pad, BLANK, 10.0, 0.0, fresh)[2].lower()


# --------------------------------------------------------------------------- #
# the grid
# --------------------------------------------------------------------------- #

def test_the_grid_is_invisible_to_the_recogniser():
    """Every grid colour is lighter than the 128 ink threshold on purpose. If
    that ever slipped, the app would hallucinate shapes on an empty canvas."""
    from ui.canvas import strokes_from_image
    for half in (False, True):
        img = grid_background(half)
        assert img.min() > 128, "a grid colour is dark enough to read as ink"
        assert strokes_from_image(img) == []


def test_an_untouched_canvas_reads_as_empty():
    out = A.on_preview(BLANK, BLANK, Session())
    assert "nothing new" in out[1].lower()


# --------------------------------------------------------------------------- #
# click to place
# --------------------------------------------------------------------------- #

def clicks(session, points, mode=A.FREE_MODE):
    for p in points:
        _, note, session = A.on_click_point(session, mode, A.CLICK, Click(p))
    return session, note


def test_clicked_corners_build_an_exact_polygon():
    """The point of the tool: no hand wobble, so no fitting, so exact
    dimensions. 80 x 60 mm because that is what was clicked."""
    session, _ = clicks(Session(), [(160, 400), (480, 400), (480, 160), (160, 160)])
    A.on_add(None, None, 10.0, 0.0, session, A.FREE_MODE, A.CLICK)
    mesh, err = session.solid()
    assert mesh is not None, err
    w, h, d = mesh.bounds[1] - mesh.bounds[0]
    assert (w, h, d) == pytest.approx((80.0, 60.0, 10.0), abs=0.01)
    assert len(mesh.faces) == 12, "an exact box is 12 triangles, not a fitted blob"


def test_the_running_note_gives_the_position_in_millimetres():
    """Clicking blind would be no better than drawing blind."""
    _, note = clicks(Session(), [(160, 400), (480, 400), (480, 160)])
    assert "mm" in note and "80 x 60" in note


def test_two_corners_are_not_a_shape_yet():
    session, note = clicks(Session(), [(100, 100), (200, 200)])
    assert "more before this is a shape" in note
    out = A.on_add(None, None, 10.0, 0.0, session, A.FREE_MODE, A.CLICK)
    assert "three corners" in out[2].lower()


def test_undo_last_corner_removes_exactly_one():
    session, _ = clicks(Session(), [(100, 400), (300, 400), (300, 200)])
    _, note, session = A.on_undo_point(session, A.FREE_MODE, A.CLICK)
    assert session.clicked["click-free"] == [(100.0, 400.0), (300.0, 400.0)]


def test_clearing_the_corners_starts_again():
    session, _ = clicks(Session(), [(100, 400), (300, 400), (300, 200)])
    _, _, session = A.on_clear_points(session, A.FREE_MODE, A.CLICK)
    assert session.clicked["click-free"] == []


def test_clicked_corners_are_consumed_by_a_build():
    session, _ = clicks(Session(), [(160, 400), (480, 400), (480, 160), (160, 160)])
    A.on_add(None, None, 10.0, 0.0, session, A.FREE_MODE, A.CLICK)
    assert session.clicked["click-free"] == []
    out = A.on_add(None, None, 10.0, 0.0, session, A.FREE_MODE, A.CLICK)
    assert len(session.doc.features) == 1
    assert "three corners" in out[2].lower()


def test_a_clicked_half_revolves_into_a_stepped_shaft():
    ax = CANVAS_W / 2
    session, _ = clicks(Session(), [(ax, 460), (ax + 70, 460), (ax + 70, 300),
                                    (ax + 35, 300), (ax + 35, 120), (ax, 120)],
                        mode=A.HALF_MODE)
    A.on_add(None, None, 10.0, 0.0, session, A.HALF_MODE, A.CLICK, A.REVOLVE_AS)
    mesh, err = session.solid()
    assert mesh is not None, err
    assert mesh.is_watertight
    expect = np.pi * 17.5 ** 2 * 40 + np.pi * 8.75 ** 2 * 45
    assert mesh.volume == pytest.approx(expect, rel=0.02)


# --------------------------------------------------------------------------- #
# revolve vs extrude, and where a revolve lands
# --------------------------------------------------------------------------- #

def test_half_mode_revolve_builds_a_ball(arch_pad):
    session = Session()
    A.on_add(BLANK, arch_pad, 10.0, 0.0, session, A.HALF_MODE, A.FREEHAND,
             A.REVOLVE_AS)
    mesh, err = session.solid()
    assert mesh is not None, err
    assert mesh.is_watertight
    sx, _sy, sz = mesh.bounds[1] - mesh.bounds[0]
    assert sz == pytest.approx(sx, rel=0.1), "a ball is as tall as it is wide"


def test_half_mode_extrude_builds_a_flat_plate(arch_pad):
    session = Session()
    A.on_add(BLANK, arch_pad, 10.0, 0.0, session, A.HALF_MODE, A.FREEHAND,
             A.EXTRUDE_AS)
    mesh, err = session.solid()
    assert mesh is not None, err
    assert (mesh.bounds[1] - mesh.bounds[0])[2] == pytest.approx(10.0, abs=0.01)


def test_a_revolve_lands_where_it_was_drawn_on_the_centreline():
    """The reported bug: a square drawn across the middle of a part cut the TOP
    of it. `half_to_radius_height` re-zeroed every profile to its own lowest
    point, so every revolve started at z=0 whatever the drawing said."""
    ax = CANVAS_W / 2
    body = pad(lambda d: d.line([(ax, 460), (ax + 80, 460), (ax + 80, 60), (ax, 60)],
                                fill=(0, 0, 0, 255), width=3))
    session = Session()
    A.on_add(BLANK, body, 10.0, 0.0, session, A.HALF_MODE, A.FREEHAND, A.REVOLVE_AS)
    groove = pad(lambda d: d.line(
        [(ax + 40, 280), (ax + 90, 280), (ax + 90, 220), (ax + 40, 220), (ax + 40, 280)],
        fill=(0, 0, 0, 255), width=3))
    A.on_cut(BLANK, groove, 10.0, 0.0, session, A.HALF_MODE, A.FREEHAND, A.REVOLVE_AS)

    mesh, err = session.solid()
    assert mesh is not None, err
    assert mesh.is_watertight

    def radius_at(z):
        sec = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
        return float(np.linalg.norm(sec.vertices[:, :2], axis=1).max())

    # The groove was drawn between canvas y 220 and 280, which is 50-65 mm up.
    assert radius_at(55) < radius_at(30), "the groove is not where it was drawn"
    assert radius_at(90) == pytest.approx(radius_at(30), rel=0.05), \
        "the top was cut instead"


def test_up_on_the_screen_is_up_in_the_model():
    """A wide base drawn at the BOTTOM of the canvas must come out as a wide
    base, not a wide top. The axis used to run down the screen."""
    ax = CANVAS_W / 2
    cone = pad(lambda d: d.line([(ax, 440), (ax + 90, 440), (ax, 120)],
                                fill=(0, 0, 0, 255), width=3))
    session = Session()
    A.on_add(BLANK, cone, 10.0, 0.0, session, A.HALF_MODE, A.FREEHAND, A.REVOLVE_AS)
    mesh, _ = session.solid()
    lo, hi = mesh.bounds[:, 2]
    near_base = mesh.section(plane_origin=[0, 0, lo + (hi - lo) * 0.15],
                             plane_normal=[0, 0, 1])
    near_top = mesh.section(plane_origin=[0, 0, lo + (hi - lo) * 0.85],
                            plane_normal=[0, 0, 1])
    base_r = np.linalg.norm(near_base.vertices[:, :2], axis=1).max()
    top_r = np.linalg.norm(near_top.vertices[:, :2], axis=1).max()
    assert base_r > top_r * 2, "the shape came out upside down"


@pytest.mark.parametrize("name,points,expect", [
    ("cylinder", [(0, 390), (70, 390), (70, 120), (0, 120)], None),
    ("cone", [(0, 390), (80, 390), (0, 120)], None),
    ("tube", [(30, 390), (70, 390), (70, 120), (30, 120), (30, 390)],
     np.pi * (17.5 ** 2 - 7.5 ** 2) * 67.5),
])
def test_the_ordinary_turned_parts_all_come_out_watertight(name, points, expect):
    """A ball was the only shape the first revolve was tested on. A cylinder was
    rejected outright for having only two points off the axis, and a tube came
    out at a fifth of its volume."""
    ax = CANVAS_W / 2
    p = pad(lambda d: d.line([(ax + x, y) for x, y in points],
                             fill=(0, 0, 0, 255), width=3))
    session = Session()
    A.on_add(BLANK, p, 10.0, 0.0, session, A.HALF_MODE, A.FREEHAND, A.REVOLVE_AS)
    mesh, err = session.solid()
    assert mesh is not None, f"{name}: {err}"
    assert mesh.is_watertight, name
    if expect:
        assert mesh.volume == pytest.approx(expect, rel=0.05), name


def test_the_same_drawing_gives_two_different_solids(arch_pad):
    a, b = Session(), Session()
    A.on_add(BLANK, arch_pad, 10.0, 0.0, a, A.HALF_MODE, A.FREEHAND, A.REVOLVE_AS)
    A.on_add(BLANK, arch_pad, 10.0, 0.0, b, A.HALF_MODE, A.FREEHAND, A.EXTRUDE_AS)
    assert a.solid()[0].volume > b.solid()[0].volume * 2


# --------------------------------------------------------------------------- #
# half mode shows its fit
# --------------------------------------------------------------------------- #

def test_half_mode_shows_what_it_fitted(arch_pad):
    """"What did I draw?" used to return an empty box in half mode - the one
    place in the app with no feedback."""
    _, note, shot, _ = A.on_preview(BLANK, arch_pad, Session(), A.HALF_MODE,
                                    A.FREEHAND, A.REVOLVE_AS)
    assert isinstance(shot, np.ndarray)
    assert shot.shape[:2] == (CANVAS_H, CANVAS_W)
    assert len(np.unique(shot.reshape(-1, 3), axis=0)) > 3
    assert "fitted as" in note


def test_a_drawn_arch_is_fitted_as_one_arc(arch_pad):
    _, note, _, _ = A.on_preview(BLANK, arch_pad, Session(), A.HALF_MODE,
                                 A.FREEHAND, A.REVOLVE_AS)
    assert "1 piece(s): arc" in note


# --------------------------------------------------------------------------- #
# suggestions, the view swap, and the layer table
# --------------------------------------------------------------------------- #

def test_the_preview_tells_the_user_what_it_thinks_and_how_sure(circle_pad):
    _, note, _, _ = A.on_preview(circle_pad, BLANK, Session())
    assert "Probably want to" in note
    assert "% confident" in note


def test_the_suggestion_flips_to_cut_once_there_is_a_body(circle_pad):
    session = Session()
    big = pad(lambda d: d.rectangle([40, 40, CANVAS_W - 40, CANVAS_H - 40],
                                    outline=(0, 0, 0, 255), width=3))
    A.on_add(big, BLANK, 10.0, 0.0, session)
    _, note, _, _ = A.on_preview(circle_pad, BLANK, session)
    assert "cut" in note.lower()


def test_the_view_swap_shows_one_surface_at_a_time():
    free = A.on_view_change(A.FREE_MODE, A.FREEHAND)
    assert [u["visible"] for u in free[:4]] == [True, False, False, False]
    half = A.on_view_change(A.HALF_MODE, A.FREEHAND)
    assert [u["visible"] for u in half[:4]] == [False, True, False, False]
    click = A.on_view_change(A.FREE_MODE, A.CLICK)
    assert [u["visible"] for u in click[:4]] == [False, False, True, True]
    assert A.on_view_change(A.HALF_MODE, A.CLICK)[4]["visible"] is True


def test_the_layer_table_says_revolve_when_it_is_a_revolve(arch_pad):
    """It said "Extrude (Cut)" for a revolved cut - and the layer table is the
    one place the user can check what a feature actually is."""
    session = Session()
    A.on_add(BLANK, arch_pad, 10.0, 0.0, session, A.HALF_MODE, A.FREEHAND,
             A.REVOLVE_AS)
    rows = A.to_rows(session.doc)
    assert "Revolve" in rows[0][2]
    assert rows[0][3] == "-", "a revolve has no depth to report"
