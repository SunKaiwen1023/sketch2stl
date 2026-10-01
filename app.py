"""Sketch3D - draw a shape, add or cut it, export an STL.  Run:  python app.py

OWNER: shared. Keep it THIN - it wires widgets to functions that live
elsewhere. Every line of logic in here is a line you will both edit.

STATE lives in a gr.State, so each browser tab gets its own document. Never use
a module-level global for it - once this is on Spaces it would be shared across
every visitor at once.

The recogniser is chosen at startup: if a trained model exists at
RECOGNIZER_PATH it is used, otherwise the rules baseline. Train with
`scripts/train_recognizer.py` and restart - nothing else changes.

--- THE CANVAS IS NEVER WRITTEN TO ------------------------------------------ #

This is the one rule in this file that is not a preference. The Sketchpads are
created with a fixed background and appear in NO event's output list, ever.

The loading hang came from this: any component in an event's output list is put
into a pending state the moment the click is dispatched, and a Sketchpad in
Gradio 6 does not reliably come back out of it when handed a new image. Two
rounds of narrower fixes - stop returning None, never let a handler raise -
each removed a real bug and each left the hang in place, because the hang is
not caused by WHAT is written to the canvas. It is caused by writing to it.

So the canvas keeps the user's ink and the app remembers which strokes it has
already turned into features (`Session.consumed`). Your earlier shapes stay
visible, which also does the job the pale ghost underlay was doing - better,
because it is the actual drawing rather than a rendering of it.

Everything else on the page is an ordinary output and updates as usual.
"""
from __future__ import annotations

import functools
import os
import tempfile
import traceback
from pathlib import Path

import gradio as gr
import numpy as np

from sketch2stl.config import (BRUSH_PX, CANVAS_H, CANVAS_W, DEFAULT_DEPTH_MM,
                               LOW_CONFIDENCE, PX_PER_MM)
from sketch2stl.corners import Segment, polyline_from_segments, segment_stroke
from sketch2stl.exporter import check, export_stl
from sketch2stl.kernel import stats
from sketch2stl.profiles import (close_to_axis, fold_to_one_side, mirror_half,
                                 profile_area, profile_from_points,
                                 profile_from_primitive)
from sketch2stl.recognizer import RuleRecognizer
from sketch2stl.recognizer.ml import MLRecognizer
from sketch2stl.session import Session
from sketch2stl.stitch import smooth_seams, stitch
from sketch2stl.suggest import addcut_arm, suggest_kind, suggest_op
from sketch2stl.types import FeatureKind, Op, PrimitiveKind, Stroke
from ui.canvas import (centerline_axis, grid_background, ink_mask, render_clicks,
                       render_half, render_recognition, strokes_from_mask)
from ui.layers import HEADERS, to_rows
from ui.preview import mesh_to_preview_file

FREE_MODE = "Free draw"
HALF_MODE = "Half + centreline"

FREEHAND = "Sketch freehand"
CLICK = "Click the corners"

EXTRUDE_AS = "Extrude (mirror the half into a block)"
REVOLVE_AS = "Revolve (spin the half around the centreline)"

RECOGNIZER_PATH = os.environ.get("RECOGNIZER_PATH", "models/recognizer")

if (Path(RECOGNIZER_PATH) / "model.joblib").exists():
    RECOGNIZER = MLRecognizer(RECOGNIZER_PATH)
    ARM = f"learned model ({RECOGNIZER_PATH})"
else:
    RECOGNIZER = RuleRecognizer()
    ARM = "rules baseline (no trained model found - see scripts/train_recognizer.py)"

OP_ARM = addcut_arm()

print(f"recogniser:      {ARM}")
print(f"add/cut guess:   {OP_ARM}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _session(state) -> Session:
    """gr.State starts as None in Gradio 6 - see the module docstring."""
    return state if isinstance(state, Session) else Session()


def never_fails(n_outputs: int, status_at: int | None):
    """A callback that cannot return nothing, whatever happens inside it.

    When a Gradio event handler raises, Gradio returns no values at all and
    every component in that event's output list stays pending. That is a bad
    state for any component, so this is still worth having even though the
    canvas is no longer among them: a traceback would otherwise leave the 3D
    preview and the layer table stuck too, with no message explaining why.

    `gr.update()` means "leave this component exactly as it was".
    """
    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:                 # noqa: BLE001 - deliberate
                traceback.print_exc()
                out = [gr.update() for _ in range(n_outputs)]
                if status_at is not None:
                    out[status_at] = (
                        f"Something went wrong handling that click "
                        f"({type(exc).__name__}: {exc}). Nothing was changed - "
                        f"your drawing is untouched. The full details are in "
                        f"the terminal running the app.")
                return out[0] if n_outputs == 1 else tuple(out)
        return wrapper
    return decorate


# `outs` is [preview, layers, status, recognised, clicks, session].
build_cb = never_fails(6, 2)
preview_cb = never_fails(4, 1)
click_cb = never_fails(3, 1)          # [clicks, status, session]
view_cb = never_fails(5, None)        # the visibility swap
export_cb = never_fails(2, 1)         # [stl_file, status]


def _canvas_key(mode: str, style: str) -> str:
    """Which of the four drawing surfaces we are reading.

    Ink retired on one must not be retired on the others, or switching mode
    would make a drawing disappear from the app's point of view while staying
    on the screen.
    """
    return f"{'click' if style == CLICK else 'ink'}-{'half' if mode == HALF_MODE else 'free'}"


def _size_note(profile) -> str:
    """The bounding box of what was drawn, in mm.

    Still shown even though the canvas now has a labelled grid: the grid makes
    the size something you can aim at, and this is the confirmation you hit it.
    """
    pts = np.asarray(profile.outer, dtype=np.float64)
    w, h = pts.max(axis=0) - pts.min(axis=0)
    return f"{w:.0f} x {h:.0f} mm"


def _confidence_line(label: str, sug) -> str:
    """One line of suggestion, honest about how sure it is."""
    mark = "?" if sug.unsure else "-"
    return (f"\n\n{mark} **{label}: {sug.value}** ({sug.confidence:.0%} "
            f"confident) · {sug.reason}")


def _panels(session: Session, message: str = ""):
    """Rebuild the solid and return everything the right-hand side shows."""
    mesh, err = session.solid()
    rows = to_rows(session.doc)
    if mesh is None:
        # Show BOTH. `message or err` used to hide the error behind the cheerful
        # "Added as Cut 1", so a cut with nothing to cut from looked like it had
        # worked and simply produced no preview.
        both = "\n\n".join(x for x in (message, err) if x)
        return None, rows, both or "Draw a shape, then Add or Cut it."

    s = stats(mesh)
    lines = [
        message,
        f"{s['size_mm'][0]:.1f} x {s['size_mm'][1]:.1f} x {s['size_mm'][2]:.1f} mm"
        f"  ·  {s['volume_mm3'] / 1000:.1f} cm3  ·  {s['n_faces']} faces"
        f"  ·  {'watertight' if s['watertight'] else 'NOT WATERTIGHT'}",
    ]
    problems = check(mesh)
    if problems:
        lines.append("Printability:\n- " + "\n- ".join(problems))
    # Blank lines, not single newlines: this goes into a gr.Markdown, and
    # Markdown folds a single newline into a space, which ran the whole status
    # panel together into one paragraph.
    return mesh_to_preview_file(mesh), rows, "\n\n".join(x for x in lines if x)


# --------------------------------------------------------------------------- #
# reading the drawing surface
# --------------------------------------------------------------------------- #
def _strokes(pad_free, pad_half, session: Session, mode: str, style: str):
    """What is on the active surface that has NOT been built yet.

    Returns (strokes_in_pixels, ink_mask_or_None). The mask comes back so the
    caller can retire exactly what it read, without re-reading the canvas.
    """
    key = _canvas_key(mode, style)

    if style == CLICK:
        pts = session.clicked.get(key) or []
        if len(pts) < 3:
            return [], None
        return [np.asarray(pts, dtype=np.float64)], None

    mask = ink_mask(pad_half if mode == HALF_MODE else pad_free)
    found = strokes_from_mask(session.new_ink(key, mask))
    return [np.asarray(st.points, dtype=np.float64) for st in found], mask


def _line_segments(points: np.ndarray) -> list[Segment]:
    """Clicked corners are already exact, so every piece is a straight line.

    No fitting, no corner detection: the user said where the corners are. This
    just wraps them in the same `Segment` the freehand path produces so the
    preview renderer does not need to know which way the shape was drawn.
    """
    return [Segment(PrimitiveKind.LINE, np.vstack([a, b]), {}, 0.0, np.vstack([a, b]))
            for a, b in zip(points[:-1], points[1:])]


def _read(pad_free, pad_half, session: Session, mode: str, style: str,
          build_as: str):
    """The active surface -> (profile, description, preview_image, half_points).

    STITCHING (freehand). The canvas returns one entry per connected ink
    component, which is one per pen stroke. People draw an outline in several
    strokes, so those are joined into a single contour before recognition. A
    four-stroke rectangle comes out RECT, not POLYLINE.

    HALF MODE. The user draws one half against the centreline, then says what to
    do with it. Revolved it is a turned part; mirrored and extruded it is a
    block. The drawing does not distinguish them, so the person does.
    """
    from sketch2stl.strokes import prepare, px_to_mm

    clicking = style == CLICK
    strokes, _mask = _strokes(pad_free, pad_half, session, mode, style)
    if not strokes:
        empty = ("Click at least three corners." if clicking else
                 "Nothing new on the canvas - draw a shape. Anything already "
                 "built has been used up; draw the next one beside it.")
        return None, empty, None, None

    stitched = stitch([np.asarray(s, dtype=np.float64) for s in strokes])
    joined_note = ""
    if stitched.n_strokes > 1:
        joined_note = f"Joined {stitched.n_strokes} strokes into one outline. "

    if mode == HALF_MODE:
        # The strokes are in PIXELS; centerline_axis() is in MILLIMETRES.
        # Convert before any geometry touches the axis.
        axis = centerline_axis()
        raw = px_to_mm(smooth_seams(stitched) if not clicking else stitched.points)
        if len(raw) < 3:
            return None, "Draw one half of the outline against the centreline.", None, None

        if clicking:
            segments, fitted = _line_segments(raw), raw
        else:
            # FIT THE HALF. It used to go to the kernel as raw traced pixels,
            # which is why a hand-drawn ball arrived as ~300 wobbly points and
            # why this mode had nothing to show in the preview: there was no fit
            # to show. Same segmenter the free-draw path uses.
            segments, _corners = segment_stroke(raw, seams=stitched.seams, closed=False)
            fitted = polyline_from_segments(segments)
            if len(fitted) < 3:                  # fitting gave up; use the trace
                segments, fitted = [], raw
        half = fold_to_one_side(fitted, axis)

        revolving = build_as == REVOLVE_AS
        ring = close_to_axis(half, axis) if revolving else mirror_half(half, axis)
        what = "Revolve" if revolving else "Mirrored half"

        profile = profile_from_points(ring)
        if profile is None:
            shot = render_half(raw, segments, None, revolving)
            return None, (joined_note + "That half did not close into a region. "
                          "Start and finish near the centreline."), shot, None

        kinds = [s.kind.value for s in segments]
        shape_note = (f", fitted as {len(segments)} piece(s): {', '.join(kinds)}"
                      if kinds else "")
        note = (joined_note + f"{what} profile, {profile_area(profile):.0f} mm2, "
                f"{_size_note(profile)}{shape_note}")
        note += _confidence_line("Looks like a", suggest_kind(half, axis))
        return profile, note, render_half(raw, segments, half, revolving), half

    if clicking:
        # Clicked corners describe the outline exactly, so there is nothing for
        # the recogniser to guess. Snapping four clicks to a "rectangle" would
        # only be able to make them less true.
        ring = px_to_mm(stitched.points)
        if np.linalg.norm(ring[0] - ring[-1]) > 1e-9:
            ring = np.vstack([ring, ring[:1]])
        profile = profile_from_points(ring)
        if profile is None:
            return None, ("Those corners do not enclose an area - they are in a "
                          "line, or the outline crosses itself."), None, None
        note = (f"Polygon from {len(ring) - 1} clicked corners, "
                f"{profile_area(profile):.0f} mm2, {_size_note(profile)}")
        return profile, note, render_clicks(stitched.points, True, False), None

    prims = RECOGNIZER.recognize_batch(
        [Stroke(points=smooth_seams(stitched))] if stitched.n_strokes > 1
        else [Stroke(points=np.asarray(s, dtype=np.float64)) for s in strokes])
    candidates = []
    for p in prims:
        prof = profile_from_primitive(p)
        if prof is not None:
            candidates.append((p, prof, profile_area(prof)))

    if not candidates:
        kinds = ", ".join(sorted({p.kind.value for p in prims}))
        shot = render_recognition(prepare(Stroke(points=strokes[0])), prims[0]) if prims else None
        return None, (joined_note + f"Found {len(prims)} shape(s) ({kinds}) but none "
                      f"encloses an area. Join the ends up."), shot, None

    prim, profile, area = max(candidates, key=lambda t: t[2])
    note = (joined_note + f"Recognised **{prim.kind.value}** "
            f"({prim.confidence:.0%} confident), {area:.0f} mm2, "
            f"{_size_note(profile)}")
    if len(candidates) > 1:
        note += f" — used the largest of {len(candidates)} closed shapes"
    if prim.confidence < LOW_CONFIDENCE:
        note += "\n\n⚠ low confidence — redraw more clearly if this is not what you meant"

    raw = next((prepare(Stroke(points=np.asarray(st, dtype=np.float64)))
                for st, pr in zip(strokes, prims) if pr is prim), None)
    return profile, note, render_recognition(raw, prim), None


# --------------------------------------------------------------------------- #
# callbacks
# --------------------------------------------------------------------------- #
def _commit(pad_free, pad_half, depth, z_base, op: Op, session: Session,
            mode: str, style: str, build_as: str):
    key = _canvas_key(mode, style)
    profile, note, shot, _half = _read(pad_free, pad_half, session, mode, style,
                                       build_as)
    if profile is None:
        prev, rows, _ = _panels(session)
        return prev, rows, note, shot, gr.update(), session

    session.submit_profile(profile)
    session.choose_op(op)

    revolving = mode == HALF_MODE and build_as == REVOLVE_AS
    kwargs = ({"kind": FeatureKind.REVOLVE, "axis": centerline_axis()}
              if revolving else {})
    try:
        feat = session.commit(depth=float(depth), z_base=float(z_base), **kwargs)
    except Exception as exc:                        # noqa: BLE001 - shown to the user
        session.pending_profile = session.pending_op = None
        session.step = "draw"
        prev, rows, _ = _panels(session)
        return prev, rows, str(exc), shot, gr.update(), session

    # If the new feature makes the document unbuildable, take it back out again.
    # Leaving it in is technically consistent - the feature list IS the model -
    # but it strands the user: every later rebuild fails too, and the only way
    # out is an Undo they have no reason to think of.
    mesh, err = session.solid()
    if mesh is None:
        session.undo()
        prev, rows, _ = _panels(session)
        return (prev, rows, f"{err}\n\n*Not added - your drawing is untouched.*",
                shot, gr.update(), session)

    # Retire the ink INSTEAD of wiping the canvas. This is the whole trick: the
    # drawing stays where the user put it, and the next read sees only what is
    # new. `session.commit` has already snapshotted the old map, so Undo puts
    # these strokes back in play.
    clicks = gr.update()
    if style == CLICK:
        session.clicked[key] = []
        clicks = render_clicks([], False, mode == HALF_MODE)
    else:
        session.retire_ink(key, ink_mask(pad_half if mode == HALF_MODE else pad_free))

    how = ("revolved around the centreline" if revolving
           else f"{feat.depth:g} mm deep from z={feat.z_base:g}")
    preview, rows, msg = _panels(
        session, f"{note}\n\nAdded as **{feat.name}** — {op.value}, {how}")
    kept = ("*Draw the next shape - the one you just built stays on the canvas "
            "so you can place it.*" if style != CLICK else
            "*Click the corners of the next shape.*")
    return preview, rows, msg + "\n\n" + kept, shot, clicks, session


@preview_cb
def on_preview(pad_free, pad_half, state, mode=FREE_MODE, style=FREEHAND,
               build_as=EXTRUDE_AS):
    """Recognise without committing, so the user can check before building.

    This is also where the add-or-cut guess lives. It is only a guess: it says
    how sure it is and it changes nothing until a button is pressed.
    """
    session = _session(state)
    profile, note, shot, _half = _read(pad_free, pad_half, session, mode, style,
                                       build_as)
    if profile is not None:
        existing = [f.profile for f in session.doc.features]
        note += _confidence_line("Probably want to", suggest_op(profile, existing))
    _, rows, _ = _panels(session)
    return rows, note, shot, session


@build_cb
def on_add(pad_free, pad_half, depth, z_base, state, mode=FREE_MODE,
           style=FREEHAND, build_as=EXTRUDE_AS):
    return _commit(pad_free, pad_half, depth, z_base, Op.ADD, _session(state),
                   mode, style, build_as)


@build_cb
def on_cut(pad_free, pad_half, depth, z_base, state, mode=FREE_MODE,
           style=FREEHAND, build_as=EXTRUDE_AS):
    return _commit(pad_free, pad_half, depth, z_base, Op.CUT, _session(state),
                   mode, style, build_as)


@build_cb
def on_undo(state, mode=FREE_MODE, style=FREEHAND):
    session = _session(state)
    ok = session.undo()
    preview, rows, msg = _panels(
        session, "Undone - that shape is on the canvas again and can be rebuilt."
        if ok else "Nothing to undo.")
    key = _canvas_key(mode, style)
    clicks = (render_clicks(session.clicked.get(key) or [], False,
                            mode == HALF_MODE) if style == CLICK else gr.update())
    return preview, rows, msg, gr.update(), clicks, session


@build_cb
def on_clear(pad_free, pad_half, state, mode=FREE_MODE, style=FREEHAND):
    """Start the model over, without touching the canvas.

    Whatever is drawn is marked as used rather than erased, because erasing it
    would mean writing to the canvas. Use the bin icon on the canvas toolbar if
    you want the ink gone too.
    """
    session = Session()
    for m in (FREE_MODE, HALF_MODE):
        pad = pad_half if m == HALF_MODE else pad_free
        session.retire_ink(_canvas_key(m, FREEHAND), ink_mask(pad))
    return (None, [], "Cleared. Draw a shape to begin - anything still on the "
            "canvas is ignored, and the bin icon on the canvas clears the ink.",
            None, render_clicks([], False, mode == HALF_MODE), session)


@click_cb
def on_click_point(state, mode, style, evt: gr.SelectData):
    """Drop a corner where the user clicked and join it to the last one."""
    session = _session(state)
    key = _canvas_key(mode, style)
    pts = list(session.clicked.get(key) or [])
    x, y = (float(evt.index[0]), float(evt.index[1])) if evt.index else (0.0, 0.0)
    pts.append((x, y))
    session.clicked[key] = pts
    return (render_clicks(pts, False, mode == HALF_MODE),
            _click_note(pts, mode), session)


@click_cb
def on_undo_point(state, mode, style):
    session = _session(state)
    key = _canvas_key(mode, style)
    pts = list(session.clicked.get(key) or [])[:-1]
    session.clicked[key] = pts
    return (render_clicks(pts, False, mode == HALF_MODE),
            _click_note(pts, mode), session)


@click_cb
def on_clear_points(state, mode, style):
    session = _session(state)
    session.clicked[_canvas_key(mode, style)] = []
    return (render_clicks([], False, mode == HALF_MODE),
            "Points cleared. Click a corner to start again.", session)


def _click_note(pts, mode: str) -> str:
    if not pts:
        return "Click a corner to start."
    xs = [p[0] / PX_PER_MM for p in pts]
    ys = [(CANVAS_H - p[1]) / PX_PER_MM for p in pts]
    last = f"last corner at {xs[-1]:.0f}, {ys[-1]:.0f} mm"
    if len(pts) < 3:
        need = 3 - len(pts)
        return f"{len(pts)} corner(s), {last}. {need} more before this is a shape."
    span = f"{max(xs) - min(xs):.0f} x {max(ys) - min(ys):.0f} mm"
    tail = ("The outline closes back to the first corner when you build."
            if mode == FREE_MODE else
            "Start and finish on the centreline for a half profile.")
    return f"{len(pts)} corners, {span}, {last}. {tail}"


@view_cb
def on_view_change(mode, style):
    """Show the surface that matches the mode, and hide the other three.

    Visibility only. No image is ever sent to a Sketchpad - see the module
    docstring for why that matters.
    """
    freehand = style == FREEHAND
    return (gr.update(visible=freehand and mode == FREE_MODE),
            gr.update(visible=freehand and mode == HALF_MODE),
            gr.update(visible=not freehand),
            gr.update(visible=not freehand),
            gr.update(visible=mode == HALF_MODE))


@export_cb
def on_export(state):
    mesh, err = _session(state).solid()
    if mesh is None:
        return None, err or "Nothing to export yet - add a shape first."
    path = Path(tempfile.gettempdir()) / "sketch3d_part.stl"
    problems = export_stl(mesh, path)
    msg = "Exported." if not problems else "Exported, but:\n- " + "\n- ".join(problems)
    return str(path), msg


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
BRUSH = gr.Brush(default_size=BRUSH_PX, colors=["#000000"],
                 default_color="#000000", color_mode="fixed")
SHEET_W, SHEET_H = CANVAS_W / PX_PER_MM, CANVAS_H / PX_PER_MM

with gr.Blocks(title="Sketch3D") as demo:
    session = gr.State()          # filled by _session() on first use

    gr.Markdown(
        f"""# Sketch3D
Draw a closed shape on the left. **Add volume** makes it a solid; **Cut volume** removes it from
what is already there. Repeat, then export an STL.

<sub>Shape recogniser: {ARM}<br>Add/cut suggestion: {OP_ARM}</sub>"""
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1. Draw")
            mode = gr.Radio(
                [FREE_MODE, HALF_MODE], value=FREE_MODE, label="Drawing mode",
                info="Half mode: draw ONE half of the outline against the "
                     "centreline, the way you would in Fusion. Because the "
                     "centreline is drawn rather than guessed, there is no axis "
                     "to infer and no axis error to make.")
            style = gr.Radio(
                [FREEHAND, CLICK], value=FREEHAND, label="How to draw it",
                info="Freehand is right for curves - the fitter turns a wobbly "
                     "arc into a true one. Straight edges are easier clicked: "
                     "each click drops a corner on the grid and the edges join "
                     "up exactly, with no hand wobble to undo.")
            build_as = gr.Radio(
                [REVOLVE_AS, EXTRUDE_AS], value=REVOLVE_AS, visible=False,
                label="What to do with the half",
                info="Half an arch revolved is a ball; mirrored and extruded it "
                     "is a block. The drawing is the same either way, so this "
                     "is your call.")

            # TWO canvases, one per mode, each with its background baked in at
            # construction. They are inputs only - nothing writes to them, not
            # even a visibility flag - so no click can put them in a loading
            # state. Switching modes shows and hides the COLUMN around each one,
            # which is why these wrappers exist.
            with gr.Column(visible=True) as free_box:
                pad_free = gr.Sketchpad(
                    height=CANVAS_H, width=CANVAS_W, label=None, type="numpy",
                    canvas_size=(CANVAS_W, CANVAS_H), value=grid_background(False),
                    # A fat brush blurs corners together and the skeletoniser
                    # then rounds them off, so rectangles read as blobs.
                    brush=BRUSH)
            with gr.Column(visible=False) as half_box:
                pad_half = gr.Sketchpad(
                    height=CANVAS_H, width=CANVAS_W, label=None, type="numpy",
                    canvas_size=(CANVAS_W, CANVAS_H), value=grid_background(True),
                    brush=BRUSH)
            clicks = gr.Image(value=render_clicks([], False, False), label=None,
                              height=CANVAS_H, interactive=False, visible=False)
            with gr.Row(visible=False) as click_tools:
                undo_pt_btn = gr.Button("Undo last corner")
                clear_pt_btn = gr.Button("Clear corners")

            gr.Markdown(
                f"<sub>The sheet is {SHEET_W:.0f} x {SHEET_H:.0f} mm and the grid "
                f"is 5 mm, so you can draw a 40 mm circle on purpose rather than "
                f"finding out afterwards. A shape can take several strokes - they "
                f"are joined automatically. <b>Your drawing is never cleared:</b> "
                f"once a shape has been built the app stops looking at it, so just "
                f"draw the next one beside it.</sub>"
            )

            recognised = gr.Image(label="What I recognised  (grey = your stroke, blue = the fitted shape)",
                                  height=220, interactive=False)

            gr.Markdown("### 2. Depth")
            with gr.Row():
                depth = gr.Number(value=DEFAULT_DEPTH_MM, label="Depth (mm)", precision=2)
                z_base = gr.Number(value=0.0, label="Start at z (mm)", precision=2)
            gr.Markdown("<sub>Depth is ignored by a revolve - its height comes "
                        "from how tall you drew the half, and where on the "
                        "centreline you drew it is where it lands.</sub>")

            gr.Markdown("### 3. Build")
            with gr.Row():
                preview_btn = gr.Button("What did I draw?")
                add_btn = gr.Button("Add volume", variant="primary")
                cut_btn = gr.Button("Cut volume", variant="stop")
            with gr.Row():
                undo_btn = gr.Button("Undo")
                clear_btn = gr.Button("Start over")

        with gr.Column(scale=5):
            gr.Markdown("### 3D Preview")
            preview = gr.Model3D(height=330, label=None)
            status = gr.Markdown("Draw a shape to begin.")

            gr.Markdown("### Features / Layers")
            layers = gr.Dataframe(headers=HEADERS, label=None, interactive=False)
            with gr.Row():
                export_btn = gr.Button("Export STL", variant="primary")
                stl_file = gr.File(label="STL")

    with gr.Accordion("How it works", open=False):
        gr.Markdown(
            """
Your pen marks are thresholded, thinned to one-pixel centrelines and traced back into ordered
paths. Strokes that end near each other are joined into one contour, so a four-stroke rectangle
is still a rectangle. Each contour is classified — line, arc, circle, rectangle, or an
unrecognised polyline — and the matching primitive is fitted by least squares, so a wobbly
circle becomes an exact one. Clicked corners skip all of that: they are already exact.

The closed outline is then either extruded to your depth or revolved around the centreline you
drew, and unioned or subtracted from the running solid with a mesh boolean. Export runs a
watertightness and printability check first.

Add-or-cut and revolve-or-extrude are *suggested* with a confidence and never decided for you.
They are geometric rules today; the same suggestion object is where the learned model goes.

The feature list on the right *is* the model: it is replayed from scratch on every edit, which
is why undo is instant and the preview can never drift out of sync with the list.
            """
        )

    # The canvases are inputs to everything and outputs of nothing.
    pads = [pad_free, pad_half]
    outs = [preview, layers, status, recognised, clicks, session]

    preview_btn.click(on_preview, pads + [session, mode, style, build_as],
                      [layers, status, recognised, session])
    add_btn.click(on_add, pads + [depth, z_base, session, mode, style, build_as], outs)
    cut_btn.click(on_cut, pads + [depth, z_base, session, mode, style, build_as], outs)
    undo_btn.click(on_undo, [session, mode, style], outs)
    clear_btn.click(on_clear, pads + [session, mode, style], outs)

    views = [free_box, half_box, clicks, click_tools, build_as]
    mode.change(on_view_change, [mode, style], views)
    style.change(on_view_change, [mode, style], views)

    clicks.select(on_click_point, [session, mode, style], [clicks, status, session])
    undo_pt_btn.click(on_undo_point, [session, mode, style], [clicks, status, session])
    clear_pt_btn.click(on_clear_points, [session, mode, style], [clicks, status, session])

    export_btn.click(on_export, [session], [stl_file, status])


if __name__ == "__main__":
    # `theme` belongs to launch() in Gradio 6; older versions take it on Blocks().
    try:
        demo.launch(theme=gr.themes.Soft())
    except TypeError:
        demo.launch()
