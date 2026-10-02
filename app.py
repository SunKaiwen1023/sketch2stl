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
from sketch2stl.hub import fetch_models
from sketch2stl.profiles import (close_to_axis, fold_to_one_side, mirror_half,
                                 nest_profiles, profile_area, profile_from_points,
                                 profile_from_primitive)
from sketch2stl.recognizer import RuleRecognizer
from sketch2stl.recognizer.hybrid import HybridRecognizer
from sketch2stl.recognizer.ml import MLRecognizer
from sketch2stl.session import Session
from sketch2stl.stitch import smooth_seams, stitch, stitch_all
from sketch2stl.suggest import addcut_arm, kind_arm, suggest_kind, suggest_op
from sketch2stl.symmetry import find_symmetry, symmetrize
from sketch2stl.types import FeatureKind, Op, PrimitiveKind, Stroke
from ui.canvas import (centerline_axis, grid_background, ink_mask, render_clicks,
                       render_half, render_recognition, render_sketch,
                       strokes_from_mask)
from ui.layers import HEADERS, to_rows
from ui.preview import mesh_to_preview_file

FREE_MODE = "Free draw"
HALF_MODE = "Half + centreline"

FREEHAND = "Sketch freehand"
CLICK = "Click the corners"

EXTRUDE_AS = "Extrude (mirror the half into a block)"
REVOLVE_AS = "Revolve (spin the half around the centreline)"
AUTO_AS = "Let ML2 decide (you can override)"

RECOGNIZER_PATH = os.environ.get("RECOGNIZER_PATH", "models/recognizer")

# Weights are never committed, so a fresh clone fetches them from the Hub once.
for _line in fetch_models(RECOGNIZER_PATH):
    print(f"models:          {_line}")

if (Path(RECOGNIZER_PATH) / "model.joblib").exists():
    # ML1 answers; when it falls back to "polyline" the circle / rectangle fits
    # get a say (see sketch2stl/recognizer/hybrid.py for why and the numbers).
    RECOGNIZER = HybridRecognizer(MLRecognizer(RECOGNIZER_PATH), RuleRecognizer())
    ARM = f"learned model + geometric check ({RECOGNIZER_PATH})"
else:
    RECOGNIZER = RuleRecognizer()
    ARM = "rules baseline (no trained model found - see scripts/train_recognizer.py)"

OP_ARM = addcut_arm()
KIND_ARM = kind_arm()

print(f"recogniser:      {ARM}")
print(f"add/cut guess:   {OP_ARM}")
print(f"revolve/mirror:  {KIND_ARM}")


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
          build_as: str, make_sym: bool = False):
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

        # ML2 DECIDES unless the person already has. Showing its guess as a line
        # of text and then building whatever the radio said made the model
        # invisible: nothing it said ever changed the part.
        sug = suggest_kind(half, axis)
        if build_as == AUTO_AS:
            revolving = sug.value == "revolve"
            who = (f"**ML2 chose: {'Revolve' if revolving else 'Mirror + extrude'}** "
                   f"({sug.confidence:.0%})" + (" - not sure, check the 3D preview or pick "
                   "one above" if sug.unsure else " - pick the other option above to override"))
        else:
            revolving = build_as == REVOLVE_AS
            who = (f"You chose {'Revolve' if revolving else 'Mirror + extrude'}; ML2 would say "
                   f"{'Revolve' if sug.value == 'revolve' else 'Mirror + extrude'} ({sug.confidence:.0%})")
        ring = close_to_axis(half, axis) if revolving else mirror_half(half, axis)
        what = "Revolve" if revolving else "Mirrored half"

        profile = profile_from_points(ring)
        if profile is None:
            shot = render_half(raw, segments, None, revolving)
            return None, (joined_note + "That half did not close into a region. "
                          "Start and finish near the centreline."), shot, {"revolving": revolving}

        kinds = [s.kind.value for s in segments]
        shape_note = (f", fitted as {len(segments)} piece(s): {', '.join(kinds)}"
                      if kinds else "")
        note = (joined_note + f"{what} profile, {profile_area(profile):.0f} mm2, "
                f"{_size_note(profile)}{shape_note}")
        note += f"\n\n{who} · {sug.reason}"
        return [profile], note, render_half(raw, segments, half, revolving), {"revolving": revolving}

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
        profiles, sym_note = _symmetry([profile], make_sym)
        return (profiles, note + sym_note,
                render_sketch(session.doc.features, profiles, [px_to_mm(stitched.points)]), {})

    # SEVERAL SHAPES AT ONCE. Each group of strokes that closes on itself is
    # its own outline; outlines inside outlines become holes. A plate drawn with
    # two holes in it used to come back as just one of the three.
    groups = stitch_all([np.asarray(s, dtype=np.float64) for s in strokes])
    found, raws = [], []
    for g in groups:
        pts = smooth_seams(g) if g.n_strokes > 1 else g.points
        prim = RECOGNIZER.recognize(Stroke(points=np.asarray(pts, dtype=np.float64)))
        raws.append(prepare(Stroke(points=np.asarray(pts, dtype=np.float64))))
        prof = profile_from_primitive(prim)
        if prof is not None:
            found.append((prim, prof))

    if not found:
        kinds = ", ".join(sorted({p.kind.value for p in RECOGNIZER.recognize_batch(
            [Stroke(points=np.asarray(s, dtype=np.float64)) for s in strokes])}))
        return None, (f"Found {len(groups)} shape(s) ({kinds}) but none encloses an "
                      f"area. Join the ends up."), render_sketch(session.doc.features, (), raws), {}

    profiles = nest_profiles([p for _, p in found])
    n_holes = sum(len(p.holes) for p in profiles)
    names = ", ".join(f"**{pr.kind.value}** ({pr.confidence:.0%})" for pr, _ in found)
    note = f"Recognised {names}"
    if len(found) > 1:
        note += (f" → {len(profiles)} solid(s)" + (f" with {n_holes} hole(s)" if n_holes else ""))
    note += "  ·  " + "; ".join(f"{profile_area(p):.0f} mm2, {_size_note(p)}" for p in profiles)
    if any(pr.confidence < LOW_CONFIDENCE for pr, _ in found):
        note += "\n\n⚠ low confidence on at least one shape — redraw it more clearly if it is wrong"
    profiles, sym_note = _symmetry(profiles, make_sym)
    return profiles, note + sym_note, render_sketch(session.doc.features, profiles, raws), {}


def _tidy(profile):
    """A light clean-up for a hand-drawn outline the kernel refused: fix
    self-touching rings, drop slivers, smooth sub-0.3 mm wiggles."""
    from shapely.geometry import Polygon
    from sketch2stl.profiles import polygon_to_profile
    try:
        g = Polygon(profile.outer, [h for h in profile.holes]).buffer(0)
        g = g.buffer(0.3, join_style=2).buffer(-0.3, join_style=2).simplify(0.2)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        return polygon_to_profile(g) if g.area > 1.0 else None
    except Exception:                                # noqa: BLE001
        return None


def _symmetry(profiles, make_sym: bool):
    """Report how symmetric each shape is, and make it exact if asked."""
    out, notes = [], []
    for p in profiles:
        try:
            s = find_symmetry(p)
        except Exception:                            # noqa: BLE001
            out.append(p); continue
        if s.score > 0.999:
            out.append(p); continue                  # already exact (a snapped circle / rect)
        if make_sym and s.is_symmetric:
            try:
                out.append(symmetrize(p, s)); notes.append(f"made symmetric ({s.score:.0%} → 100%)")
                continue
            except Exception:                        # noqa: BLE001
                pass
        out.append(p)
        if s.is_symmetric:
            notes.append(f"looks {s.score:.0%} symmetric — tick **Make symmetric** to straighten it")
        elif make_sym:
            notes.append(f"only {s.score:.0%} symmetric, left as drawn")
    return out, ("\n\n◇ " + "; ".join(notes)) if notes else ""


# --------------------------------------------------------------------------- #
# callbacks
# --------------------------------------------------------------------------- #
def _commit(pad_free, pad_half, depth, z_base, op: Op, session: Session,
            mode: str, style: str, build_as: str, make_sym: bool = False):
    key = _canvas_key(mode, style)
    profiles, note, shot, extra = _read(pad_free, pad_half, session, mode, style,
                                        build_as, make_sym)
    if not profiles:
        prev, rows, _ = _panels(session)
        return prev, rows, note, shot, gr.update(), session

    revolving = mode == HALF_MODE and (extra or {}).get("revolving", build_as == REVOLVE_AS)
    kwargs = ({"kind": FeatureKind.REVOLVE, "axis": centerline_axis()}
              if revolving else {})
    try:
        made = session.commit_many(profiles, op, depth=float(depth),
                                   z_base=float(z_base), **kwargs)
    except Exception as exc:                        # noqa: BLE001 - shown to the user
        session.pending_profile = session.pending_op = None
        session.step = "draw"
        prev, rows, _ = _panels(session)
        return prev, rows, str(exc), shot, gr.update(), session

    # If the new features make the document unbuildable, take them back out -
    # then try again shape by shape, with a light clean-up, so ONE awkward hand-
    # drawn hole does not block the others (or the whole click).
    mesh, err = session.solid()
    skipped = 0
    if mesh is None:
        session.undo()
        made = []
        for prof in profiles:
            ok = False
            for cand in (prof, _tidy(prof)):
                if cand is None:
                    continue
                try:
                    trial = session.commit_many([cand], op, depth=float(depth),
                                                z_base=float(z_base), **kwargs)
                except Exception:                     # noqa: BLE001
                    continue
                if session.solid()[0] is not None:
                    made += trial; ok = True
                    break
                session.undo()
            skipped += not ok
        if not made:
            prev, rows, _ = _panels(session)
            return (prev, rows, f"{err}\n\n*Not added - your drawing is untouched.*",
                    shot, gr.update(), session)

    # Retire the ink INSTEAD of wiping the canvas (see the module docstring).
    clicks = gr.update()
    if style == CLICK:
        session.clicked[key] = []
        clicks = render_clicks([], False, mode == HALF_MODE)
    else:
        session.retire_ink(key, ink_mask(pad_half if mode == HALF_MODE else pad_free))

    how = ("revolved around the centreline" if revolving
           else f"{float(depth):g} mm deep from z={float(z_base):g}")
    names = ", ".join(f"**{f.name}**" for f in made)
    extra_note = (f"\n\n⚠ {skipped} shape(s) could not be built and were left out - "
                  f"redraw them a little more cleanly." if skipped else "")
    preview, rows, msg = _panels(session, f"{note}\n\nAdded {names} — {op.value}, {how}{extra_note}")
    if mode != HALF_MODE:
        shot = render_sketch(session.doc.features)   # the clean sketch, now including it
    kept = ("*Draw the next shape - the one you just built stays on the canvas "
            "so you can place it.*" if style != CLICK else
            "*Click the corners of the next shape.*")
    return preview, rows, msg + "\n\n" + kept, shot, clicks, session


@preview_cb
def on_preview(pad_free, pad_half, state, mode=FREE_MODE, style=FREEHAND,
               build_as=EXTRUDE_AS, make_sym=False):
    """Recognise without committing, so the user can check before building.

    This is also where the add-or-cut guess lives. It is only a guess: it says
    how sure it is and it changes nothing until a button is pressed.
    """
    session = _session(state)
    profiles, note, shot, _extra = _read(pad_free, pad_half, session, mode, style,
                                         build_as, make_sym)
    if profiles:
        existing = [f.profile for f in session.doc.features]
        note += _confidence_line("Probably want to", suggest_op(profiles[0], existing))
    _, rows, _ = _panels(session)
    return rows, note, shot, session


@build_cb
def on_add(pad_free, pad_half, depth, z_base, state, mode=FREE_MODE,
           style=FREEHAND, build_as=EXTRUDE_AS, make_sym=False):
    return _commit(pad_free, pad_half, depth, z_base, Op.ADD, _session(state),
                   mode, style, build_as, make_sym)


@build_cb
def on_cut(pad_free, pad_half, depth, z_base, state, mode=FREE_MODE,
           style=FREEHAND, build_as=EXTRUDE_AS, make_sym=False):
    return _commit(pad_free, pad_half, depth, z_base, Op.CUT, _session(state),
                   mode, style, build_as, make_sym)


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
# the layer list (read-only: what has been built, newest on top)
# --------------------------------------------------------------------------- #
BRUSH = gr.Brush(default_size=BRUSH_PX, colors=["#000000"],
                 default_color="#000000", color_mode="fixed")
SHEET_W, SHEET_H = CANVAS_W / PX_PER_MM, CANVAS_H / PX_PER_MM


def _layer_html(state) -> str:
    """Read-only layer list: newest on top, an icon for how it was made, add or cut."""
    session = _session(state)
    if not session.doc.features:
        return "<div class='s3-sub'>Nothing built yet - draw a shape, then Add volume.</div>"
    icon = {"extrude": "▭", "revolve": "◐", "mirror_extrude": "◫"}
    rows = []
    for i, f in reversed(list(enumerate(session.doc.features))):
        cut = f.op is Op.CUT
        size = "revolved" if f.kind is FeatureKind.REVOLVE else f"{f.depth:g} mm"
        rows.append(
            f"<div class='s3-row'><span class='s3-ico'>{icon.get(f.kind.value, '▭')}</span>"
            f"<span class='s3-name'>{f.name}</span>"
            f"<span class='s3-tag {'cut' if cut else 'add'}'>{'cut' if cut else 'add'}</span>"
            f"<span class='s3-dim'>{size}</span></div>")
    return "".join(rows)


# Figma-like chrome: a thin top bar, LAYERS on the left, the canvas in the
# middle, a DESIGN panel on the right. Pure styling - every component and every
# event is the same as before, so none of this can bring back the loading hang.
HEAD = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
        'family=Inter:wght@400;500;600&display=swap">'
        # Light theme only. In a dark-mode browser Gradio switches to white text,
        # which vanished on these light panels - the Depth box looked empty.
        "<script>(function(){var u=new URL(window.location.href);"
        "if(u.searchParams.get('__theme')!=='light'){u.searchParams.set('__theme','light');"
        "window.location.replace(u.toString());}})();</script>")

CSS = """
:root, .gradio-container, .dark, .dark .gradio-container { color-scheme: light; --s3-bg:#f5f5f5; --s3-panel:#ffffff; --s3-line:#e6e6e6;
  --s3-ink:#1e1e1e; --s3-mute:#4d4d4d; --s3-blue:#0d99ff; --s3-sel:#e5f4ff; }
.gradio-container { background: var(--s3-bg) !important; max-width: 100% !important;
  padding: 0 !important; font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
  font-size: 12px !important; color: var(--s3-ink); }
#s3-topbar { background:#2c2c2c; color:#fff; padding:8px 14px; margin:0 !important;
  border-radius:0 !important; align-items:center; gap:14px; }
#s3-topbar *, #s3-topbar label span, .gradio-container #s3-topbar label span { color:#fff !important; }
#s3-topbar .s3-title { font-weight:600; font-size:13px; letter-spacing:.01em; }
#s3-topbar .s3-badge { font-size:11px; opacity:.9; }
#s3-topbar label, #s3-topbar .wrap { background:transparent !important; border:none !important; }
#s3-topbar input[type=radio] + span, #s3-topbar label span { font-size:12px; }
#s3-topbar fieldset { padding:0 !important; }
#s3-topbar .block { background:transparent !important; border:none !important; box-shadow:none !important; }
.s3-panel { background: var(--s3-panel) !important; border-right:1px solid var(--s3-line);
  border-left:1px solid var(--s3-line); min-height: calc(100vh - 52px); padding:10px 12px !important; }
.s3-panel .block, .s3-panel .form { border:none !important; box-shadow:none !important;
  background:transparent !important; }
.s3-h { font-weight:600; font-size:11px; color:var(--s3-ink); padding:6px 0 4px;
  border-bottom:1px solid var(--s3-line); margin-bottom:6px; letter-spacing:.02em; }
.s3-h span { color:var(--s3-mute); font-weight:400; }
.s3-panel label > span, .s3-panel .label-wrap span { font-size:11px !important; color:#333 !important;
  font-weight:500 !important; }
.gradio-container .info, .gradio-container [data-testid=block-info] { color:#4d4d4d !important; }
.gradio-container label span, .gradio-container .wrap label { color:#1e1e1e !important; opacity:1 !important; }
.s3-panel input[type=text], .s3-panel input[type=number], .s3-panel textarea {
  color:#1e1e1e !important; font-size:12px !important; border-radius:4px !important;
  border:1px solid transparent !important; background:#f5f5f5 !important; }
.s3-panel input[type=text]:focus, .s3-panel input[type=number]:focus {
  border-color: var(--s3-blue) !important; background:#fff !important; }
.s3-panel input[type=checkbox]:checked, .s3-panel input[type=radio]:checked {
  background-color: var(--s3-blue) !important; border-color: var(--s3-blue) !important; }
.s3-panel button { border-radius:6px !important; font-size:12px !important; font-weight:500 !important;
  box-shadow:none !important; }
.s3-panel button.primary { background: var(--s3-blue) !important; border-color: var(--s3-blue) !important; color:#fff !important; }
.s3-panel button.secondary { background:#fff !important; border:1px solid var(--s3-line) !important; color:var(--s3-ink) !important; }
.s3-panel button.stop { background:#fff !important; border:1px solid #f24822 !important; color:#f24822 !important; }
#layer-list .s3-row { display:flex; align-items:center; gap:8px; padding:7px 6px;
  border-bottom:1px solid #f0f0f0; font-size:12px; color:var(--s3-ink); }
#layer-list .s3-ico { width:16px; text-align:center; color:#333; }
#layer-list .s3-name { flex:1; font-weight:500; }
#layer-list .s3-tag { font-size:10px; padding:1px 6px; border-radius:8px; font-weight:600; }
#layer-list .s3-tag.add { background:#e5f4ff; color:#0b6fc2; }
#layer-list .s3-tag.cut { background:#ffe9e3; color:#c4320a; }
#layer-list .s3-dim { color:#4d4d4d; font-size:11px; }
#layer-props { background:#fff !important; border:none !important; }
#layer-props > div, #layer-props .block { background:#fff !important; }
.s3-status, .s3-status p { font-size:12px !important; line-height:1.45 !important; }
.s3-status strong { font-weight:600; }
#s3-canvas { background: var(--s3-bg) !important; padding:12px !important; }
#s3-canvas .block { border-radius:6px !important; border:1px solid var(--s3-line) !important; }
.s3-status { font-size:12px; color:var(--s3-ink); }
.s3-sub { color:#4d4d4d; font-size:11px; }
footer { display:none !important; }
"""

with gr.Blocks(title="Sketch3D") as demo:
    session = gr.State()          # filled by _session() on first use

    # ---------------------------------------------------------------- top bar
    with gr.Row(elem_id="s3-topbar", equal_height=True):
        gr.HTML(f"<span class='s3-title'>◆ Sketch3D</span>&nbsp;&nbsp;"
                f"<span class='s3-badge'>ML1 · {ARM.split(' (')[0]}  |  ML2 · "
                f"{'ResNet-18' if 'ResNet' in KIND_ARM else 'rules'}</span>")
        mode = gr.Radio([FREE_MODE, HALF_MODE], value=FREE_MODE, show_label=False,
                        container=False, scale=2)
        style = gr.Radio([FREEHAND, CLICK], value=FREEHAND, show_label=False,
                         container=False, scale=2)

    with gr.Row(equal_height=False):
        # ------------------------------------------------------------ LAYERS
        with gr.Column(scale=2, min_width=220, elem_classes="s3-panel"):
            gr.HTML("<div class='s3-h'>Layers <span>· what you have built, in order</span></div>")
            layer_list = gr.HTML(_layer_html(Session()), elem_id="layer-list")
            with gr.Row():
                undo_btn = gr.Button("Undo", size="sm")
                clear_btn = gr.Button("Start over", size="sm")
            gr.HTML("<div class='s3-h' style='margin-top:14px'>3D preview</div>")
            preview = gr.Model3D(height=260, label=None, show_label=False)

        # ------------------------------------------------------------ CANVAS
        with gr.Column(scale=6, min_width=520, elem_id="s3-canvas"):
            # TWO canvases, one per mode, each with its background baked in at
            # construction. They are inputs only - nothing writes to them, not
            # even a visibility flag - so no click can put them in a loading
            # state. Switching modes shows and hides the COLUMN around each one.
            with gr.Column(visible=True) as free_box:
                pad_free = gr.Sketchpad(
                    height=CANVAS_H, width=CANVAS_W, label=None, type="numpy",
                    canvas_size=(CANVAS_W, CANVAS_H), value=grid_background(False),
                    brush=BRUSH, layers=False)
            with gr.Column(visible=False) as half_box:
                pad_half = gr.Sketchpad(
                    height=CANVAS_H, width=CANVAS_W, label=None, type="numpy",
                    canvas_size=(CANVAS_W, CANVAS_H), value=grid_background(True),
                    brush=BRUSH, layers=False)
            clicks = gr.Image(value=render_clicks([], False, False), label=None,
                              height=CANVAS_H, interactive=False, visible=False)
            with gr.Row(visible=False) as click_tools:
                undo_pt_btn = gr.Button("Undo last corner", size="sm")
                clear_pt_btn = gr.Button("Clear corners", size="sm")
            gr.Markdown(
                f"<span class='s3-sub'>Sheet {SHEET_W:.0f} × {SHEET_H:.0f} mm, 5 mm grid. "
                f"Draw several shapes at once - a shape inside another becomes a hole. "
                f"Your ink is never cleared; built shapes are simply not read again.</span>")
            recognised = gr.Image(label="Clean sketch  (grey = already built, blue = what it reads as)",
                                  height=300, interactive=False)

        # ------------------------------------------------------------ DESIGN
        with gr.Column(scale=3, min_width=280, elem_classes="s3-panel"):
            gr.HTML("<div class='s3-h'>Build</div>")
            build_as = gr.Radio(
                [AUTO_AS, REVOLVE_AS, EXTRUDE_AS], value=AUTO_AS, visible=False,
                label="Half becomes",
                info="ML2 decides by default and says how sure it is - pick one to override.")
            make_sym = gr.Checkbox(False, label="Make symmetric")
            with gr.Row():
                depth = gr.Number(value=DEFAULT_DEPTH_MM, label="Depth (mm)", precision=2, min_width=90)
                z_base = gr.Number(value=0.0, label="Start z (mm)", precision=2, min_width=90)
            preview_btn = gr.Button("What did I draw?", size="sm")
            with gr.Row():
                add_btn = gr.Button("Add volume", variant="primary", size="sm")
                cut_btn = gr.Button("Cut volume", variant="stop", size="sm")
            status = gr.Markdown("Draw a shape to begin.", elem_classes="s3-status")

            gr.HTML("<div class='s3-h' style='margin-top:14px'>Export</div>")
            export_btn = gr.Button("Export STL", variant="primary", size="sm")
            stl_file = gr.File(label="STL", show_label=False)

            # Kept (hidden) because every build callback returns the table rows.
            layers = gr.Dataframe(headers=HEADERS, label=None, interactive=False,
                                  show_label=False, visible=False)
            with gr.Accordion("How it works", open=False):
                gr.Markdown(
                    """
Pen marks are thinned to centrelines and traced into paths. Strokes whose ends meet are joined into
one outline; several outlines are read at once, and one inside another becomes a hole. **ML1**
(trained from scratch) names each outline - line, arc, circle, rectangle, polyline - and the exact
shape is fitted. In half mode **ML2** (fine-tuned ResNet-18) decides revolve or mirror-extrude and
says how sure it is; you can always override it. The layer list *is* the model: it is replayed on
every edit, so editing an old layer, hiding it or reordering it just rebuilds.
""")

    # The canvases are inputs to everything and outputs of nothing.
    pads = [pad_free, pad_half]
    outs = [preview, layers, status, recognised, clicks, session]

    preview_btn.click(on_preview, pads + [session, mode, style, build_as, make_sym],
                      [layers, status, recognised, session])
    relist = dict(fn=_layer_html, inputs=[session], outputs=[layer_list])
    add_btn.click(on_add, pads + [depth, z_base, session, mode, style, build_as, make_sym],
                  outs).then(**relist)
    cut_btn.click(on_cut, pads + [depth, z_base, session, mode, style, build_as, make_sym],
                  outs).then(**relist)
    undo_btn.click(on_undo, [session, mode, style], outs).then(**relist)
    clear_btn.click(on_clear, pads + [session, mode, style], outs).then(**relist)

    views = [free_box, half_box, clicks, click_tools, build_as]
    mode.change(on_view_change, [mode, style], views)
    style.change(on_view_change, [mode, style], views)

    clicks.select(on_click_point, [session, mode, style], [clicks, status, session])
    undo_pt_btn.click(on_undo_point, [session, mode, style], [clicks, status, session])
    clear_pt_btn.click(on_clear_points, [session, mode, style], [clicks, status, session])

    export_btn.click(on_export, [session], [stl_file, status])


def launch(**kwargs):
    """Launch with the Figma-like styling. In Gradio 6 `css`/`theme` go to launch()."""
    try:
        return demo.launch(css=CSS, theme=gr.themes.Base(), head=HEAD, **kwargs)
    except TypeError:                                 # older Gradio: no css/theme here
        return demo.launch(**kwargs)


if __name__ == "__main__":
    launch()
