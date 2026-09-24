"""Sketch3D - draw a shape, add or cut it, export an STL.  Run:  python app.py

OWNER: shared. Keep it THIN - it wires widgets to functions that live
elsewhere. Every line of logic in here is a line you will both edit.

STATE lives in a gr.State, so each browser tab gets its own document. Never use
a module-level global for it - once this is on Spaces it would be shared across
every visitor at once.

The recogniser is chosen at startup: if a trained model exists at
RECOGNIZER_PATH it is used, otherwise the rules baseline. Train with
`scripts/train_recognizer.py` and restart - nothing else changes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from sketch2stl.config import CANVAS_H, CANVAS_W, DEFAULT_DEPTH_MM, LOW_CONFIDENCE
from sketch2stl.exporter import check, export_stl
from sketch2stl.kernel import stats
from sketch2stl.profiles import profile_area, profile_from_primitive
from sketch2stl.recognizer import RuleRecognizer
from sketch2stl.recognizer.ml import MLRecognizer
from sketch2stl.session import Session
from sketch2stl.types import Op
from ui.canvas import strokes_from_image
from ui.layers import HEADERS, to_rows
from ui.preview import mesh_to_preview_file

RECOGNIZER_PATH = os.environ.get("RECOGNIZER_PATH", "models/recognizer")

if (Path(RECOGNIZER_PATH) / "model.joblib").exists():
    RECOGNIZER = MLRecognizer(RECOGNIZER_PATH)
    ARM = f"learned model ({RECOGNIZER_PATH})"
else:
    RECOGNIZER = RuleRecognizer()
    ARM = "rules baseline (no trained model found - see scripts/train_recognizer.py)"

print(f"recogniser: {ARM}")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _panels(session: Session, message: str = ""):
    """Rebuild the solid and return everything the right-hand side shows."""
    mesh, err = session.solid()
    rows = to_rows(session.doc)
    if mesh is None:
        return None, rows, message or err or "Draw a shape, then Add or Cut it."

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
    return mesh_to_preview_file(mesh), rows, "\n".join(x for x in lines if x)


def _read_canvas(sketch):
    """Canvas -> (profile, description) or (None, why not)."""
    strokes = strokes_from_image(sketch)
    if not strokes:
        return None, "The canvas is empty - draw a closed shape first."

    prims = RECOGNIZER.recognize_batch(strokes)
    candidates = []
    for p in prims:
        prof = profile_from_primitive(p)
        if prof is not None:
            candidates.append((p, prof, profile_area(prof)))

    if not candidates:
        kinds = ", ".join(sorted({p.kind.value for p in prims}))
        return None, (f"Found {len(prims)} stroke(s) ({kinds}) but none encloses an area. "
                      f"Draw a closed outline - join the ends up.")

    # largest closed region wins; a stray tick mark should not become the part
    prim, profile, area = max(candidates, key=lambda t: t[2])
    note = f"Recognised **{prim.kind.value}** ({prim.confidence:.0%} confident), {area:.0f} mm2"
    if len(candidates) > 1:
        note += f" — used the largest of {len(candidates)} closed shapes"
    if prim.confidence < LOW_CONFIDENCE:
        note += "\n⚠ low confidence — redraw more clearly if this is not what you meant"
    return profile, note


def _commit(sketch, depth, z_base, op: Op, session: Session):
    profile, note = _read_canvas(sketch)
    if profile is None:
        _, rows, _ = _panels(session)
        return None, rows, note, session

    session.submit_profile(profile)
    session.choose_op(op)
    try:
        feat = session.commit(depth=float(depth), z_base=float(z_base))
    except Exception as exc:                        # noqa: BLE001 - shown to the user
        _, rows, _ = _panels(session)
        return None, rows, str(exc), session

    preview, rows, msg = _panels(
        session, f"{note}\nAdded as **{feat.name}** — {op.value} {feat.depth:g} mm "
                 f"from z={feat.z_base:g}")
    return preview, rows, msg, session


# --------------------------------------------------------------------------- #
# callbacks
# --------------------------------------------------------------------------- #
def on_preview(sketch, session: Session):
    """Recognise without committing, so the user can check before building."""
    _, note = _read_canvas(sketch)
    _, rows, _ = _panels(session)
    return rows, note, session


def on_add(sketch, depth, z_base, session):
    return _commit(sketch, depth, z_base, Op.ADD, session)


def on_cut(sketch, depth, z_base, session):
    return _commit(sketch, depth, z_base, Op.CUT, session)


def on_undo(session: Session):
    ok = session.undo()
    preview, rows, msg = _panels(session, "Undone." if ok else "Nothing to undo.")
    return preview, rows, msg, session


def on_clear(_session):
    session = Session()
    return None, [], "Cleared. Draw a shape to begin.", session


def on_export(session: Session):
    mesh, err = session.solid()
    if mesh is None:
        return None, err or "Nothing to export yet - add a shape first."
    path = Path(tempfile.gettempdir()) / "sketch3d_part.stl"
    problems = export_stl(mesh, path)
    msg = "Exported." if not problems else "Exported, but:\n- " + "\n- ".join(problems)
    return str(path), msg


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
with gr.Blocks(title="Sketch3D", theme=gr.themes.Soft()) as demo:
    session = gr.State(Session)

    gr.Markdown(
        f"""# Sketch3D
Draw a closed shape on the left. **Add volume** makes it a solid; **Cut volume** removes it from
what is already there. Repeat, then export an STL.

<sub>Recogniser: {ARM}</sub>"""
    )

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 1. Draw")
            sketch = gr.Sketchpad(
                height=CANVAS_H, width=CANVAS_W, label=None, type="numpy",
                canvas_size=(CANVAS_W, CANVAS_H),
            )
            gr.Markdown(
                f"<sub>The canvas is {CANVAS_W / 4:.0f} x {CANVAS_H / 4:.0f} mm. "
                f"Draw one closed outline at a time.</sub>"
            )

            gr.Markdown("### 2. Depth")
            with gr.Row():
                depth = gr.Number(value=DEFAULT_DEPTH_MM, label="Depth (mm)", precision=2)
                z_base = gr.Number(value=0.0, label="Start at z (mm)", precision=2)

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
            layers = gr.Dataframe(headers=HEADERS, label=None, interactive=False,
                                  row_count=(0, "dynamic"))
            with gr.Row():
                export_btn = gr.Button("Export STL", variant="primary")
                stl_file = gr.File(label="STL")

    with gr.Accordion("How it works", open=False):
        gr.Markdown(
            """
Your pen marks are thresholded, thinned to one-pixel centrelines and traced back into ordered
paths. Each path is classified — line, arc, circle, rectangle, or an unrecognised polyline —
and then the matching primitive is fitted by least squares, so a wobbly circle becomes an exact
one. The closed outline is cleaned up with Shapely, extruded to your depth, and unioned or
subtracted from the running solid with a mesh boolean. Export runs a watertightness and
printability check first.

The feature list on the right *is* the model: it is replayed from scratch on every edit, which
is why undo is instant and the preview can never drift out of sync with the list.
            """
        )

    outs = [preview, layers, status, session]
    preview_btn.click(on_preview, [sketch, session], [layers, status, session])
    add_btn.click(on_add, [sketch, depth, z_base, session], outs)
    cut_btn.click(on_cut, [sketch, depth, z_base, session], outs)
    undo_btn.click(on_undo, [session], outs)
    clear_btn.click(on_clear, [session], outs)
    export_btn.click(on_export, [session], [stl_file, status])


if __name__ == "__main__":
    demo.launch()
