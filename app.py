"""Sketch3D - Gradio entry point.  Run with:  python app.py

OWNER: shared. Both of you will touch this, so keep it THIN - it should only
wire widgets to functions that live elsewhere. Every line of logic in here is a
line you will both edit and a merge conflict waiting to happen.

The layout mirrors the mockup: sketch on the left, 3D preview top right, feature
stack bottom right, export in the corner.

STATE: `Session` lives in a gr.State, so each browser tab gets its own document.
Do not use a module-level global - it would be shared across every visitor once
this is on Spaces.
"""
from __future__ import annotations

import gradio as gr
import numpy as np

from sketch2stl.config import CANVAS_H, CANVAS_W, DEFAULT_DEPTH_MM
from sketch2stl.exporter import check, export_stl
from sketch2stl.kernel import stats
from sketch2stl.profiles import profile_from_primitive
from sketch2stl.recognizer import RuleRecognizer
from sketch2stl.session import Session
from sketch2stl.types import Op, Stroke
from ui.layers import HEADERS, to_rows
from ui.preview import mesh_to_preview_file

RECOGNIZER = RuleRecognizer()      # swap for MLRecognizer(path) once trained


def _refresh(session: Session):
    """Rebuild the solid and return everything the right-hand panels need."""
    mesh, err = session.solid()
    rows = to_rows(session.doc)
    if mesh is None:
        return None, rows, err or "Draw a shape to begin."
    s = stats(mesh)
    problems = check(mesh)
    msg = (f"{s['n_faces']} faces · "
           f"{s['size_mm'][0]:.1f} x {s['size_mm'][1]:.1f} x {s['size_mm'][2]:.1f} mm · "
           f"{'watertight' if s['watertight'] else 'NOT watertight'}")
    if problems:
        msg += "\n\nPrintability:\n- " + "\n- ".join(problems)
    return mesh_to_preview_file(mesh), rows, msg


def on_add(sketch, depth, session: Session):
    """The whole step-by-step flow in one button, for now."""
    return _commit(sketch, depth, Op.ADD, session)


def on_cut(sketch, depth, session: Session):
    return _commit(sketch, depth, Op.CUT, session)


def _commit(sketch, depth, op: Op, session: Session):
    # TODO (Serena): replace this placeholder with ui.canvas.strokes_from_image(sketch).
    # Until that exists, the app demonstrates the pipeline with a fixed shape so
    # that PK's half can be developed and demoed independently of the canvas.
    strokes = _placeholder_strokes(op)

    prims = RECOGNIZER.recognize_batch(strokes)
    profiles = [p for p in (profile_from_primitive(x) for x in prims) if p is not None]
    if not profiles:
        _, rows, _ = _refresh(session)
        return None, rows, "That shape is not closed - I cannot extrude it.", session

    session.submit_profile(profiles[0])
    session.choose_op(op)
    try:
        session.commit(depth=float(depth))
    except Exception as exc:                       # noqa: BLE001 - shown to the user
        _, rows, _ = _refresh(session)
        return None, rows, str(exc), session

    preview, rows, msg = _refresh(session)
    return preview, rows, msg, session


def _placeholder_strokes(op: Op) -> list[Stroke]:
    """Stand-in for real canvas input. DELETE once strokes_from_image works."""
    from sketch2stl.strokes import mm_to_px
    if op is Op.ADD:
        pts = []
        for (x0, y0), (x1, y1) in [((40, 30), (120, 30)), ((120, 30), (120, 90)),
                                   ((120, 90), (40, 90)), ((40, 90), (40, 30))]:
            for s in np.linspace(0, 1, 50, endpoint=False):
                pts.append([x0 + (x1 - x0) * s, y0 + (y1 - y0) * s])
        pts = np.asarray(pts)
    else:
        a = np.linspace(0, 2 * np.pi, 120)
        pts = np.column_stack([80 + 15 * np.cos(a), 60 + 15 * np.sin(a)])
    return [Stroke(points=mm_to_px(pts))]


def on_undo(session: Session):
    session.undo()
    preview, rows, msg = _refresh(session)
    return preview, rows, msg, session


def on_clear(session: Session):
    session = Session()
    return None, [], "Cleared. Draw a shape to begin.", session


def on_export(session: Session):
    mesh, err = session.solid()
    if mesh is None:
        return None, err or "Nothing to export yet."
    path = "/tmp/sketch2stl_export.stl"
    problems = export_stl(mesh, path)
    return path, ("Exported." if not problems
                  else "Exported, but: " + " ".join(problems))


with gr.Blocks(title="Sketch3D") as demo:
    session = gr.State(Session)

    gr.Markdown("## Sketch3D  ·  draw a shape, add or cut it, export an STL")

    with gr.Row():
        with gr.Column(scale=5):
            gr.Markdown("### 2D Sketch")
            sketch = gr.Sketchpad(height=CANVAS_H, width=CANVAS_W, label=None)
            with gr.Row():
                depth = gr.Number(value=DEFAULT_DEPTH_MM, label="Depth (mm)", precision=2)
                add_btn = gr.Button("Add volume", variant="primary")
                cut_btn = gr.Button("Cut volume")
            with gr.Row():
                undo_btn = gr.Button("Undo")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=5):
            gr.Markdown("### 3D Preview")
            preview = gr.Model3D(height=340, label=None)
            status = gr.Textbox(label="Status", lines=4, interactive=False,
                                value="Draw a shape to begin.")
            gr.Markdown("### Features / Layers")
            layers = gr.Dataframe(headers=HEADERS, label=None, interactive=False)
            export_btn = gr.Button("Export STL", variant="primary")
            stl_file = gr.File(label="STL")

    outs = [preview, layers, status, session]
    add_btn.click(on_add, [sketch, depth, session], outs)
    cut_btn.click(on_cut, [sketch, depth, session], outs)
    undo_btn.click(on_undo, [session], outs)
    clear_btn.click(on_clear, [session], outs)
    export_btn.click(on_export, [session], [stl_file, status])


if __name__ == "__main__":
    demo.launch()
