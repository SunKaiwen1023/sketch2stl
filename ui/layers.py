"""The Features/Layers panel: a Document -> a table the user can read.

OWNER: PK.

Gradio has no tree widget, so the mockup's layer stack becomes a DataFrame plus
a row of buttons. Less pretty, same information, and the ordering is what
actually matters - the list IS the model.
"""
from __future__ import annotations

from sketch2stl.types import Document

HEADERS = ["#", "Name", "Operation", "Depth (mm)", "Base Z (mm)", "Visible"]


def to_rows(doc: Document) -> list[list]:
    """Newest feature last, matching the build order."""
    rows = []
    for i, f in enumerate(doc.features):
        # Label by what the feature DOES, not by its position. Row 1 used to read
        # "Extrude (New Body)" unconditionally, so a cut placed first - which is
        # an error, there is nothing to cut from - looked like a valid base solid.
        if f.op.value == "cut":
            op = "Extrude (Cut)"
        else:
            op = "Extrude (New Body)" if i == 0 else "Extrude (Add)"
        rows.append([i + 1, f.name, op, round(f.depth, 2), round(f.z_base, 2),
                     "yes" if f.visible else "no"])
    return rows


# TODO (PK): let the user click a row to edit its depth, and re-run build().
# Because build() is a pure function of the feature list, editing a depth and
# rebuilding is already correct - it just needs wiring.
