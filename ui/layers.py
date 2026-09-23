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
    return [
        [i + 1, f.name,
         "Extrude (New Body)" if i == 0 else
         ("Extrude (Cut)" if f.op.value == "cut" else "Extrude (Add)"),
         round(f.depth, 2), round(f.z_base, 2), "yes" if f.visible else "no"]
        for i, f in enumerate(doc.features)
    ]


# TODO (PK): let the user click a row to edit its depth, and re-run build().
# Because build() is a pure function of the feature list, editing a depth and
# rebuilding is already correct - it just needs wiring.
