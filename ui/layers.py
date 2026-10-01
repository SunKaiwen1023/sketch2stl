"""The Features/Layers panel: a Document -> a table the user can read.

OWNER: PK.

Gradio has no tree widget, so the mockup's layer stack becomes a DataFrame plus
a row of buttons. Less pretty, same information, and the ordering is what
actually matters - the list IS the model.
"""
from __future__ import annotations

from sketch2stl.types import Document, FeatureKind

HEADERS = ["#", "Name", "Operation", "Depth (mm)", "Base Z (mm)", "Visible"]

_KIND = {
    FeatureKind.EXTRUDE: "Extrude",
    FeatureKind.REVOLVE: "Revolve",
    FeatureKind.MIRROR_EXTRUDE: "Mirror + extrude",
}


def to_rows(doc: Document) -> list[list]:
    """Newest feature last, matching the build order."""
    rows = []
    for i, f in enumerate(doc.features):
        # Label by what the feature DOES, not by its position. Row 1 used to read
        # "Extrude (New Body)" unconditionally, so a cut placed first - which is
        # an error, there is nothing to cut from - looked like a valid base solid.
        #
        # And say HOW the volume was made, not just what it did with it. A
        # revolved cut showed up here as "Extrude (Cut)", which is the only
        # place the user could check what a feature actually is - so the one
        # place it must not lie.
        kind = _KIND.get(f.kind, str(f.kind))
        if f.op.value == "cut":
            what = "Cut"
        else:
            what = "New Body" if i == 0 else "Add"
        depth = "-" if f.kind is FeatureKind.REVOLVE else round(f.depth, 2)
        rows.append([i + 1, f.name, f"{kind} ({what})", depth,
                     round(f.z_base, 2), "yes" if f.visible else "no"])
    return rows


# TODO (PK): let the user click a row to edit its depth, and re-run build().
# Because build() is a pure function of the feature list, editing a depth and
# rebuilding is already correct - it just needs wiring.
