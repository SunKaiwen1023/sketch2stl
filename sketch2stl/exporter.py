"""STL export plus the printability checks that make the output actually usable.

OWNER: PK.

Exporting an STL is one line of trimesh. The value here is everything around it:
a mesh that is not watertight will slice into garbage, and a user who finds that
out in their slicer twenty minutes later will not come back. Check first, say so
plainly, export anyway with a warning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

from .config import MAX_BUILD_MM, MIN_WALL_MM


def check(mesh: trimesh.Trimesh) -> list[str]:
    """Return a list of human-readable problems. Empty list means good to print."""
    problems: list[str] = []

    if not mesh.is_watertight:
        problems.append(
            "The mesh is not watertight - it has holes in its surface. A slicer "
            "will produce unpredictable results. Try simplifying the sketch."
        )
    if mesh.volume <= 0:
        problems.append("The solid has zero or negative volume - its faces may be inverted.")

    size = mesh.bounds[1] - mesh.bounds[0]
    for axis, dim, limit in zip("XYZ", size, MAX_BUILD_MM):
        if dim > limit:
            problems.append(
                f"{dim:.0f} mm in {axis} exceeds a typical {limit} mm build volume. "
                f"It will need scaling or splitting."
            )
    if float(size.min()) < MIN_WALL_MM:
        problems.append(
            f"The thinnest dimension is {size.min():.2f} mm, below the {MIN_WALL_MM} mm "
            f"most FDM printers can hold. It may not survive removal from the bed."
        )
    return problems


def repair(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Best-effort cleanup before export. Non-destructive: works on a copy."""
    m = mesh.copy()
    m.remove_duplicate_faces() if hasattr(m, "remove_duplicate_faces") else None
    m.remove_unreferenced_vertices()
    m.merge_vertices()
    trimesh.repair.fix_normals(m)
    trimesh.repair.fill_holes(m)
    return m


def export_stl(mesh: trimesh.Trimesh, path: str | Path, do_repair: bool = True) -> list[str]:
    """Write a binary STL. Returns any remaining problems (empty = clean)."""
    m = repair(mesh) if do_repair else mesh
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m.export(str(path))
    return check(m)


# --------------------------------------------------------------------------- #
# TODO (PK):
#  1. Wall-thickness measurement that is more honest than a bounding box - ray
#     casting inwards from each face gives a real minimum thickness map.
#  2. Overhang detection: faces whose normal is more than ~45 degrees from
#     vertical need support. `mesh.face_normals` plus a dot product is enough to
#     colour them red in the preview, which is a great demo moment.
#  3. 3MF export alongside STL - it carries units, so nobody has to guess.
# --------------------------------------------------------------------------- #
