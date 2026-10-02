"""3D preview: mesh -> something Gradio can display.

OWNER: PK.

gr.Model3D takes a file path, so the preview is just "write a temporary GLB and
hand over the path". GLB rather than STL because Model3D renders it with
materials and STL comes out flat grey.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import trimesh

PREVIEW_RGBA = (90, 160, 230, 255)     # Figma-ish blue, so edges and holes shade visibly


def mesh_to_preview_file(mesh: trimesh.Trimesh | None) -> str | None:
    """Write the mesh somewhere gr.Model3D can read it. None if there is no mesh."""
    if mesh is None:
        return None
    # A NEW file every time. One fixed name was shared by every visitor and could
    # be served from cache, so after a Cut the preview could keep showing the
    # part from before - the cut looked like it had not happened.
    fd, name = tempfile.mkstemp(prefix="sketch2stl_preview_", suffix=".glb")
    with open(fd, "wb") as fh:
        fh.write(_for_viewer(mesh).export(file_type="glb"))
    return name


def _for_viewer(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """A copy that reads well in gr.Model3D: centred, Z-up turned into the
    viewer's Y-up, and given a colour. Plain white on the default background
    made a plate with a hole look like a featureless blob, so a Cut seemed not
    to have happened. Only the preview changes - the STL is untouched."""
    m = mesh.copy()
    m.apply_translation(-m.bounding_box.centroid)
    m.apply_scale(100.0 / max(float(m.extents.max()), 1e-6))   # same size on screen whatever the part
    m.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    m.visual = trimesh.visual.ColorVisuals(m, face_colors=PREVIEW_RGBA)
    return m


# TODO (PK): colour overhang faces red before export - mesh.visual.face_colors
# with a mask on face_normals[:, 2]. Costs ten lines and makes the printability
# story visible instead of textual.
