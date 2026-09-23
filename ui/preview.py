"""3D preview: mesh -> something Gradio can display.

OWNER: PK.

gr.Model3D takes a file path, so the preview is just "write a temporary GLB and
hand over the path". GLB rather than STL because Model3D renders it with
materials and STL comes out flat grey.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import trimesh


def mesh_to_preview_file(mesh: trimesh.Trimesh | None) -> str | None:
    """Write the mesh somewhere gr.Model3D can read it. None if there is no mesh."""
    if mesh is None:
        return None
    path = Path(tempfile.gettempdir()) / "sketch2stl_preview.glb"
    mesh.export(str(path))
    return str(path)


# TODO (PK): colour overhang faces red before export - mesh.visual.face_colors
# with a mask on face_normals[:, 2]. Costs ten lines and makes the printability
# story visible instead of textual.
