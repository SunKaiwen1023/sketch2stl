"""The geometry kernel: a Document (ordered feature stack) -> one solid mesh.

OWNER: PK.

This is the heart of the app and it is deliberately tiny. PK's constraint - keep
it at the extrusion level - means the entire modelling vocabulary is:

    extrude a closed profile from z_base to z_base+depth, then UNION or SUBTRACT
    it from the running body.

That is enough to build the plate in the mockup, it is enough for most printable
parts, and it avoids every hard problem in solid modelling. Resist adding
revolve or sweep before the presentation.

BUILDING IS A PURE FUNCTION of the feature list. That gives you undo for free
(drop the last feature, rebuild), makes the whole thing testable with no UI, and
means the 3D preview can never drift out of sync with the layer panel.
"""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

from .config import CUT_OVERSHOOT_MM, MIN_FEATURE_DEPTH_MM
from .types import Document, Feature, Op, Profile


class KernelError(RuntimeError):
    """Raised when a feature cannot be built. Carries a message meant for the user."""


def _profile_to_polygon(profile: Profile) -> Polygon:
    poly = Polygon(profile.outer, [h for h in profile.holes])
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 0:
        raise KernelError("This shape encloses no area - try drawing a closed outline.")
    return poly


def extrude(profile: Profile, depth: float, z_base: float = 0.0) -> trimesh.Trimesh:
    """Closed profile -> a solid prism, sitting at z_base."""
    if depth < MIN_FEATURE_DEPTH_MM:
        raise KernelError(f"Depth must be at least {MIN_FEATURE_DEPTH_MM} mm.")
    poly = _profile_to_polygon(profile)
    mesh = trimesh.creation.extrude_polygon(poly, height=float(depth))
    if z_base:
        mesh.apply_translation([0.0, 0.0, float(z_base)])
    return mesh


def _boolean(a: trimesh.Trimesh, b: trimesh.Trimesh, op: Op) -> trimesh.Trimesh:
    """Union or difference, with a readable error instead of a library traceback."""
    try:
        if op is Op.ADD:
            out = trimesh.boolean.union([a, b])
        else:
            out = trimesh.boolean.difference([a, b])
    except Exception as exc:                        # noqa: BLE001 - surfaced to the user
        raise KernelError(
            f"The {op.value} operation failed. This usually means the shapes touch "
            f"exactly at a face; nudging the depth by 0.1 mm normally fixes it. "
            f"({type(exc).__name__})"
        ) from exc

    if isinstance(out, list):
        if not out:
            raise KernelError("That cut removed the entire body.")
        out = max(out, key=lambda m: m.volume)
    if out is None or out.is_empty or len(out.faces) == 0:
        raise KernelError("That cut removed the entire body.")
    return out


def build(doc: Document) -> trimesh.Trimesh | None:
    """Replay the feature stack into a single mesh. None if there is nothing yet.

    A CUT is extruded slightly taller than requested and started slightly lower,
    so it always pokes out of both faces of the body it is cutting. Coplanar
    faces are the number one cause of boolean failures, and this one line avoids
    almost all of them.
    """
    body: trimesh.Trimesh | None = None

    for feat in doc.active():
        if feat.op is Op.CUT:
            tool = extrude(feat.profile,
                           feat.depth + 2 * CUT_OVERSHOOT_MM,
                           feat.z_base - CUT_OVERSHOOT_MM)
        else:
            tool = extrude(feat.profile, feat.depth, feat.z_base)

        if body is None:
            if feat.op is Op.CUT:
                raise KernelError(
                    f"'{feat.name}' is a cut, but there is nothing to cut from yet. "
                    f"Draw a shape and add it as a new body first."
                )
            body = tool
        else:
            body = _boolean(body, tool, feat.op)

    return body


def stats(mesh: trimesh.Trimesh) -> dict[str, float | bool | tuple]:
    """Numbers for the UI panel and for the printability check."""
    bounds = mesh.bounds
    size = bounds[1] - bounds[0]
    return {
        "watertight": bool(mesh.is_watertight),
        "volume_mm3": float(mesh.volume),
        "area_mm2": float(mesh.area),
        "n_faces": int(len(mesh.faces)),
        "size_mm": tuple(float(v) for v in size),
        "euler": int(mesh.euler_number),
    }


# --------------------------------------------------------------------------- #
# TODO (PK), in order:
#
#  1. SKETCH PLANES. Everything currently extrudes along +Z from the XY plane.
#     The mockup's "Top" dropdown implies Front and Right too. The change is
#     small - build in XY as now, then apply a rotation to the resulting mesh -
#     but decide the convention early and write it in docs/architecture.md.
#
#  2. EXTRUDE FROM A FACE. Right now z_base is typed by the user. Letting them
#     pick the top face of the existing body and start there is the single
#     biggest step towards feeling like real CAD. `mesh.facets` gives you the
#     candidate planes.
#
#  3. FILLETS/CHAMFERS on vertical edges. Cheap version: buffer the 2-D profile
#     inwards and outwards before extruding (shapely's `buffer` with
#     join_style=1 rounds corners). Gets you the rounded rectangle in the mockup
#     without touching 3-D at all.
#
#  4. CACHING. Rebuilding the whole stack on every edit is fine at five features
#     and slow at fifty. Cache the mesh after each feature and rebuild only from
#     the first changed one.
# --------------------------------------------------------------------------- #
