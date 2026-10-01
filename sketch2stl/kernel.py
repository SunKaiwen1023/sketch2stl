"""The geometry kernel: a Document (ordered feature stack) -> one solid mesh.

OWNER: PK.

This is the heart of the app and it is deliberately tiny. PK's constraint - keep
it at the extrusion level - means the entire modelling vocabulary is:

    extrude a closed profile from z_base to z_base+depth, then UNION or SUBTRACT
    it from the running body.

REVOLVE was added on 29 Sep after the review: the prof asked for more than
extrusion, and half-a-profile-around-a-centreline is how every CAD package does
a rotationally symmetric part. Sweep and loft are still out of scope.

NOTE FOR PK: you listed "revolve in the kernel" as yours. Serena wrote this on
29 Sep because half mode is useless without it - the whole point of drawing half
an arc against a centreline is to spin it into a ball. Delete or replace it
freely; the tests in tests/test_revolve_kernel.py are the contract either way.

BUILDING IS A PURE FUNCTION of the feature list. That gives you undo for free
(drop the last feature, rebuild), makes the whole thing testable with no UI, and
means the 3D preview can never drift out of sync with the layer panel.
"""
from __future__ import annotations

import numpy as np
import trimesh
from shapely.geometry import Polygon

from .config import CUT_OVERSHOOT_MM, MIN_FEATURE_DEPTH_MM
from .config import ARC_SEGMENTS
from .types import (Axis, Document, Feature, FeatureKind, Op, Profile)


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


def half_to_radius_height(points: np.ndarray, axis: Axis) -> np.ndarray:
    """A half profile drawn against a centreline -> (radius, height) pairs.

    The centreline the user drew becomes the model's vertical axis, so distance
    FROM the centreline is radius and distance ALONG it is height. That is what
    makes a half arc come out as a ball standing up, rather than a ball lying on
    its side in the sketch plane.

    Radius is taken as the absolute distance, so a stroke that wobbles a
    fraction of a millimetre over the line does not produce a negative radius
    and a self-intersecting solid.

    HEIGHT IS MEASURED FROM `axis.point`, and nothing is subtracted from it.
    An earlier version normalised each profile to start at height 0, which
    looked tidy and was badly wrong: it threw away WHERE on the centreline the
    shape was drawn, so every revolve started at z=0. A cut drawn deliberately
    across the middle of a part was silently moved to its base and took the
    wrong slice out. The axis now carries the datum - `ui.canvas.centerline_axis`
    puts its origin at the bottom of the canvas pointing up - so every feature
    in a document shares one vertical zero and lands where it was drawn.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    radius = np.abs(axis.signed_distance(pts))
    height = (pts - axis.point) @ axis.direction
    return np.column_stack([radius, height])


REVOLVE_SIMPLIFY_MM = 0.2   # how far the thinned outline may stray
AXIS_TOL_MM = 1e-3          # a micron: anything nearer the axis IS on it


def _clean_revolve_profile(rh: np.ndarray) -> np.ndarray:
    """Tidy a CLOSED (radius, height) ring so it is safe to sweep.

    Two jobs, and deliberately no third one:

      DEDUPE  consecutive points closer together than AXIS_TOL_MM sweep into a
              ring of zero area, which is a hole in the surface however it got
              there. A fitted stroke produces these where `sharpen()` puts a
              corner at the intersection of two lines and it lands a few
              billionths off the axis.
      THIN    Ramer-Douglas-Peucker to REVOLVE_SIMPLIFY_MM. A hand-traced half
              arrives as several hundred skeleton points; without this a ball
              comes out at 36,000 faces and the preview crawls.

    WHAT THIS NO LONGER DOES, and why it matters: an earlier version sorted the
    ring by height and merged it into one radius per height band, because
    `trimesh.creation.revolve` is usually shown with a monotone profile. That
    silently destroys any profile that is not a function of height - a tube,
    whose half is a rectangle standing away from the axis, has TWO radii at
    every height, and came out at a fifth of its true volume. Passing the
    closed ring through untouched revolves the loop itself, which is correct
    for a tube, a stepped shaft, a cylinder and a ball alike.
    """
    if len(rh) < 2:
        return rh
    keep = np.concatenate([[True],
                           (np.abs(np.diff(rh, axis=0)) > AXIS_TOL_MM).any(axis=1)])
    rh = rh[keep]
    return _rdp(rh, REVOLVE_SIMPLIFY_MM) if len(rh) > 2 else rh


def _ring_area(ring: np.ndarray) -> float:
    """Signed shoelace area. Positive means counter-clockwise."""
    x, y = ring[:, 0], ring[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _rdp(points: np.ndarray, tol: float) -> np.ndarray:
    """Ramer-Douglas-Peucker, iterative so a long outline cannot blow the stack."""
    n = len(points)
    if n < 3:
        return points
    keep = np.zeros(n, dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = points[i], points[j]
        seg = b - a
        length = float(np.linalg.norm(seg))
        chunk = points[i + 1:j]
        if length < 1e-12:
            dist = np.linalg.norm(chunk - a, axis=1)
        else:
            rel = chunk - a
            # 2-D cross product by hand: np.cross on 2-vectors is deprecated.
            dist = np.abs(seg[0] * rel[:, 1] - seg[1] * rel[:, 0]) / length
        k = int(np.argmax(dist))
        if dist[k] > tol:
            keep[i + 1 + k] = True
            stack.append((i, i + 1 + k))
            stack.append((i + 1 + k, j))
    return points[keep]


def revolve(profile_points: np.ndarray, axis: Axis,
            angle: float = 2.0 * np.pi) -> trimesh.Trimesh:
    """Spin a closed half profile around its centreline.

    `profile_points` is a CLOSED ring in millimetres, in sketch space, lying on
    one side of `axis`: the half the user drew plus a return leg down the
    centreline (see `profiles.close_to_axis`).

    The ring is mapped to (radius, height) and revolved AS A LOOP. That is the
    whole design decision, and it is worth stating because the obvious
    alternative is wrong: `trimesh.creation.revolve` is usually demonstrated
    with a monotone profile - one radius per height - and it is tempting to
    sort the points by height to force that. Doing so quietly destroys every
    profile that is not a function of height. A tube's half is a rectangle
    standing away from the axis, with two radii at every height; sorted, it
    came out at a fifth of its true volume, watertight and plausible-looking.

    Revolving the closed loop instead handles a ball, a cylinder, a stepped
    shaft and a tube with the same code. Points on the axis sweep to nothing,
    and trimesh drops the zero-area triangles they produce.
    """
    ring = np.asarray(profile_points, dtype=np.float64).reshape(-1, 2)
    if len(ring) > 1 and np.linalg.norm(ring[0] - ring[-1]) > AXIS_TOL_MM:
        ring = np.vstack([ring, ring[:1]])           # tolerate an open profile

    rh = half_to_radius_height(ring, axis)

    # These two checks go BEFORE thinning, so the user gets the specific reason
    # rather than the generic "too short" the thinned ring would produce.
    if rh[:, 0].max() <= AXIS_TOL_MM:
        raise KernelError("That half profile sits on the centreline - it has no "
                          "radius to sweep. Draw it to one side.")
    if rh[:, 1].max() - rh[:, 1].min() <= AXIS_TOL_MM:
        raise KernelError("That half profile has no height to sweep.")

    rh = _clean_revolve_profile(rh)
    if len(rh) < 4:
        raise KernelError("That half profile is too short to revolve.")

    area = _ring_area(rh)
    if abs(area) <= AXIS_TOL_MM:
        raise KernelError("That half profile encloses no area, so there is "
                          "nothing to sweep. Draw a closed half.")
    if area < 0:
        # trimesh wants a counter-clockwise loop or the normals point inwards.
        rh = rh[::-1]

    if np.linalg.norm(rh[0] - rh[-1]) > AXIS_TOL_MM:
        rh = np.vstack([rh, rh[:1]])

    try:
        mesh = trimesh.creation.revolve(rh, angle=float(angle),
                                        sections=ARC_SEGMENTS,
                                        cap=angle < 2 * np.pi - 1e-9)
    except Exception as exc:                        # noqa: BLE001
        raise KernelError(f"Could not revolve that profile ({type(exc).__name__}). "
                          f"Try a simpler half outline.") from exc
    if mesh is None or mesh.is_empty:
        raise KernelError("That revolve produced nothing.")
    if mesh.volume < 0:
        mesh.invert()
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
        if feat.kind is FeatureKind.REVOLVE:
            tool = revolve(feat.profile.outer, feat.axis, feat.angle)
            if feat.z_base:
                tool.apply_translation([0.0, 0.0, float(feat.z_base)])
            if feat.op is Op.CUT:
                # A revolve has no "depth" to overshoot, so nudge it down and
                # scale it a hair proud of the body to avoid coplanar faces.
                tool.apply_translation([0.0, 0.0, -CUT_OVERSHOOT_MM])
        elif feat.op is Op.CUT:
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
