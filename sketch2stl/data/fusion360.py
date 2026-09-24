"""Read Fusion 360 Gallery *Reconstruction* JSON files into 2-D labelled curves.

OWNER: Serena.

WHAT THIS DATASET ACTUALLY IS, AND WHY THAT MATTERS
---------------------------------------------------
The Reconstruction subset is 8,625 real CAD models, each a *construction
sequence*: sketches made of exact parametric curves, followed by extrude
features that add or cut material. It contains **no hand-drawn strokes at all**.

So it cannot train a stroke recogniser directly. What it can do is better:

    every curve in it carries a perfect ground-truth label.

A `Circle3D` entry says, exactly, "this is a circle, centre here, radius this".
Pair that with `data/synth.py`, which turns a clean curve into a plausible
hand-drawn version of itself, and you get tens of thousands of perfectly
labelled training strokes out of a dataset that has none - drawn from the
distribution of shapes that appear in *real engineering parts*, not from shapes
someone invented for a homework.

The strokes you and PK draw by hand then become the TEST set. Train on
synthetic, test on real: that is a domain-gap study, and it is a much more
interesting result than "we drew 400 shapes and got 94%".

UNITS AND GEOMETRY
------------------
Fusion 360 works in centimetres and radians. Everything returned here is
converted to MILLIMETRES to match the rest of this codebase.

Curve coordinates are already in each sketch's OWN frame, so x/y are the sketch
coordinates directly - no projection onto the sketch basis. See `_to_2d` for the
evidence in the data.

Points come in two forms depending on where the curve lives: inline dicts inside
`profiles`, or UUID references into the sketch's `points` table inside `curves`.
`_resolve` handles both.

LICENSE - READ THIS
-------------------
The Fusion 360 Gallery Dataset is under a custom Autodesk licence, NOT an open
one. Non-commercial research only. Coursework qualifies. But:

  * Do NOT commit the dataset to git, and do NOT push it or anything derived
    from it to a public Hugging Face dataset the way you did for HW1.
  * Derived data (your synthesised strokes) is a "Modified Set" and may only be
    shared under terms at least as restrictive, clearly marked as a
    modification, with attribution.
  * A trained MODEL is fine to share - it is not the dataset.
  * Attribute "Fusion 360 Gallery Dataset" in the report.

See docs/dataset.md for the full summary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from ..config import ARC_SEGMENTS
from ..types import PrimitiveKind

CM_TO_MM = 10.0

# How Fusion names curve types -> our vocabulary.
CURVE_TYPE_MAP = {
    "Line3D": PrimitiveKind.LINE,
    "SketchLine": PrimitiveKind.LINE,
    "Arc3D": PrimitiveKind.ARC,
    "SketchArc": PrimitiveKind.ARC,
    "Circle3D": PrimitiveKind.CIRCLE,
    "SketchCircle": PrimitiveKind.CIRCLE,
}

# Fusion extrude operations -> our ADD / CUT vocabulary.
OPERATION_MAP = {
    "NewBodyFeatureOperation": "add",
    "JoinFeatureOperation": "add",
    "CutFeatureOperation": "cut",
    "IntersectFeatureOperation": "intersect",   # we do not support this
}


@dataclass
class CadCurve:
    """One parametric curve from one sketch, in 2-D sketch space, millimetres."""
    kind: PrimitiveKind
    points: np.ndarray            # (M, 2) polygonal approximation, mm
    params: dict[str, float]      # exact parameters - the ground truth
    model_id: str
    sketch_id: str


def _vec(d: dict | None, keys=("x", "y", "z")) -> np.ndarray:
    """Pull a vector out of Fusion's {"x":..,"y":..,"z":..} dicts."""
    if not isinstance(d, dict):
        return np.zeros(3)
    return np.array([float(d.get(k, 0.0)) for k in keys])


def _resolve(ref, points: dict) -> dict | None:
    """A point is either inline, or a UUID into the sketch's `points` table.

    The two places curves live use different conventions:

      profiles -> loops -> profile_curves   inline {"type":"Point3D","x":..}
      curves                                "c7780e6e-e2f3-11ea-..." (a UUID)

    Writing the parser against only the first form is what made
    `'str' object has no attribute 'get'` show up on the very first real file.
    """
    if isinstance(ref, dict):
        return ref
    if isinstance(ref, str):
        return points.get(ref)
    return None


def _to_2d(p: dict | None, points: dict) -> np.ndarray | None:
    """A curve's point -> 2-D sketch coordinates in MILLIMETRES.

    NO PROJECTION IS NEEDED - the coordinates are already in the sketch's own
    frame, not world space. The evidence is in the data: a circle whose stored
    `normal` is (0, 0, 1) sits in a sketch whose `z_axis` is (1, 0, 0). If the
    curve were in world coordinates its normal would have to match the sketch's
    z-axis. It does not, so the curve is expressed sketch-locally and x/y are
    the sketch coordinates directly.

    Projecting onto the sketch basis - which is what this function used to do -
    silently produced garbage for every sketch not on the world XY plane.
    """
    d = _resolve(p, points)
    if d is None:
        return None
    return np.array([float(d.get("x", 0.0)), float(d.get("y", 0.0))]) * CM_TO_MM


def _line(curve: dict, points: dict) -> tuple[np.ndarray, dict] | None:
    p0 = _to_2d(curve.get("start_point"), points)
    p1 = _to_2d(curve.get("end_point"), points)
    if p0 is None or p1 is None or np.linalg.norm(p1 - p0) < 1e-6:
        return None
    return np.vstack([p0, p1]), {"x1": p0[0], "y1": p0[1], "x2": p1[0], "y2": p1[1]}


def _circle(curve: dict, points: dict) -> tuple[np.ndarray, dict] | None:
    centre = _to_2d(curve.get("center_point"), points)
    r = curve.get("radius")
    if centre is None or r is None:
        return None
    radius = float(r) * CM_TO_MM
    if radius <= 0:
        return None
    a = np.linspace(0.0, 2 * np.pi, ARC_SEGMENTS, endpoint=False)
    pts = np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])
    return np.vstack([pts, pts[:1]]), {"cx": centre[0], "cy": centre[1], "r": radius}


def _arc(curve: dict, points: dict) -> tuple[np.ndarray, dict] | None:
    centre = _to_2d(curve.get("center_point"), points)
    r = curve.get("radius")
    if centre is None or r is None:
        return None
    radius = float(r) * CM_TO_MM
    if radius <= 0:
        return None

    # Prefer the stored angles; derive them from the endpoints otherwise,
    # because not every record carries both.
    a0, a1 = curve.get("start_angle"), curve.get("end_angle")
    if a0 is None or a1 is None:
        s = _to_2d(curve.get("start_point"), points)
        e = _to_2d(curve.get("end_point"), points)
        if s is None or e is None:
            return None
        d0, d1 = s - centre, e - centre
        a0, a1 = np.arctan2(d0[1], d0[0]), np.arctan2(d1[1], d1[0])
    a0, a1 = float(a0), float(a1)
    if a1 <= a0:
        a1 += 2 * np.pi

    sweep = a1 - a0
    n = max(8, int(ARC_SEGMENTS * sweep / (2 * np.pi)))
    a = np.linspace(a0, a1, n)
    pts = np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])
    return pts, {"cx": centre[0], "cy": centre[1], "r": radius, "a0": a0, "a1": a1}


_BUILDERS = {PrimitiveKind.LINE: _line, PrimitiveKind.CIRCLE: _circle,
             PrimitiveKind.ARC: _arc}


def curves_from_model(data: dict, model_id: str = "") -> list[CadCurve]:
    """All 2-D sketch curves in one reconstruction JSON.

    Curves live in two places and the PROFILES path is strictly better:

      profiles -> loops -> profile_curves   inline coordinates, plus `is_outer`
                                            telling you outer ring vs hole
      curves                                UUID references into `points`, and
                                            includes construction geometry

    So profiles first, the curve table only as a fallback for sketches that
    were never turned into a profile.
    """
    out: list[CadCurve] = []
    entities = data.get("entities") or {}

    for sketch_id, ent in entities.items():
        if ent.get("type") != "Sketch":
            continue
        points = ent.get("points") or {}

        raw: list[tuple[dict, bool]] = []
        for prof in (ent.get("profiles") or {}).values():
            for loop in prof.get("loops") or []:
                for c in loop.get("profile_curves") or []:
                    if isinstance(c, dict):
                        raw.append((c, bool(loop.get("is_outer", True))))

        if not raw:
            for c in (ent.get("curves") or {}).values():
                # construction geometry is scaffolding the designer drew to
                # position real geometry against - it is not part of the shape
                if isinstance(c, dict) and not c.get("construction_geom"):
                    raw.append((c, True))

        seen: set[tuple] = set()
        for curve, is_outer in raw:
            kind = CURVE_TYPE_MAP.get(curve.get("type") or curve.get("curve_type") or "")
            if kind is None:
                continue
            built = _BUILDERS[kind](curve, points)
            if built is None:
                continue
            pts, params = built
            key = (kind, tuple(np.round(pts[0], 4)), tuple(np.round(pts[-1], 4)),
                   round(params.get("r", 0.0), 4))
            if key in seen:
                continue
            seen.add(key)
            params = {**params, "is_outer": float(is_outer)}
            out.append(CadCurve(kind, pts, params, model_id, sketch_id))

    return out


def operations_from_model(data: dict) -> list[str]:
    """The extrude operations in timeline order: 'add', 'cut' or 'intersect'.

    Not needed for training the recogniser, but it is a free, real-world prior
    on how often designers cut versus add - a nice line in the report, and a
    sanity check that ADD/CUT is the right vocabulary for real parts.
    """
    entities = data.get("entities") or {}
    timeline = data.get("timeline") or []
    ops = []
    for step in timeline:
        ent = entities.get(step.get("entity", ""))
        if not ent or ent.get("type") != "ExtrudeFeature":
            continue
        op = OPERATION_MAP.get(ent.get("operation", ""), None)
        if op:
            ops.append(op)
    return ops


def iter_models(root: str | Path, model_ids: list[str] | None = None,
                limit: int | None = None) -> Iterator[tuple[str, dict]]:
    """Yield (model_id, parsed_json) from a reconstruction directory.

    `root` is the folder holding the *.json files - usually r1.0.1/reconstruction.
    `model_ids` restricts to a list, e.g. the 'train' entry of train_test.json.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"{root} does not exist. Point --data at r1.0.1/reconstruction.")

    if model_ids is not None:
        paths = [root / f"{m}.json" for m in model_ids]
    else:
        paths = sorted(root.glob("*.json"))

    n = 0
    for path in paths:
        if not path.exists():
            continue
        try:
            with open(path, encoding="utf-8") as f:
                yield path.stem, json.load(f)
        except Exception:
            continue                       # a handful of files in any dataset are broken
        n += 1
        if limit is not None and n >= limit:
            return


def load_split(split_path: str | Path) -> dict[str, list[str]]:
    """Read the official train_test.json. Returns {'train': [...], 'test': [...]}.

    USE THIS. Splitting the curves randomly would put curves from the same CAD
    model on both sides, and a model's curves are highly self-similar - the same
    leakage problem as HW2's parent_id, wearing a CAD hat.
    """
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    return {"train": list(split.get("train", [])), "test": list(split.get("test", []))}
