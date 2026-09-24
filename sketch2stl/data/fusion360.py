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

Sketch curves are stored as 3-D points in world space, but they all lie on the
sketch's plane. Each sketch carries a `transform` with an origin and x/y/z axes,
so the 2-D sketch coordinate of a point is just its projection onto that basis:

    u = (p - origin) . x_axis
    v = (p - origin) . y_axis

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
    if d is None:
        return np.zeros(3)
    return np.array([float(d.get(k, 0.0)) for k in keys])


def _basis(sketch: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(origin, x_axis, y_axis) for a sketch, in cm. Falls back to world XY."""
    t = sketch.get("transform") or sketch.get("reference_plane") or {}
    origin = _vec(t.get("origin"))
    x_axis = _vec(t.get("x_axis")) if t.get("x_axis") else np.array([1.0, 0.0, 0.0])
    y_axis = _vec(t.get("y_axis")) if t.get("y_axis") else np.array([0.0, 1.0, 0.0])
    for ax, default in ((x_axis, [1.0, 0, 0]), (y_axis, [0, 1.0, 0])):
        n = np.linalg.norm(ax)
        if n < 1e-9:
            ax[:] = default
        else:
            ax /= n
    return origin, x_axis, y_axis


def _to_2d(p3: np.ndarray, origin: np.ndarray,
           x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """World 3-D point (cm) -> sketch 2-D point (mm)."""
    d = p3 - origin
    return np.array([float(d @ x_axis), float(d @ y_axis)]) * CM_TO_MM


def _line(curve: dict, o, xa, ya) -> tuple[np.ndarray, dict] | None:
    s = curve.get("start_point")
    e = curve.get("end_point")
    if s is None or e is None:
        return None
    p0 = _to_2d(_vec(s), o, xa, ya)
    p1 = _to_2d(_vec(e), o, xa, ya)
    if np.linalg.norm(p1 - p0) < 1e-6:
        return None
    return np.vstack([p0, p1]), {"x1": p0[0], "y1": p0[1], "x2": p1[0], "y2": p1[1]}


def _circle(curve: dict, o, xa, ya) -> tuple[np.ndarray, dict] | None:
    c = curve.get("center_point")
    r = curve.get("radius")
    if c is None or r is None:
        return None
    centre = _to_2d(_vec(c), o, xa, ya)
    radius = float(r) * CM_TO_MM
    if radius <= 0:
        return None
    a = np.linspace(0.0, 2 * np.pi, ARC_SEGMENTS, endpoint=False)
    pts = np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])
    pts = np.vstack([pts, pts[:1]])
    return pts, {"cx": centre[0], "cy": centre[1], "r": radius}


def _arc(curve: dict, o, xa, ya) -> tuple[np.ndarray, dict] | None:
    c = curve.get("center_point")
    r = curve.get("radius")
    if c is None or r is None:
        return None
    centre = _to_2d(_vec(c), o, xa, ya)
    radius = float(r) * CM_TO_MM
    if radius <= 0:
        return None

    # Prefer the stored angles; fall back to deriving them from the endpoints,
    # because not every record carries both.
    a0 = curve.get("start_angle")
    a1 = curve.get("end_angle")
    if a0 is None or a1 is None:
        s = curve.get("start_point")
        e = curve.get("end_point")
        if s is None or e is None:
            return None
        p0 = _to_2d(_vec(s), o, xa, ya) - centre
        p1 = _to_2d(_vec(e), o, xa, ya) - centre
        a0, a1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
    a0, a1 = float(a0), float(a1)
    if a1 <= a0:
        a1 += 2 * np.pi

    sweep = a1 - a0
    n = max(8, int(ARC_SEGMENTS * sweep / (2 * np.pi)))
    a = np.linspace(a0, a1, n)
    pts = np.column_stack([centre[0] + radius * np.cos(a), centre[1] + radius * np.sin(a)])
    return pts, {"cx": centre[0], "cy": centre[1], "r": radius, "a0": a0, "a1": a1}


_BUILDERS = {PrimitiveKind.LINE: _line, PrimitiveKind.CIRCLE: _circle, PrimitiveKind.ARC: _arc}


def curves_from_model(data: dict, model_id: str = "") -> list[CadCurve]:
    """All 2-D sketch curves in one reconstruction JSON."""
    out: list[CadCurve] = []
    entities = data.get("entities") or {}

    for sketch_id, ent in entities.items():
        if ent.get("type") != "Sketch":
            continue
        o, xa, ya = _basis(ent)

        # Curves appear in two places: a flat `curves` dict, and nested inside
        # `profiles -> loops -> profile_curves`. Walk both and de-duplicate,
        # because which one is populated varies across the dataset.
        raw: list[dict] = []
        cdict = ent.get("curves")
        if isinstance(cdict, dict):
            raw.extend(c for c in cdict.values() if isinstance(c, dict))
        for prof in (ent.get("profiles") or {}).values():
            for loop in prof.get("loops") or []:
                raw.extend(c for c in (loop.get("profile_curves") or []) if isinstance(c, dict))

        seen: set[tuple] = set()
        for curve in raw:
            kind = CURVE_TYPE_MAP.get(curve.get("type") or curve.get("curve_type") or "")
            if kind is None:
                continue
            built = _BUILDERS[kind](curve, o, xa, ya)
            if built is None:
                continue
            pts, params = built
            key = (kind, tuple(np.round(pts[0], 4)), tuple(np.round(pts[-1], 4)),
                   round(params.get("r", 0.0), 4))
            if key in seen:
                continue
            seen.add(key)
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
