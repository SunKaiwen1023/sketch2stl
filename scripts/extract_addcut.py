#!/usr/bin/env python3
"""Pull an ADD/CUT training set out of the Fusion 360 reconstruction dataset.

    python scripts/extract_addcut.py --data ~/Downloads/r1.0.1/reconstruction \
                                     --out data/addcut.csv

WHY THIS EXISTS
---------------
`axisym.csv` answers a different question. It labels each design as
axisymmetric or not, which is the REVOLVE question - is this half meant to be
spun? It contains nothing at all about add versus cut, so no amount of
retraining on it will produce the suggestion the UI needs.

The good news is that no new data has to be collected or hand-labelled. Every
extrude in these same JSON files already records what it did to the body:

    NewBodyFeatureOperation     -> ADD      (a new solid)
    JoinFeatureOperation        -> ADD      (union with the body)
    CutFeatureOperation         -> CUT      (difference)
    IntersectFeatureOperation   -> skipped  (we do not offer intersect)

So this is one extraction pass over data already on disk, not a collection
effort. That is the whole answer to "do we need more data".

WHAT COMES OUT
--------------
One row per extrude step, with the label and the geometry that was visible at
the moment the user drew it - never anything from later steps, which would be
leakage. The features are deliberately the ones the app can also compute at
draw time from a sketched profile, because a feature the app cannot reproduce
is useless however well it scores here:

    step_index        how many extrudes came before this one
    n_loops           loops in the profile (an outer plus any inner rings)
    has_inner_loop    an annulus drawn as one profile
    area_mm2          area of this profile
    area_ratio        its area over the largest area drawn so far
    overlap_frac      fraction of THIS profile covered by earlier profiles,
                      projected onto this sketch plane. The single strongest
                      signal, and exactly what the rules baseline uses.
    inside_frac       fraction that sits inside a single earlier profile
    centroid_dist     distance from this centroid to the nearest earlier one,
                      divided by sqrt(that profile's area) so it is scale free
    is_circular       every curve is a circle or arc about one centre
    coplanar_prior    whether any earlier sketch shared this plane's normal

SELF-CHECK
----------
If the operation field is missing or named something else in your copy of the
dataset, the script says so and prints the keys it actually found on an extrude
rather than silently writing an empty file.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from shapely.geometry import Polygon
    from shapely.ops import unary_union
except ImportError:                                  # pragma: no cover
    print("shapely is needed: pip install shapely")
    raise

ARC_SAMPLES = 24
OP_TO_LABEL = {
    "NewBodyFeatureOperation": "add",
    "JoinFeatureOperation": "add",
    "CutFeatureOperation": "cut",
}
OP_KEYS = ("operation", "operation_type", "extrude_operation")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

def vec(p) -> np.ndarray:
    return np.array([p["x"], p["y"], p["z"]], dtype=np.float64)


def sketch_frame(sketch):
    """Origin and the two in-plane axes of a sketch, in world coordinates."""
    t = sketch["transform"]
    return vec(t["origin"]), vec(t["x_axis"]), vec(t["y_axis"]), vec(t["z_axis"])


def to_2d(point3, origin, xa, ya) -> np.ndarray:
    """A sketch-space point already lies in the plane; keep its 2D coordinates.

    The dataset stores curve points in SKETCH space, so x and y are already the
    in-plane coordinates. `origin`/`xa`/`ya` are carried through only so the
    same function can be pointed at world-space points if a future version of
    the dataset stores them that way.
    """
    p = vec(point3)
    if abs(p[2]) < 1e-9:
        return p[:2]
    rel = p - origin
    return np.array([rel @ xa, rel @ ya])


def sample_curve(curve, frame) -> np.ndarray:
    """A curve -> a polyline of 2D points. Unknown types degrade, never crash."""
    origin, xa, ya, _ = frame
    kind = curve.get("type", "")

    def pt(key):
        return to_2d(curve[key], origin, xa, ya)

    if kind == "Line3D":
        return np.vstack([pt("start_point"), pt("end_point")])

    if kind == "Circle3D":
        c, r = pt("center_point"), float(curve["radius"])
        a = np.linspace(0, 2 * np.pi, ARC_SAMPLES * 2, endpoint=False)
        return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])

    if kind == "Arc3D":
        c, r = pt("center_point"), float(curve["radius"])
        a0 = np.arctan2(*(pt("start_point") - c)[::-1])
        a1 = np.arctan2(*(pt("end_point") - c)[::-1])
        if a1 <= a0:
            a1 += 2 * np.pi
        a = np.linspace(a0, a1, ARC_SAMPLES)
        return np.column_stack([c[0] + r * np.cos(a), c[1] + r * np.sin(a)])

    if kind == "Ellipse3D":
        c = pt("center_point")
        major = to_2d(curve["major_axis"], origin, xa, ya)
        major = major / (np.linalg.norm(major) or 1.0)
        R, r = float(curve["major_axis_radius"]), float(curve["minor_axis_radius"])
        a = np.linspace(0, 2 * np.pi, ARC_SAMPLES * 2, endpoint=False)
        minor = np.array([-major[1], major[0]])
        return c + np.outer(R * np.cos(a), major) + np.outer(r * np.sin(a), minor)

    # NurbsCurve3D and anything else: the control polygon is a good enough
    # approximation for an area, and far better than dropping the loop.
    if curve.get("control_points"):
        return np.vstack([to_2d(p, origin, xa, ya) for p in curve["control_points"]])
    if "start_point" in curve and "end_point" in curve:
        return np.vstack([pt("start_point"), pt("end_point")])
    return np.empty((0, 2))


def loop_polygon(loop, frame) -> Polygon | None:
    pts = [sample_curve(c, frame) for c in loop.get("profile_curves", [])]
    pts = [p for p in pts if len(p)]
    if not pts:
        return None
    ring = np.vstack(pts)
    if len(ring) < 3:
        return None
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (not poly.is_empty and poly.area > 0) else None


def profile_polygon(profile, frame):
    """A profile's loops -> (polygon, n_loops, has_inner_loop).

    The largest loop is the outer boundary; anything strictly inside it is a
    hole. Loops that merely overlap are unioned rather than subtracted, because
    a malformed profile should degrade to something measurable.
    """
    polys = [p for p in (loop_polygon(lp, frame) for lp in profile.get("loops", []))
             if p is not None]
    if not polys:
        return None, 0, False
    polys.sort(key=lambda p: -p.area)
    outer, holes = polys[0], []
    for p in polys[1:]:
        if outer.contains(p.buffer(-1e-9)):
            holes.append(p)
    shape = outer
    for h in holes:
        shape = shape.difference(h)
    if not shape.is_valid:
        shape = shape.buffer(0)
    return (shape if not shape.is_empty else None), len(polys), bool(holes)


def is_circular(profile) -> bool:
    curves = [c for lp in profile.get("loops", []) for c in lp.get("profile_curves", [])]
    return bool(curves) and all(c.get("type") in ("Circle3D", "Arc3D") for c in curves)


# --------------------------------------------------------------------------- #
# one design
# --------------------------------------------------------------------------- #

def operation_of(feature) -> str | None:
    for k in OP_KEYS:
        if k in feature:
            return feature[k]
    return None


def rows_for_design(path: str, diag: Counter) -> list[dict]:
    with open(path) as fh:
        doc = json.load(fh)
    ents = doc["entities"]
    design = os.path.basename(path)[:-5]

    prior: list[tuple[Polygon, np.ndarray]] = []       # (polygon, plane normal)
    out: list[dict] = []

    steps = [ents[t["entity"]] for t in doc.get("timeline", [])
             if ents.get(t["entity"], {}).get("type") == "ExtrudeFeature"]

    for index, feat in enumerate(steps):
        op = operation_of(feat)
        if op is None:
            diag["no_operation_field"] += 1
            diag[f"keys:{','.join(sorted(feat)[:12])}"] += 1
            continue
        label = OP_TO_LABEL.get(op)
        if label is None:
            diag[f"skipped_op:{op}"] += 1
            continue

        shapes = []
        for ref in feat.get("profiles", []):
            sk = ents.get(ref.get("sketch"))
            if sk is None:
                continue
            frame = sketch_frame(sk)
            prof = sk.get("profiles", {}).get(ref.get("profile"))
            if prof is None:
                continue
            poly, n_loops, has_inner = profile_polygon(prof, frame)
            if poly is not None:
                shapes.append((poly, frame[3], n_loops, has_inner, is_circular(prof)))

        if not shapes:
            diag["no_readable_profile"] += 1
            continue

        poly, normal, n_loops, has_inner, circular = max(shapes, key=lambda s: s[0].area)

        # --- features, using only what existed BEFORE this step -------------- #
        coplanar = [(p, n) for p, n in prior if abs(abs(float(n @ normal)) - 1) < 1e-6]
        if coplanar:
            covered = unary_union([p for p, _ in coplanar])
            overlap = poly.intersection(covered).area / poly.area
            inside = max(poly.intersection(p).area for p, _ in coplanar) / poly.area
            near = min(coplanar, key=lambda pn: poly.centroid.distance(pn[0].centroid))
            scale = float(np.sqrt(near[0].area)) or 1.0
            dist = poly.centroid.distance(near[0].centroid) / scale
            biggest = max(p.area for p, _ in coplanar)
        else:
            overlap = inside = 0.0
            dist = -1.0
            biggest = max((p.area for p, _ in prior), default=0.0)

        out.append(dict(
            design=design,
            project=design.split("_")[0],
            step_index=index,
            label=label,
            operation=op,
            n_loops=n_loops,
            has_inner_loop=int(has_inner),
            is_circular=int(circular),
            coplanar_prior=int(bool(coplanar)),
            area_mm2=round(poly.area, 4),
            area_ratio=round(poly.area / biggest, 4) if biggest else -1.0,
            overlap_frac=round(float(overlap), 4),
            inside_frac=round(float(inside), 4),
            centroid_dist=round(float(dist), 4),
        ))
        prior.append((poly, normal))

    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True,
                    help="the reconstruction folder full of *.json")
    ap.add_argument("--out", default="data/addcut.csv")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N designs, for a quick look")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.data), "*.json")))
    if not files:
        print(f"No .json files under {args.data}")
        return 1
    if args.limit:
        files = files[:args.limit]

    diag: Counter = Counter()
    rows: list[dict] = []
    for i, f in enumerate(files, 1):
        try:
            rows.extend(rows_for_design(f, diag))
        except Exception as exc:                     # noqa: BLE001 - one bad file
            diag[f"failed:{type(exc).__name__}"] += 1
        if i % 500 == 0:
            print(f"  {i}/{len(files)} designs, {len(rows)} steps so far", flush=True)

    if not rows:
        print("NOTHING EXTRACTED. Diagnostics:")
        for k, v in diag.most_common(20):
            print(f"  {k}: {v}")
        print("\nIf you see 'no_operation_field' above, the operation is stored "
              "under a different key in your copy. The 'keys:' line lists what "
              "an extrude actually has - add the right one to OP_KEYS.")
        return 1

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    labels = Counter(r["label"] for r in rows)
    later = [r for r in rows if r["step_index"] > 0]
    majority = max(Counter(r["label"] for r in later).values()) / len(later) if later else 0
    rule = sum((r["overlap_frac"] >= 0.9) == (r["label"] == "cut") for r in later)

    print(f"\n{len(rows)} steps from {len({r['design'] for r in rows})} designs "
          f"({len({r['project'] for r in rows})} projects) -> {args.out}")
    print(f"labels: {dict(labels)}")
    print(f"steps after the first: {len(later)}")
    print(f"  majority-class baseline      {majority:.3f}")
    print(f"  geometric rule (overlap>=.9) {rule / len(later):.3f}" if later else "")
    if diag:
        print("notes:", dict(diag.most_common(8)))
    print("\nNext:  python scripts/train_addcut.py --data " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
