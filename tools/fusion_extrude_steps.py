"""Training data for ML 2 ("smart suggestions"): one row per extrude step in Fusion 360.

For every ExtrudeFeature in every design of Fusion 360 Gallery Reconstruction r1.0.1:
  - the sketch profile it extruded (the shape the user would draw),
  - the part as it was BEFORE this step, seen from the same sketch plane (context),
  - what the designer did: new body / add (join) / cut / intersect, and how far (thickness).

Output (data/extrude_steps/, gitignored):
  steps.csv            one row per step: design, step index, operation, extent, thickness_cm,
                       profile_size_cm, thickness_rel (= thickness / profile size), overlap with
                       the prior part, project split
  images_<chunk>.npz   imgs uint8 (N, 3, 128, 128) and step_id (N,)
                         channel 0: profile, framed together with the context
                         channel 1: prior part silhouette in the sketch plane (empty for step 0)
                         channel 2: profile alone, centred and scaled to fit (shape only)

Usage (dataset at ../r1.0.1, run in chunks because it is slow-ish):
  python tools/fusion_extrude_steps.py --start 0 --stop 1500     ... up to 9000
  python tools/fusion_extrude_steps.py --merge
"""
from __future__ import annotations

import argparse, csv, glob, hashlib, json, math, os
import numpy as np
import cv2

S = 128
SRC_DEFAULT = "../r1.0.1"
OUT_DEFAULT = "data/extrude_steps"
OPS = {"NewBodyFeatureOperation": "new", "JoinFeatureOperation": "add",
       "CutFeatureOperation": "cut", "IntersectFeatureOperation": "intersect",
       "NewComponentFeatureOperation": "new"}


# ------------------------------------------------------------------ sketch curves -> 2-D points
def P(p): return np.array([p["x"], p["y"]], float)

def bspline(c, n=48):
    cp = np.array([[q["x"], q["y"]] for q in c["control_points"]]); k = c["degree"]
    t = np.array(c["knots"], float); w = np.array(c.get("weights") or [1.0] * len(cp), float)
    out = []
    for u in np.linspace(t[k], t[-k - 1], n):
        N = np.array([1.0 if (t[i] <= u < t[i + 1]) or (u == t[-k - 1] and t[i] < u <= t[i + 1]) else 0.0 for i in range(len(t) - 1)])
        for d in range(1, k + 1):
            N2 = np.zeros(len(t) - 1 - d)
            for i in range(len(N2)):
                a = (u - t[i]) / (t[i + d] - t[i]) * N[i] if t[i + d] != t[i] else 0
                b = (t[i + d + 1] - u) / (t[i + d + 1] - t[i + 1]) * N[i + 1] if t[i + d + 1] != t[i + 1] else 0
                N2[i] = a + b
            N = N2
        N = N[:len(cp)] * w; s = N.sum()
        out.append((N[:, None] * cp).sum(0) / s if s > 0 else cp[-1])
    return np.array(out)

def sample(c):
    t = c["type"]
    if t == "Line3D": return np.array([P(c["start_point"]), P(c["end_point"])])
    if t == "Circle3D":
        a = np.linspace(0, 2 * math.pi, 97); return P(c["center_point"]) + c["radius"] * np.stack([np.cos(a), np.sin(a)], 1)
    if t == "Arc3D":
        rv = c["reference_vector"]; a0 = math.atan2(rv["y"], rv["x"]); sg = 1 if c["normal"]["z"] >= 0 else -1
        a = a0 + sg * np.linspace(c["start_angle"], c["end_angle"], max(8, int(abs(c["end_angle"] - c["start_angle"]) / 0.1)))
        return P(c["center_point"]) + c["radius"] * np.stack([np.cos(a), np.sin(a)], 1)
    if t == "Ellipse3D":
        m = c["major_axis"]; g = math.atan2(m["y"], m["x"]); a = np.linspace(0, 2 * math.pi, 97)
        x = c["major_axis_radius"] * np.cos(a); y = c["minor_axis_radius"] * np.sin(a)
        return P(c["center_point"]) + np.stack([x * math.cos(g) - y * math.sin(g), x * math.sin(g) + y * math.cos(g)], 1)
    if t == "NurbsCurve3D": return bspline(c)
    raise ValueError(t)

def chain(curves):
    segs = [sample(c) for c in curves]
    if len(segs) == 1: return segs[0]
    pts = list(segs[0])
    if min(np.linalg.norm(segs[0][0] - segs[1][0]), np.linalg.norm(segs[0][0] - segs[1][-1])) < 1e-6: pts = pts[::-1]
    for s in segs[1:]:
        if np.linalg.norm(s[-1] - pts[-1]) < np.linalg.norm(s[0] - pts[-1]): s = s[::-1]
        pts.extend(s[1:])
    return np.array(pts)


# ------------------------------------------------------------------ geometry helpers
def read_obj(path):
    V, F = [], []
    for line in open(path):
        if line.startswith("v "): V.append([float(x) for x in line.split()[1:4]])
        elif line.startswith("f "): F.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    return np.array(V, float), np.array(F, int)

def to_sketch(V, tf):
    o = np.array([tf["origin"][k] for k in "xyz"]); xa = np.array([tf["x_axis"][k] for k in "xyz"])
    ya = np.array([tf["y_axis"][k] for k in "xyz"]); d = V - o
    return np.stack([d @ xa, d @ ya], 1)

def raster(loops_or_tris, lo, scale, kind):
    img = np.zeros((S * 4, S * 4), np.uint8)
    def px(p): q = (p - lo) * scale * 4; return np.stack([q[:, 0], S * 4 - q[:, 1]], 1).round().astype(np.int32)
    if kind == "tris":
        # one call per triangle: a single fillPoly over many triangles uses even-odd filling,
        # so overlapping triangles would cancel out and leave stripes
        for t in loops_or_tris:
            q = px(t)
            if abs((q[1, 0] - q[0, 0]) * (q[2, 1] - q[0, 1]) - (q[2, 0] - q[0, 0]) * (q[1, 1] - q[0, 1])) > 0:
                cv2.fillConvexPoly(img, q, 255)
    else:
        for outer, pts in loops_or_tris: cv2.fillPoly(img, [px(pts)], 255 if outer else 0)
    return cv2.resize(img, (S, S), interpolation=cv2.INTER_AREA)

def split_for(project, official):
    s = official.get(project, "train")
    if s == "train" and int(hashlib.md5(project.encode()).hexdigest(), 16) % 100 < 15: s = "val"
    return s


# ------------------------------------------------------------------ main extraction
def extract(a):
    files = sorted(glob.glob(os.path.join(a.src, "reconstruction", "*.json")))[a.start:a.stop]
    tt = json.load(open(os.path.join(a.src, "train_test.json"))); votes = {}
    for sp in ("train", "test"):
        for did in tt[sp]: votes.setdefault(did.rsplit("_", 1)[0], []).append(sp)
    official = {p: max(set(v), key=v.count) for p, v in votes.items()}
    rows, imgs, skipped = [], [], {}
    def skip(why): skipped[why] = skipped.get(why, 0) + 1
    for f in files:
        did = os.path.basename(f)[:-5]
        try: d = json.load(open(f))
        except Exception: skip("bad_json"); continue
        E = d["entities"]; proj = d["metadata"]["parent_project"]
        steps = [s for s in d["sequence"] if s["type"] == "ExtrudeFeature"]
        prev_obj = None
        for k, s in enumerate(steps):
            e = E[s["entity"]]
            try:
                sks = {p["sketch"] for p in e["profiles"]}
                if len(sks) != 1: skip("multi_sketch"); raise StopIteration
                sk = E[next(iter(sks))]
                loops = []
                for p in e["profiles"]:
                    for l in sorted(sk["profiles"][p["profile"]]["loops"], key=lambda l: not l["is_outer"]):
                        loops.append((l["is_outer"], chain(l["profile_curves"])))
                prof_pts = np.vstack([p for _, p in loops])
                plo, phi = prof_pts.min(0), prof_pts.max(0); psize = float((phi - plo).max())
                if psize <= 0: skip("degenerate"); raise StopIteration
                # context: the part before this step, projected onto this sketch plane
                tris = np.zeros((0, 3, 2))
                if prev_obj and os.path.exists(os.path.join(a.src, "reconstruction", prev_obj)):
                    V, F = read_obj(os.path.join(a.src, "reconstruction", prev_obj))
                    if len(F): tris = to_sketch(V, sk["transform"])[F]
                allp = np.vstack([prof_pts, tris.reshape(-1, 2)]) if len(tris) else prof_pts
                lo, hi = allp.min(0), allp.max(0); span = float((hi - lo).max())
                lo = (lo + hi) / 2 - span * 0.55; scale = S / (span * 1.1)
                ch_prof = raster(loops, lo, scale, "loops")
                ch_ctx = raster(tris, lo, scale, "tris") if len(tris) else np.zeros((S, S), np.uint8)
                c2 = (plo + phi) / 2 - psize * 0.55; ch_shape = raster(loops, c2, S / (psize * 1.1), "loops")
                pa = (ch_prof > 127).sum(); ov = float(((ch_prof > 127) & (ch_ctx > 127)).sum() / pa) if pa else 0.0
                d1 = abs(e["extent_one"]["distance"]["value"])
                d2 = abs(e["extent_two"]["distance"]["value"]) if e.get("extent_two") and e["extent_type"] == "TwoSidesFeatureExtentType" else 0.0
                thick = d1 + d2
                sid = f"{did}_s{k:02d}"
                rows.append(dict(step_id=sid, design_id=did, project=proj, split=split_for(proj, official),
                    step=k, n_steps=len(steps), op=OPS.get(e["operation"], e["operation"]),
                    extent_type=e["extent_type"].replace("FeatureExtentType", ""), d1_cm=round(d1, 5), d2_cm=round(d2, 5),
                    thickness_cm=round(thick, 5), profile_size_cm=round(psize, 5),
                    thickness_rel=round(thick / psize, 5), n_profiles=len(e["profiles"]), n_loops=len(loops),
                    has_prior_part=int(len(tris) > 0), overlap_prior=round(ov, 4)))
                imgs.append(np.stack([ch_prof, ch_ctx, ch_shape]))
            except StopIteration:
                pass
            except Exception as ex:
                skip("error:" + type(ex).__name__)
            prev_obj = s.get("obj", prev_obj)
    os.makedirs(os.path.join(a.out, "parts"), exist_ok=True)
    stem = os.path.join(a.out, "parts", f"part_{a.start:05d}")
    np.savez_compressed(stem + ".npz", imgs=np.array(imgs, np.uint8).reshape(-1, 3, S, S), step_id=np.array([r["step_id"] for r in rows]))
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["step_id"]); w.writeheader(); w.writerows(rows)
    json.dump(skipped, open(stem + "_skipped.json", "w"))
    print(f"designs {len(files)} -> steps {len(rows)}  skipped {skipped}")

def merge(a):
    rows = []
    for c in sorted(glob.glob(os.path.join(a.out, "parts", "part_*.csv"))): rows += list(csv.DictReader(open(c)))
    with open(os.path.join(a.out, "steps.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    from collections import Counter
    th = np.array([float(r["thickness_rel"]) for r in rows])
    summ = dict(steps=len(rows), designs=len({r["design_id"] for r in rows}),
                ops=Counter(r["op"] for r in rows), ops_after_step0=Counter(r["op"] for r in rows if r["step"] != "0"),
                split=Counter(r["split"] for r in rows), extent=Counter(r["extent_type"] for r in rows),
                thickness_rel_p10_p50_p90=[round(float(np.percentile(th, q)), 3) for q in (10, 50, 90)])
    json.dump(summ, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=dict); print(json.dumps(summ, indent=1, default=dict))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT); ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=10 ** 9)
    ap.add_argument("--merge", action="store_true"); a = ap.parse_args()
    merge(a) if a.merge else extract(a)
