"""Synthetic training strokes from the Fusion 360 Gallery Reconstruction dataset.

WHY: the ML arm needs far more than the ~500 strokes we can draw by hand. Every
sketch in Fusion 360 already records exactly which primitives a designer drew
(Line3D, Arc3D, Circle3D, closed loops), so the labels are free and correct.
We turn each primitive into a stroke that looks hand-drawn (wobble, jitter,
overshoot at closure, uneven sampling, slower pen at corners) in CANVAS PIXELS,
the same space `types.Stroke` uses.

THIS IS SYNTHETIC / AUGMENTED DATA. Keep it separate from the hand-drawn
dataset, never mix it into a test split, and say so in the dataset card.

Labels follow `types.PrimitiveKind`:
  line     Line3D curves
  arc      Arc3D curves sweeping 30-330 degrees
  circle   Circle3D curves
  rect     closed profile loops with 4 straight sides at right angles (one stroke)
  polyline every other closed profile loop (polygons, slots, L-shapes...) (one stroke)

Grouping: session_id = "fusion_<parent project>", split by project so near-duplicate
designs from one project never land in both train and test.

Usage (the dataset folder sits next to this repo by default):
  python tools/fusion_strokes.py --src ../r1.0.1 --out data/strokes/synthetic --start 0 --stop 3000
  python tools/fusion_strokes.py --src ../r1.0.1 --out data/strokes/synthetic --start 3000 --stop 6000
  python tools/fusion_strokes.py --src ../r1.0.1 --out data/strokes/synthetic --start 6000 --stop 9000
  python tools/fusion_strokes.py --out data/strokes/synthetic --merge

Output after --merge:
  fusion_strokes.npz   points (float32, all strokes concatenated), t (float32),
                       offsets (int64, stroke i = points[offsets[i]:offsets[i+1]])
  labels.csv           stroke_id, label, session_id, author, date, device, split, source_design, closed
"""
from __future__ import annotations

import argparse, csv, glob, hashlib, json, math, os, random
import numpy as np

CANVAS_W, CANVAS_H = 640, 480          # mirrors sketch2stl/config.py (kept standalone on purpose)
KEEP = {"line": 0.10, "arc": 0.6, "circle": 0.35, "rect": 1.0, "polyline": 0.5}
CAP_PER_CLASS = 6000                   # applied at --merge, keeps classes balanced
SEED = 20260923


# ----------------------------------------------------------------------------- geometry
def P(p): return np.array([p["x"], p["y"]], float)

def sample_curve(c, n=64):
    t = c["type"]
    if t == "Line3D":
        return np.array([P(c["start_point"]), P(c["end_point"])])
    if t == "Circle3D":
        a = np.linspace(0, 2 * math.pi, n); o = P(c["center_point"]); r = c["radius"]
        return o + r * np.stack([np.cos(a), np.sin(a)], 1)
    if t == "Arc3D":
        o = P(c["center_point"]); r = c["radius"]; rv = c["reference_vector"]
        a0 = math.atan2(rv["y"], rv["x"]); sgn = 1 if c["normal"]["z"] >= 0 else -1
        a = a0 + sgn * np.linspace(c["start_angle"], c["end_angle"], max(6, int(n * abs(c["end_angle"] - c["start_angle"]) / (2 * math.pi))))
        return o + r * np.stack([np.cos(a), np.sin(a)], 1)
    return None                                   # splines / ellipses: skipped

def chain(curves):
    segs = [sample_curve(c) for c in curves]
    if any(s is None for s in segs): return None
    pts = list(segs[0])
    if len(segs) > 1 and min(np.linalg.norm(segs[0][0] - segs[1][0]), np.linalg.norm(segs[0][0] - segs[1][-1])) < 1e-6:
        pts = pts[::-1]
    for s in segs[1:]:
        if np.linalg.norm(s[-1] - pts[-1]) < np.linalg.norm(s[0] - pts[-1]): s = s[::-1]
        pts.extend(s[1:])
    return np.array(pts)

def is_rect(curves):
    if any(c["type"] != "Line3D" for c in curves): return False
    dirs, offs = [], []
    pts = np.array([P(c["start_point"]) for c in curves] + [P(c["end_point"]) for c in curves])
    span = max(np.ptp(pts, axis=0).max(), 1e-9)
    for c in curves:                               # merge collinear pieces into one edge
        v = P(c["end_point"]) - P(c["start_point"]); n = np.linalg.norm(v)
        if n <= 0: return False
        v = v / n; v = v if (v[0] > 1e-9 or (abs(v[0]) <= 1e-9 and v[1] > 0)) else -v
        off = v[0] * c["start_point"]["y"] - v[1] * c["start_point"]["x"]
        if not any(abs(v[0] * d[1] - v[1] * d[0]) < 0.01 and abs(off - o) < 0.01 * span for d, o in zip(dirs, offs)):
            dirs.append(v); offs.append(off)
    if len(dirs) != 4: return False
    cross = [abs(dirs[i][0] * dirs[j][1] - dirs[i][1] * dirs[j][0]) for i in range(4) for j in range(i + 1, 4)]
    return all(c < 0.02 or c > 0.98 for c in cross) and sum(c < 0.02 for c in cross) == 2


# ----------------------------------------------------------------------------- "hand-drawn" noise
def resample_by_length(pts, step):
    d = np.sqrt((np.diff(pts, axis=0) ** 2).sum(1)); s = np.concatenate([[0], np.cumsum(d)])
    if s[-1] <= 0: return pts
    q = np.arange(0, s[-1], step); q = np.append(q, s[-1])
    return np.stack([np.interp(q, s, pts[:, 0]), np.interp(q, s, pts[:, 1])], 1)

def handify(shape, closed, rng):
    """shape: (N,2) in arbitrary units -> hand-drawn-looking stroke in canvas px + timestamps."""
    # 1. place on the canvas at a random size / position / small rotation
    shape = shape - shape.mean(0)
    ext = np.ptp(shape, axis=0).max()
    if ext <= 0: return None
    target = rng.uniform(50, 380)
    ang = rng.normal(0, math.radians(4)) if rng.random() < 0.85 else rng.uniform(-math.pi, math.pi)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    pts = (shape / ext * target) @ R.T
    pts[:, 1] *= -1                                # model y-up -> canvas y-down
    lo, hi = pts.min(0), pts.max(0)
    if (hi - lo)[0] > CANVAS_W - 20 or (hi - lo)[1] > CANVAS_H - 20:
        pts *= min((CANVAS_W - 20) / max((hi - lo)[0], 1), (CANVAS_H - 20) / max((hi - lo)[1], 1)); lo, hi = pts.min(0), pts.max(0)
    def _u(a, b): return rng.uniform(a, b) if b > a else (a + b) / 2
    pts += np.array([_u(10 - lo[0], CANVAS_W - 10 - hi[0]), _u(10 - lo[1], CANVAS_H - 10 - hi[1])])

    # 2. dense even resample, start anywhere on a closed loop, random direction
    dense = resample_by_length(pts, 1.0)
    if closed:
        dense = dense[:-1]; k = rng.integers(len(dense)); dense = np.roll(dense, -k, axis=0)
        # overshoot or leave a gap at closure (-4% .. +6% of the perimeter)
        extra = int(len(dense) * rng.uniform(-0.04, 0.06))
        dense = np.vstack([dense, dense[:extra]]) if extra > 0 else dense[:len(dense) + extra]
    else:
        e = np.ptp(dense, axis=0).max() * rng.uniform(-0.03, 0.03)   # endpoints a little short/long
        if len(dense) > 2 and e != 0:
            d0 = dense[0] - dense[1]; d1 = dense[-1] - dense[-2]
            dense[0] += d0 / (np.linalg.norm(d0) + 1e-9) * e; dense[-1] += d1 / (np.linalg.norm(d1) + 1e-9) * e
    if rng.random() < 0.5: dense = dense[::-1]
    if len(dense) < 4: return None

    # 3. low-frequency wobble along the normal + small tremor
    tang = np.gradient(dense, axis=0); tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
    nrm = np.stack([-tang[:, 1], tang[:, 0]], 1)
    s = np.linspace(0, 1, len(dense)); size = np.ptp(dense, axis=0).max()
    wob = sum(rng.uniform(0.003, 0.012) * size * np.sin(2 * math.pi * rng.uniform(0.5, 3.5) * s + rng.uniform(0, 6.3)) for _ in range(3))
    dense = dense + nrm * wob[:, None] + rng.normal(0, 0.35, dense.shape)

    # 4. uneven sampling like a mouse / pen: 2-7 px between points, slower at corners
    curv = np.linalg.norm(np.gradient(tang, axis=0), axis=1)
    speed = rng.uniform(250, 700) / (1 + 25 * np.convolve(curv, np.ones(9) / 9, "same"))     # px / s
    d = np.sqrt((np.diff(dense, axis=0) ** 2).sum(1)); tt = np.concatenate([[0], np.cumsum(d / speed[1:])])
    keep = [0]; acc = 0.0; step = rng.uniform(2, 7)
    for i in range(1, len(dense)):
        acc += d[i - 1]
        if acc >= step: keep.append(i); acc = 0.0; step = rng.uniform(2, 7)
    if keep[-1] != len(dense) - 1: keep.append(len(dense) - 1)
    out = np.clip(dense[keep], 0, [CANVAS_W - 1, CANVAS_H - 1])
    return out.astype(np.float32), tt[keep].astype(np.float32)


# ----------------------------------------------------------------------------- extraction
def split_for(project, official):
    s = official.get(project, "train")
    if s == "train" and int(hashlib.md5(project.encode()).hexdigest(), 16) % 100 < 15: s = "val"
    return s

def extract(args):
    rng_keep = random.Random(SEED + args.start); rng = np.random.default_rng(SEED + args.start)
    files = sorted(glob.glob(os.path.join(args.src, "reconstruction", "*.json")))[args.start:args.stop]
    tt = json.load(open(os.path.join(args.src, "train_test.json")))
    votes = {}
    for sp in ("train", "test"):
        for did in tt[sp]: votes.setdefault(did.rsplit("_", 1)[0], []).append(sp)
    official = {p: max(set(v), key=v.count) for p, v in votes.items()}
    pts_all, t_all, rows = [], [], []
    for f in files:
        did = os.path.basename(f)[:-5]
        try: d = json.load(open(f))
        except Exception: continue
        proj = d["metadata"]["parent_project"]
        for e in d["entities"].values():
            if e["type"] != "Sketch": continue
            cands = []
            for cid, c in e.get("curves", {}).items():
                if c.get("construction_geom") or c.get("reference"): continue
                if c["type"] == "SketchLine": kind = "line"
                elif c["type"] == "SketchCircle": kind = "circle"
                elif c["type"] == "SketchArc": kind = "arc"
                else: continue
                cands.append((kind, cid))
            # curve geometry lives in profile loops (sketch space); index it by curve id
            geo, loops = {}, []
            for pr in e.get("profiles", {}).values():
                for l in pr["loops"]:
                    loops.append(l["profile_curves"])
                    for pc in l["profile_curves"]: geo.setdefault(pc["curve"], pc)
            for kind, cid in cands:
                if rng_keep.random() > KEEP[kind]: continue
                g = geo.get(cid)
                if g is None or g["type"] != {"line": "Line3D", "arc": "Arc3D", "circle": "Circle3D"}[kind]: continue
                if kind == "arc":
                    sw = abs(g["end_angle"] - g["start_angle"])
                    if not (math.radians(30) <= sw <= math.radians(330)): continue
                shape = sample_curve(g)
                _add(rows, pts_all, t_all, kind, shape, kind == "circle", rng, did, proj, official)
            seen = set()
            for lp in loops:                                     # whole closed loops -> rect / polyline
                if len(lp) < 3: continue
                key = tuple(sorted(pc["curve"] for pc in lp))
                if key in seen: continue
                seen.add(key)
                kind = "rect" if is_rect(lp) else "polyline"
                if rng_keep.random() > KEEP[kind]: continue
                shape = chain(lp)
                if shape is None: continue
                _add(rows, pts_all, t_all, kind, shape, True, rng, did, proj, official)
    os.makedirs(os.path.join(args.out, "parts"), exist_ok=True)
    _save(os.path.join(args.out, "parts", f"part_{args.start:05d}"), pts_all, t_all, rows)
    print(f"designs {len(files)} -> strokes {len(rows)}", {k: sum(r['label'] == k for r in rows) for k in KEEP})

def _add(rows, pts_all, t_all, kind, shape, closed, rng, did, proj, official):
    r = handify(shape, closed, rng)
    if r is None or len(r[0]) < 4: return
    pts_all.append(r[0]); t_all.append(r[1])
    rows.append(dict(label=kind, session_id=f"fusion_{proj}", author="synthetic-fusion", date="2026-09-23",
                     device="synthetic", split=split_for(proj, official), source_design=did, closed=int(closed)))

def _save(stem, pts_all, t_all, rows):
    offs = np.concatenate([[0], np.cumsum([len(p) for p in pts_all])]).astype(np.int64)
    np.savez_compressed(stem + ".npz", points=np.concatenate(pts_all) if pts_all else np.zeros((0, 2), np.float32),
                        t=np.concatenate(t_all) if t_all else np.zeros(0, np.float32), offsets=offs)
    with open(stem + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["label"]); w.writeheader(); w.writerows(rows)

def merge(args):
    rng = random.Random(SEED)
    strokes, rows = [], []
    for stem in sorted(glob.glob(os.path.join(args.out, "parts", "part_*.npz"))):
        z = np.load(stem); o = z["offsets"]; zp = z["points"]; zt = z["t"]    # load once, not per stroke
        rs = list(csv.DictReader(open(stem[:-4] + ".csv")))
        for i, r in enumerate(rs):
            strokes.append((zp[o[i]:o[i + 1]], zt[o[i]:o[i + 1]])); rows.append(r)
    idx = list(range(len(rows))); rng.shuffle(idx)
    kept, count = [], {}
    for i in idx:
        k = rows[i]["label"]
        if count.get(k, 0) < CAP_PER_CLASS: kept.append(i); count[k] = count.get(k, 0) + 1
    kept.sort()
    out_rows = [dict(stroke_id=f"fz{n:06d}", **rows[i]) for n, i in enumerate(kept)]
    _save(os.path.join(args.out, "fusion_strokes"), [strokes[i][0] for i in kept], [strokes[i][1] for i in kept], out_rows)
    os.replace(os.path.join(args.out, "fusion_strokes.csv"), os.path.join(args.out, "labels.csv"))
    summary = {"strokes": len(out_rows), "per_class": count,
               "per_split": {s: sum(r["split"] == s for r in out_rows) for s in ("train", "val", "test")},
               "sessions": len({r["session_id"] for r in out_rows})}
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="../r1.0.1"); ap.add_argument("--out", default="data/strokes/synthetic")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--stop", type=int, default=10**9)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    merge(a) if a.merge else extract(a)
