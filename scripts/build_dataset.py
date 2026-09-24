#!/usr/bin/env python3
"""Fusion 360 Gallery CAD files -> a labelled synthetic hand-drawn stroke dataset.

    python scripts/build_dataset.py \
        --data ../r1.0.1/reconstruction \
        --split ../r1.0.1/train_test.json \
        --out data/strokes \
        --per-class 4000

Run `scripts/inspect_dataset.py` FIRST. It checks the parser against your actual
files and tells you whether this will work before you wait on it.

WHAT COMES OUT
--------------
    data/strokes/train.npz   strokes (N, 64, 2) mm, labels (N,), model_ids (N,)
    data/strokes/test.npz    same, from the official held-out models
    data/strokes/meta.json   counts, style, seed, provenance

WHY THE SPLIT COMES FROM train_test.json
----------------------------------------
Curves from one CAD model are highly self-similar - the same part's four fillet
arcs have nearly identical radii. Splitting curves randomly puts siblings on
both sides and inflates the score. The official split is by MODEL, which is the
correct grouping. Same lesson as HW2's parent_id, wearing a CAD hat.

LICENCE: the output is a "Modified Set" of the Fusion 360 Gallery Dataset. Keep
it local. Do not push it to a public HF dataset. See docs/dataset.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sketch2stl.config import RESAMPLE_N                                   # noqa: E402
from sketch2stl.data.fusion360 import (curves_from_model, iter_models,     # noqa: E402
                                       load_split, operations_from_model)
from sketch2stl.data.synth import (STYLES, deviation, handdraw,  # noqa: E402
                                   to_canvas_scale)              # noqa: E402
from sketch2stl.strokes import prepare_points, resample                    # noqa: E402
from sketch2stl.types import PrimitiveKind                                 # noqa: E402

# Only these three come straight from the CAD file. RECT and POLYLINE are
# constructed below, because Fusion stores a rectangle as four separate lines
# and has no concept of "a messy scribble" at all.
CAD_CLASSES = [PrimitiveKind.LINE, PrimitiveKind.ARC, PrimitiveKind.CIRCLE]


def _rect_from_lines(curves, rng) -> np.ndarray | None:
    """Build a RECT stroke by walking four connected lines from one sketch.

    A user drawing a rectangle in one stroke produces a single closed path, so
    that is what the training example has to be. Fusion stores the same shape as
    four independent Line3D entries, so we stitch them back together.
    """
    lines = [c for c in curves if c.kind is PrimitiveKind.LINE]
    if len(lines) < 4:
        return None
    for _ in range(6):
        seed = lines[rng.integers(len(lines))]
        ring = [seed.points[0], seed.points[1]]
        used = {id(seed)}
        for _ in range(3):
            best, best_d = None, 2.0
            for c in lines:
                if id(c) in used:
                    continue
                for a, b in ((c.points[0], c.points[-1]), (c.points[-1], c.points[0])):
                    d = float(np.linalg.norm(a - ring[-1]))
                    if d < best_d:
                        best, best_d = (c, b), d
            if best is None:
                break
            used.add(id(best[0]))
            ring.append(best[1])
        if len(ring) == 5 and np.linalg.norm(ring[-1] - ring[0]) < 2.0:
            ring[-1] = ring[0]
            return np.asarray(ring, dtype=np.float64)
    return None


def _polyline_from_curves(curves, rng) -> np.ndarray | None:
    """A deliberately ambiguous closed blob: the 'none of the above' class.

    Without it the classifier has no way to say "I do not know", and every
    scribble gets forced into being a circle. Built by perturbing a real closed
    loop until it is no longer any single primitive.
    """
    closed = [c for c in curves if c.kind is PrimitiveKind.CIRCLE]
    if not closed:
        return None
    base = closed[int(rng.integers(len(closed)))].points
    pts = resample(base, 48)
    centre = pts.mean(axis=0)
    radii = np.linalg.norm(pts - centre, axis=1)
    lobes = rng.integers(3, 6)
    phase = rng.uniform(0, 2 * np.pi)
    t = np.linspace(0, 2 * np.pi, len(pts), endpoint=False)
    scale = 1.0 + rng.uniform(0.18, 0.45) * np.sin(lobes * t + phase)
    out = centre + (pts - centre) / np.maximum(radii, 1e-9)[:, None] * (radii * scale)[:, None]
    return np.vstack([out, out[:1]])


def _draw(clean, style, rng, rescale=True, closed=None):
    """Rescale to canvas size, then distort. Returns (stroke, clean_at_canvas_scale).

    The deviation must be measured against the RESCALED curve, not the original -
    otherwise it reports the rescaling, not the hand.
    """
    base = to_canvas_scale(clean, rng) if rescale else np.asarray(clean, dtype=np.float64)
    return handdraw(base, style, rng, closed=closed), base


def collect(root, model_ids, per_class, style, rng, max_models=None, rescale=True):
    """Walk the CAD models and emit synthetic strokes, balanced across classes."""
    quota = Counter()
    strokes, labels, owners, devs = [], [], [], []
    ops = Counter()
    n_models = 0
    t0 = time.time()

    for model_id, data in iter_models(root, model_ids, limit=max_models):
        n_models += 1
        ops.update(operations_from_model(data))
        curves = curves_from_model(data, model_id)
        if not curves:
            continue

        # the three classes that exist verbatim in the CAD file
        for c in curves:
            if c.kind not in CAD_CLASSES or quota[c.kind] >= per_class:
                continue
            stroke, base = _draw(c.points, style, rng, rescale)
            strokes.append(prepare_points(stroke))
            labels.append(c.kind.value)
            owners.append(model_id)
            devs.append(deviation(stroke, base))
            quota[c.kind] += 1

        # the two we construct
        if quota[PrimitiveKind.RECT] < per_class:
            ring = _rect_from_lines(curves, rng)
            if ring is not None:
                stroke, base = _draw(ring, style, rng, rescale, closed=True)
                strokes.append(prepare_points(stroke))
                labels.append(PrimitiveKind.RECT.value)
                owners.append(model_id)
                devs.append(deviation(stroke, base))
                quota[PrimitiveKind.RECT] += 1

        if quota[PrimitiveKind.POLYLINE] < per_class:
            blob = _polyline_from_curves(curves, rng)
            if blob is not None:
                stroke, base = _draw(blob, style, rng, rescale, closed=True)
                strokes.append(prepare_points(stroke))
                labels.append(PrimitiveKind.POLYLINE.value)
                owners.append(model_id)
                devs.append(deviation(stroke, base))
                quota[PrimitiveKind.POLYLINE] += 1

        if n_models % 250 == 0:
            done = sum(quota.values())
            print(f"  {n_models:5d} models, {done:6d} strokes "
                  f"({time.time() - t0:.0f}s)  {dict(quota)}", flush=True)

        if all(quota[k] >= per_class for k in
               CAD_CLASSES + [PrimitiveKind.RECT, PrimitiveKind.POLYLINE]):
            break

    return (np.asarray(strokes, dtype=np.float32), np.asarray(labels),
            np.asarray(owners), np.asarray(devs, dtype=np.float32), n_models, ops)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="r1.0.1/reconstruction directory")
    ap.add_argument("--split", default=None, help="train_test.json (strongly recommended)")
    ap.add_argument("--out", default="data/strokes")
    ap.add_argument("--per-class", type=int, default=4000, help="strokes per class, train")
    ap.add_argument("--per-class-test", type=int, default=800)
    ap.add_argument("--style", default="typical", choices=list(STYLES))
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--max-models", type=int, default=None, help="cap, for a quick trial run")
    ap.add_argument("--no-rescale", action="store_true",
                    help="skip canvas rescaling - an ablation, not a normal run. "
                         "Distortion is then applied at each part's native size, so a "
                         "2 m beam gets near-zero relative noise and a 3 mm pin gets 9%. "
                         "Useful to quantify how much that artefact was worth.")
    args = ap.parse_args()

    style = STYLES[args.style]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.split:
        split = load_split(args.split)
        print(f"official split: {len(split['train'])} train models, "
              f"{len(split['test'])} test models")
    else:
        print("WARNING: no --split given. Curves from one model will land on both sides "
              "of the split and your test score will be inflated. Pass train_test.json.")
        split = {"train": None, "test": None}

    meta = {"style": args.style, "seed": args.seed, "resample_n": RESAMPLE_N,
            "canvas_rescale": not args.no_rescale,
            "source": "Fusion 360 Gallery Dataset (Reconstruction), synthesised strokes",
            "licence": "Autodesk non-commercial research. Do not redistribute publicly."}

    for name, ids, per_class in (("train", split["train"], args.per_class),
                                 ("test", split["test"], args.per_class_test)):
        print(f"\n=== {name} ===")
        rng = np.random.default_rng(args.seed + (0 if name == "train" else 1))
        X, y, owner, dev, n_models, ops = collect(
            args.data, ids, per_class, style, rng, args.max_models,
            rescale=not args.no_rescale)

        if len(X) == 0:
            print("  NOTHING EXTRACTED. Run scripts/inspect_dataset.py to find out why.")
            continue

        np.savez_compressed(out / f"{name}.npz", strokes=X, labels=y, model_ids=owner,
                            deviation_mm=dev)
        counts = dict(Counter(y.tolist()))
        print(f"  {len(X)} strokes from {n_models} models -> {out / (name + '.npz')}")
        print(f"  per class: {counts}")
        print(f"  mean deviation from the clean curve: {dev.mean():.2f} mm "
              f"(sd {dev.std():.2f}, max {dev.max():.2f})")
        if dev.mean() > 1.5:
            print("  !! that is high for a hand. Expect 0.3-0.8 mm. Check --no-rescale "
                  "is not set, or lower HandStyle.tremor_mm in sketch2stl/data/synth.py.")
        meta[name] = {"n": int(len(X)), "n_models": n_models, "per_class": counts,
                      "mean_deviation_mm": float(dev.mean())}
        if name == "train" and ops:
            total = sum(ops.values())
            meta["extrude_operations"] = {k: {"n": v, "pct": round(100 * v / total, 1)}
                                          for k, v in ops.items()}
            print(f"  extrude operations seen in real CAD: "
                  f"{ {k: f'{100 * v / total:.0f}%' for k, v in ops.items()} }")

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {out / 'meta.json'}")
    print("\nNext:  python scripts/train_recognizer.py --data", out)


if __name__ == "__main__":
    main()
