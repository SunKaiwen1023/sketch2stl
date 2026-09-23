"""Load stroke datasets (synthetic Fusion strokes and hand-drawn sessions) into one format,
and rasterise strokes into small images for image models (the fine-tuned ResNet arm).

Both datasets end up as: list of (N,2) float arrays in CANVAS PIXELS + a list of label rows.

    from tools.strokes_io import load_npz_dataset, load_hand_sessions, rasterize
    strokes, rows = load_npz_dataset("data/strokes/synthetic")        # fusion_strokes.npz + labels.csv
    strokes, rows = load_hand_sessions("data/strokes/raw")            # JSON files from stroke_collector.html
    img = rasterize(strokes[0], size=64)                              # (64, 64) uint8, white stroke on black
"""
from __future__ import annotations

import csv, glob, json, os
import numpy as np

CLASSES = ["line", "arc", "circle", "rect", "polyline"]      # same order as recognizer/ml.py CLASSES


def load_npz_dataset(folder: str):
    z = np.load(os.path.join(folder, "fusion_strokes.npz"))
    P, O = z["points"], z["offsets"]
    rows = list(csv.DictReader(open(os.path.join(folder, "labels.csv"))))
    return [P[O[i]:O[i + 1]].astype(np.float64) for i in range(len(rows))], rows


def load_hand_sessions(folder: str):
    """Read every session JSON written by tools/stroke_collector.html."""
    strokes, rows = [], []
    for f in sorted(glob.glob(os.path.join(folder, "*.json"))):
        s = json.load(open(f))
        for st in s["strokes"]:
            pts = np.asarray(st["points"], dtype=np.float64).reshape(-1, 2)
            if len(pts) < 4:
                continue
            strokes.append(pts)
            rows.append(dict(stroke_id=st["stroke_id"], label=st["label"], session_id=s["session_id"],
                             author=s["author"], date=s["date"], device=s["device"], split="",
                             source_design="", closed=""))
    return strokes, rows


def rasterize(points_px: np.ndarray, size: int = 64, thickness: int = 2) -> np.ndarray:
    """Centre, scale to fit (keeping aspect ratio, so a line stays thin), draw as a polyline."""
    import cv2
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    img = np.zeros((size, size), np.uint8)
    if len(pts) < 2:
        return img
    lo, hi = pts.min(0), pts.max(0)
    scale = (size - 8) / max((hi - lo).max(), 1e-6)
    q = (pts - (lo + hi) / 2) * scale + size / 2
    cv2.polylines(img, [np.round(q).astype(np.int32)], False, 255, thickness, cv2.LINE_AA)
    return img
