"""Hand-designed features for the classical ML arm.

OWNER: Serena.

Twelve numbers per stroke. Why features rather than raw points:

  * They work on a few thousand samples. A 1-D CNN on (64, 2) inputs needs far
    more data before it beats this.
  * They are explainable. When the model confuses a circle for a rectangle you
    can look at `fill_ratio` and see why, which is worth a great deal in a
    write-up.
  * They reuse HW2 Problem 1 almost exactly - a table of numbers with a label,
    fed to a tabular model.

FEATURE_NAMES is the column order and it is part of the contract between
training and inference. Append new features at the END, never in the middle, or
every model you have already trained silently starts reading the wrong columns.
"""
from __future__ import annotations

import numpy as np

from ..strokes import arc_length, normalize, resample
from .rules import fit_circle, fit_line

FEATURE_NAMES = [
    "closure_ratio",        # gap between endpoints / total length. ~0 = closed
    "aspect_ratio",         # bbox longest / shortest side
    "fill_ratio",           # enclosed area / bbox area. circle ~0.79, rect ~1.0
    "perimeter_ratio",      # path length / bbox perimeter
    "circle_residual",      # mean radial deviation from the best-fit circle
    "line_residual",        # mean deviation from the best-fit line
    "curvature_mean",       # mean absolute turning per unit length
    "curvature_std",        # low for a circle (constant turn), high for a rect
    "total_turning",        # sum of turning angles / 2pi. ~1 for any closed loop
    "corner_score",         # fraction of points whose turning exceeds a threshold
    "radial_std",           # std of distance-to-centroid / mean. 0 for a circle
    "straight_frac",        # fraction of length in near-straight runs
]


def _turning(pts: np.ndarray) -> np.ndarray:
    """Signed turning angle at each interior point, radians."""
    d = np.diff(pts, axis=0)
    a = np.arctan2(d[:, 1], d[:, 0])
    t = np.diff(a)
    return (t + np.pi) % (2 * np.pi) - np.pi        # wrap to [-pi, pi]


def _polygon_area(pts: np.ndarray) -> float:
    """Shoelace area of the closed version of a path."""
    p = pts if np.allclose(pts[0], pts[-1]) else np.vstack([pts, pts[:1]])
    x, y = p[:, 0], p[:, 1]
    return float(abs(0.5 * np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def extract(points_mm: np.ndarray) -> dict[str, float]:
    """One stroke (already in mm) -> the feature dict. Never raises."""
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return {k: 0.0 for k in FEATURE_NAMES}

    pts = resample(pts, 64)
    norm, _ = normalize(pts)                 # scale-invariant: shape, not size

    s = arc_length(norm)
    total = float(s[-1]) or 1.0
    gap = float(np.linalg.norm(norm[0] - norm[-1]))

    lo, hi = norm.min(axis=0), norm.max(axis=0)
    w, h = float(hi[0] - lo[0]), float(hi[1] - lo[1])
    w, h = max(w, 1e-9), max(h, 1e-9)

    area = _polygon_area(norm)
    turn = _turning(norm)
    abs_turn = np.abs(turn)

    centroid = norm.mean(axis=0)
    radii = np.linalg.norm(norm - centroid, axis=1)
    mean_r = float(radii.mean()) or 1e-9

    _, _, _, circ_res = fit_circle(norm)
    _, _, line_res = fit_line(norm)

    seg = np.linalg.norm(np.diff(norm, axis=0), axis=1)
    straight = seg[1:][abs_turn < np.deg2rad(5)].sum() if len(seg) > 1 else 0.0

    return {
        "closure_ratio":   gap / total,
        "aspect_ratio":    max(w, h) / min(w, h),
        "fill_ratio":      area / (w * h),
        "perimeter_ratio": total / (2 * (w + h)),
        "circle_residual": float(circ_res),
        "line_residual":   float(line_res),
        "curvature_mean":  float(abs_turn.mean()),
        "curvature_std":   float(abs_turn.std()),
        "total_turning":   float(abs(turn.sum()) / (2 * np.pi)),
        "corner_score":    float((abs_turn > np.deg2rad(30)).mean()),
        "radial_std":      float(radii.std() / mean_r),
        "straight_frac":   float(straight / total),
    }


def extract_matrix(strokes: list[np.ndarray]) -> np.ndarray:
    """Many strokes -> (N, 12) in FEATURE_NAMES order."""
    return np.array([[extract(s)[k] for k in FEATURE_NAMES] for s in strokes], dtype=np.float64)
