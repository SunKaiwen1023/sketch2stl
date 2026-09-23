"""Raw canvas strokes -> clean, fixed-length, millimetre-space point arrays.

OWNER: Serena.

This is the first half of the ML story. Nothing here is learned - it is the
preprocessing every recogniser (rules or ML) sits on top of. Getting it right
matters more than the model: a resampled, normalised stroke makes a rule-based
fit work well and makes a small network trainable on a few hundred examples.
"""
from __future__ import annotations

import numpy as np

from .config import CANVAS_H, MIN_STROKE_PTS, PX_PER_MM, RESAMPLE_N
from .types import Stroke


def px_to_mm(points_px: np.ndarray) -> np.ndarray:
    """Canvas pixels -> model millimetres.

    Canvas y grows downward, model y grows upward, so y is flipped here. This is
    the ONLY place that flip happens; if a shape ever comes out mirrored, look
    here first.
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] / PX_PER_MM
    out[:, 1] = (CANVAS_H - pts[:, 1]) / PX_PER_MM
    return out


def mm_to_px(points_mm: np.ndarray) -> np.ndarray:
    """Model millimetres -> canvas pixels. The inverse of px_to_mm."""
    pts = np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0] * PX_PER_MM
    out[:, 1] = CANVAS_H - pts[:, 1] * PX_PER_MM
    return out


def arc_length(points: np.ndarray) -> np.ndarray:
    """Cumulative distance along a polyline. Returns (N,), starting at 0."""
    d = np.diff(np.asarray(points, dtype=np.float64), axis=0)
    seg = np.sqrt((d ** 2).sum(axis=1))
    return np.concatenate([[0.0], np.cumsum(seg)])


def resample(points: np.ndarray, n: int = RESAMPLE_N) -> np.ndarray:
    """Resample a polyline to exactly `n` points, evenly spaced by arc length.

    This is what makes strokes comparable. Two people drawing the same circle
    produce wildly different point counts depending on how fast they moved; after
    this they produce the same shape sampled the same way.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return np.repeat(pts, n, axis=0)[:n]

    s = arc_length(pts)
    total = s[-1]
    if total <= 0:
        return np.repeat(pts[:1], n, axis=0)

    targets = np.linspace(0.0, total, n)
    out = np.empty((n, 2))
    out[:, 0] = np.interp(targets, s, pts[:, 0])
    out[:, 1] = np.interp(targets, s, pts[:, 1])
    return out


def normalize(points: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Centre at origin and scale to fit a unit box. Returns (normalised, transform).

    The ML recogniser is trained on normalised strokes so it learns SHAPE, not
    size or position - a 10 mm circle and a 100 mm circle should classify the
    same. The returned transform lets you put the fitted primitive back where the
    user drew it.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    centre = pts.mean(axis=0)
    centred = pts - centre
    scale = float(np.abs(centred).max())
    if scale <= 0:
        scale = 1.0
    return centred / scale, {"cx": float(centre[0]), "cy": float(centre[1]), "scale": scale}


def denormalize(points: np.ndarray, transform: dict[str, float]) -> np.ndarray:
    """Undo `normalize`."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return pts * transform["scale"] + np.array([transform["cx"], transform["cy"]])


def is_closed(points: np.ndarray, tol: float) -> bool:
    """Did the user come back to where they started?"""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return False
    return bool(np.linalg.norm(pts[0] - pts[-1]) <= tol)


def close_ring(points: np.ndarray) -> np.ndarray:
    """Append the first point if the ring is not already closed."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[:1]])
    return pts


def prepare(stroke: Stroke) -> np.ndarray | None:
    """The standard pipeline: pixels -> mm -> resampled. None if the stroke is junk.

    Every recogniser starts by calling this. If you add a preprocessing step,
    add it here so the rule-based and ML arms stay comparable - that comparison
    is the evaluation section of the report.
    """
    if len(stroke.points) < MIN_STROKE_PTS:
        return None
    mm = px_to_mm(stroke.points)
    return resample(mm)


# --------------------------------------------------------------------------- #
# TODO (Serena), roughly in order:
#
#  1. Jitter/tremor smoothing. A moving average or a Savitzky-Golay filter before
#     resampling. Measure whether it actually improves recognition accuracy -
#     if it does not, leave it out and say so in the report.
#  2. Corner detection, so a single stroke that goes round a rectangle can be
#     split into four lines instead of classified as one POLYLINE. Curvature
#     peaks on the resampled stroke are the usual approach.
#  3. Decide whether to use timestamps. Speed dips at corners, which is a strong
#     free signal - but only if the canvas gives you `t`. Check what Gradio
#     actually provides before building on it.
# --------------------------------------------------------------------------- #
