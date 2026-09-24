"""Raw canvas strokes -> clean, fixed-length, millimetre-space point arrays.

OWNER: Serena.

This is the first half of the ML story. Nothing here is learned - it is the
preprocessing every recogniser (rules or ML) sits on top of. Getting it right
matters more than the model: a resampled, normalised stroke makes a rule-based
fit work well and makes a small network trainable on a few hundred examples.
"""
from __future__ import annotations

import numpy as np

from .config import (CANVAS_H, CLOSE_FRACTION, MIN_STROKE_PTS, PX_PER_MM,
                     RESAMPLE_N, SMOOTH_WINDOW)
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


def is_closed(points: np.ndarray, tol: float, frac: float = CLOSE_FRACTION) -> bool:
    """Did the user come back to where they started?

    Two tests, either is enough:
      absolute  - the gap is under `tol` millimetres
      relative  - the gap is under `frac` of the stroke's own length

    The relative one matters. A fixed 3 mm tolerance is generous on a 10 mm
    circle and impossibly strict on a 300 mm outline, so big shapes were being
    read as open - a closed rectangle would come back as a POLYLINE, and a
    circle as an ARC whose chord encloses a sliver instead of a disc.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return False
    gap = float(np.linalg.norm(pts[0] - pts[-1]))
    if gap <= tol:
        return True
    total = float(arc_length(pts)[-1])
    return bool(total > 0 and gap / total <= frac)


def close_ring(points: np.ndarray) -> np.ndarray:
    """Append the first point if the ring is not already closed."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if np.linalg.norm(pts[0] - pts[-1]) > 1e-9:
        pts = np.vstack([pts, pts[:1]])
    return pts


def smooth(points: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Damp hand tremor with a short moving average, endpoints held fixed.

    Resampling already averages a lot - a 1000-point skeleton collapsing to 64
    points is itself a low-pass filter - and least-squares fitting is inherently
    noise-robust. So this is a small extra win, not a transformation.

    It is applied in `prepare_points`, which BOTH the training-set builder and
    the live app call. Smoothing only one of them would mean the model trains on
    rough strokes and sees smooth ones, which quietly costs accuracy in a way
    that no test would catch.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if window <= 1 or len(pts) < window * 2:
        return pts
    k = np.ones(window) / window
    out = pts.copy()
    out[:, 0] = np.convolve(pts[:, 0], k, mode="same")
    out[:, 1] = np.convolve(pts[:, 1], k, mode="same")
    half = window // 2 + 1
    out[:half] = pts[:half]          # a moving average pulls the ends inward
    out[-half:] = pts[-half:]
    return out


def prepare_points(points_mm: np.ndarray) -> np.ndarray:
    """Resample then smooth. The single shared entry point for both pipelines."""
    return smooth(resample(np.asarray(points_mm, dtype=np.float64).reshape(-1, 2)))


def prepare(stroke: Stroke) -> np.ndarray | None:
    """The standard pipeline: pixels -> mm -> resampled. None if the stroke is junk.

    Every recogniser starts by calling this. If you add a preprocessing step,
    add it here so the rule-based and ML arms stay comparable - that comparison
    is the evaluation section of the report.
    """
    if len(stroke.points) < MIN_STROKE_PTS:
        return None
    return prepare_points(px_to_mm(stroke.points))


# --------------------------------------------------------------------------- #
# TODO (Serena), roughly in order:
#
#  1. Corner detection, so a single stroke that goes round a rectangle can be
#     split into four lines instead of classified as one POLYLINE. Curvature
#     peaks on the resampled stroke are the usual approach.
#  2. Decide whether to use timestamps. Speed dips at corners, which is a strong
#     free signal - but only if the canvas gives you `t`. Check what Gradio
#     actually provides before building on it.
# --------------------------------------------------------------------------- #
