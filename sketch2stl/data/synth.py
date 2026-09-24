"""Turn a clean CAD curve into a plausible hand-drawn stroke.

OWNER: Serena.

THIS IS THE IDEA THAT MAKES THE FUSION DATASET USEFUL.

The Fusion 360 Gallery has perfect parametric curves and zero hand-drawn
strokes. Your recogniser needs hand-drawn strokes with known labels. So:

    clean CAD curve  ──distort──►  synthetic hand-drawn stroke
    (label is known exactly)        (label carries over for free)

Five effects, each modelling something a real hand actually does. Every one is
controllable, so you can report an ablation - "which distortion mattered?" is a
genuinely interesting result, and it is cheap to produce.

    1. TREMOR          low-frequency wobble perpendicular to the path. A hand is
                       not a plotter. Modelled as smoothed noise, not white
                       noise, because white noise is trivially filtered out and
                       would make the task artificially easy.
    2. SPEED VARIATION people slow at corners and speed up on straights, so
                       points bunch up at corners. Resampling by a warped arc
                       length reproduces that.
    3. OVERSHOOT       strokes run past their endpoint, or stop short of it.
                       This is the main reason closed shapes do not close, and
                       it is what profiles.py has to survive.
    4. CORNER ROUNDING the hand cannot turn instantly. Sharp corners get a small
                       radius.
    5. SLANT + SCALE   a mild global affine wobble - nobody draws a perfect
                       square axis-aligned.

CALIBRATE AGAINST REALITY. Draw thirty shapes by hand, measure their mean
deviation from the ideal fit, and tune `tremor_mm` until the synthetic strokes
have a similar spread. Otherwise you are training on a fantasy, and the
domain-gap section of the report writes itself in the wrong direction.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..strokes import arc_length, resample


@dataclass
class HandStyle:
    """How shaky a hand to simulate. Defaults are a moderately careful adult.

    Tune these against real strokes rather than trusting them. `--style` in the
    dataset builder exposes the three presets.
    """
    tremor_mm: float = 0.45          # std of the perpendicular wobble
    tremor_smooth: int = 9           # window over which the wobble is smoothed
    speed_warp: float = 0.35         # 0 = even spacing, 1 = heavy bunching
    overshoot_mm: float = 1.2        # max run-past / stop-short at each end
    corner_round_mm: float = 0.8     # radius applied to sharp corners
    slant_deg: float = 1.5           # global shear
    scale_jitter: float = 0.02       # global scale wobble
    n_points: int = 96               # points per synthetic stroke before resampling


NEAT = HandStyle(tremor_mm=0.2, speed_warp=0.2, overshoot_mm=0.4,
                 corner_round_mm=0.4, slant_deg=0.6, scale_jitter=0.01)
TYPICAL = HandStyle()
SHAKY = HandStyle(tremor_mm=0.9, speed_warp=0.6, overshoot_mm=2.5,
                  corner_round_mm=1.6, slant_deg=3.0, scale_jitter=0.05)

# Calibrated against PK's 325 real mouse-drawn strokes, 2026-09-24.
# Measured: real circles fit with a median radial residual of 2.38 mm. TYPICAL
# produced 0.30 mm - SEVEN TIMES too clean. A model trained on TYPICAL has never
# seen a stroke as rough as the ones it meets in the app, which is most of why
# the synthetic-to-real transfer was poor. Derive this with scripts/calibrate.py.
MOUSE = HandStyle(tremor_mm=2.5, tremor_smooth=7, speed_warp=0.45,
                  overshoot_mm=2.0, corner_round_mm=1.2, slant_deg=2.5,
                  scale_jitter=0.04)

STYLES = {"neat": NEAT, "typical": TYPICAL, "shaky": SHAKY, "mouse": MOUSE}


# --------------------------------------------------------------------------- #
# A real hand is not uniformly shaky, and that turned out to matter a lot.
#
# Measured on PK's 325 real mouse strokes:
#
#     real lines    median line-fit residual    0.28 mm   drawn fast, confident
#     real arcs     median circle-fit residual  1.24 mm
#     real circles  median circle-fit residual  2.38 mm   drawn slow, error piles up
#
# An 8x spread between the steadiest and shakiest shape. Applying one tremor_mm
# to all of them - which is what `--style mouse` did at first - makes synthetic
# lines 1.62 mm rough when real ones are 0.28. Every synthetic line then blows
# past LINE_RESIDUAL_MM and the rules arm scored F1 0.000 on lines.
#
# line/arc/circle below are measured. rect and polyline are interpolated: a
# rectangle is four confident strokes so it sits near arcs, and a polyline is
# whatever was left over. Re-derive with scripts/calibrate.py.
# --------------------------------------------------------------------------- #
TREMOR_BY_KIND = {
    "line":     0.45,   # measured
    "arc":      2.00,   # measured
    "circle":   4.00,   # measured
    "rect":     1.50,   # interpolated
    "polyline": 3.00,   # interpolated
}


def style_for(kind, base: HandStyle = MOUSE) -> HandStyle:
    """The drawing style for one shape class. Falls back to `base` if unknown."""
    key = getattr(kind, "value", str(kind))
    tremor = TREMOR_BY_KIND.get(key)
    return base if tremor is None else replace(base, tremor_mm=tremor)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    """Moving average. Turns white noise into the low-frequency wobble a hand makes."""
    if window <= 1 or len(x) < window:
        return x
    k = np.ones(window) / window
    return np.convolve(x, k, mode="same")


def _normals(pts: np.ndarray) -> np.ndarray:
    """Unit normal at each point of a polyline."""
    d = np.gradient(pts, axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    mag = np.linalg.norm(n, axis=1, keepdims=True)
    mag[mag < 1e-9] = 1.0
    return n / mag


def _warp_speed(pts: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """Resample with non-uniform spacing, so points bunch where a hand slowed."""
    if amount <= 0:
        return pts
    s = arc_length(pts)
    if s[-1] <= 0:
        return pts
    u = s / s[-1]
    # a smooth random monotone warp of [0, 1] -> [0, 1]
    k = rng.uniform(-amount, amount, 4)
    warped = u + sum(c * np.sin((i + 1) * np.pi * u) / ((i + 1) * np.pi) for i, c in enumerate(k))
    warped = np.clip(warped, 0.0, 1.0)
    warped = np.maximum.accumulate(warped)          # keep it monotone
    if warped[-1] <= warped[0]:
        return pts
    warped = (warped - warped[0]) / (warped[-1] - warped[0])
    out = np.empty_like(pts)
    out[:, 0] = np.interp(warped, u, pts[:, 0])
    out[:, 1] = np.interp(warped, u, pts[:, 1])
    return out


def _round_corners(pts: np.ndarray, radius_mm: float) -> np.ndarray:
    """Blunt sharp turns - a hand cannot change direction instantly."""
    if radius_mm <= 0 or len(pts) < 5:
        return pts
    step = max(1.0, float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))))
    window = int(np.clip(round(radius_mm / step), 1, max(1, len(pts) // 4)))
    if window <= 1:
        return pts
    k = np.ones(window) / window
    out = pts.copy()
    out[:, 0] = np.convolve(pts[:, 0], k, mode="same")
    out[:, 1] = np.convolve(pts[:, 1], k, mode="same")
    out[:window] = pts[:window]                      # do not drag the endpoints
    out[-window:] = pts[-window:]
    return out


def _endpoints(pts: np.ndarray, overshoot_mm: float, closed: bool,
               rng: np.random.Generator) -> np.ndarray:
    """Run past the end, or stop short of it. The main reason shapes do not close."""
    if overshoot_mm <= 0 or len(pts) < 3:
        return pts
    out = pts.copy()
    for idx, nbr in ((0, 1), (-1, -2)):
        d = out[idx] - out[nbr]
        n = np.linalg.norm(d)
        if n < 1e-9:
            continue
        # A closed shape is usually over-run (you come back past your start);
        # an open one is as likely to stop short.
        low = 0.0 if closed else -overshoot_mm
        out[idx] = out[idx] + (d / n) * rng.uniform(low, overshoot_mm)
    return out


def _affine(pts: np.ndarray, slant_deg: float, scale_jitter: float,
            rng: np.random.Generator) -> np.ndarray:
    """A mild global shear and scale. Nobody draws perfectly upright."""
    centre = pts.mean(axis=0)
    shear = np.tan(np.deg2rad(rng.uniform(-slant_deg, slant_deg)))
    sx = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    sy = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    m = np.array([[sx, shear], [0.0, sy]])
    return (pts - centre) @ m.T + centre


def to_canvas_scale(points: np.ndarray, rng: np.random.Generator | None = None,
                    target_min: float = 25.0, target_max: float = 100.0,
                    centre: tuple[float, float] = (80.0, 60.0)) -> np.ndarray:
    """Rescale a CAD curve to a size a person would plausibly draw it at.

    THIS MUST HAPPEN BEFORE `handdraw`, and leaving it out is a subtle and
    damaging bug.

    Real parts in the Fusion dataset range from a 3 mm pin to a 2 m beam. A
    person draws all of them at roughly canvas size, and hand tremor is in
    CANVAS units - your hand does not shake proportionally to the thing you are
    depicting. Distorting a curve at its native scale means the 2 m beam gets
    0.45 % relative noise (unrealistically clean) while the 3 mm pin gets 9.4 %
    (an unrecognisable mess).

    The classifier then learns "big shapes are clean, small shapes are noisy" -
    a rule that is pure artefact and cannot possibly hold at inference time,
    because by then every stroke arrives at canvas scale.

    Measured on circles before this fix: deviation ran 0.38 mm at 4 mm across
    and 9.10 mm at 2000 mm across. After it, deviation is flat.
    """
    rng = rng or np.random.default_rng()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    # The sheet is 160 x 120 mm, so the SHORT side is the binding constraint.
    # A 140 mm target centred at y=60 would run off the top and bottom.
    extent = float(np.max(hi - lo))
    if extent <= 1e-9:
        return pts
    target = float(rng.uniform(target_min, target_max))
    return (pts - (lo + hi) / 2.0) * (target / extent) + np.asarray(centre, dtype=np.float64)


def handdraw(points: np.ndarray, style: HandStyle = TYPICAL,
             rng: np.random.Generator | None = None,
             closed: bool | None = None) -> np.ndarray:
    """Clean curve (M, 2) in mm -> a synthetic hand-drawn stroke (N, 2) in mm."""
    rng = rng or np.random.default_rng()
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return pts

    if closed is None:
        closed = bool(np.linalg.norm(pts[0] - pts[-1]) < 1e-6)

    pts = resample(pts, style.n_points)
    pts = _warp_speed(pts, style.speed_warp, rng)
    pts = _round_corners(pts, style.corner_round_mm)

    if style.tremor_mm > 0:
        noise = rng.normal(0.0, 1.0, len(pts))
        noise = _smooth(noise, style.tremor_smooth)
        sd = noise.std()
        if sd > 1e-9:
            noise = noise / sd * style.tremor_mm
        pts = pts + _normals(pts) * noise[:, None]

    pts = _endpoints(pts, style.overshoot_mm, closed, rng)
    pts = _affine(pts, style.slant_deg, style.scale_jitter, rng)
    return pts


def deviation(stroke: np.ndarray, clean: np.ndarray, dense: int = 512) -> float:
    """Mean distance from each stroke point to the nearest point ON the clean curve, mm.

    The clean curve must be DENSELY resampled first. A straight line arrives here
    as just two endpoints, and measuring against those two points reports the
    distance to the ends rather than to the line - which inflates the number by
    an order of magnitude and makes a perfectly good stroke look like a scribble.

    Use this to calibrate: draw real shapes, fit the ideal primitive, measure the
    deviation, then tune `tremor_mm` until synthetic strokes match. Put both
    numbers in the report - it is the evidence that the synthetic data is not a
    fantasy.
    """
    s = np.asarray(stroke, dtype=np.float64).reshape(-1, 2)
    c = np.asarray(clean, dtype=np.float64).reshape(-1, 2)
    if len(c) < 2:
        return float("nan")
    c = resample(c, max(dense, len(c)))
    d = np.sqrt(((s[:, None, :] - c[None, :, :]) ** 2).sum(axis=2))
    return float(d.min(axis=1).mean())
