"""Find the real corners in a hand-drawn stroke, and fit each piece between them.

OWNER: Serena.  Added after the 29 Sep review: a drawn star came out with
rounded tips.

WHY THE STAR WAS ROUND
----------------------
The old POLYLINE path was a fallback: when nothing else fitted, we handed the
*smoothed, resampled* stroke straight through. Both of those steps destroy
corners on purpose.

    resample(64)      a 10-point star has 10 tips; 64 points spread evenly by
                      arc length rarely land ON a tip, so each one gets cut
    smooth(window=5)  a moving average over 5 points is exactly the operation
                      that turns a sharp vertex into a fillet

So by the time anything looked at the shape, the corners were already gone.
The fix is not a better smoother. It is to look for corners FIRST, on the raw
stroke, and only then resample each straight piece.

HOW A CORNER IS FOUND
---------------------
Two signals, because either alone is unreliable:

1. TURNING ANGLE. Estimate a tangent from the points within CORNER_WINDOW_MM
   before a point and another from the points after it. The angle between them
   is how sharply the path turns there. A circle drawn at canvas scale turns a
   few degrees per sample; a star tip turns about 108.

   The window matters more than the threshold. Measured on one point it is pure
   noise, since consecutive mouse samples are a pixel apart and tremor dominates.
   Measured over 3 mm of arc length, tremor averages out and real corners stay.

2. PEN SPEED. People decelerate into a deliberate corner and accelerate out of
   it. Hand wobble has no such signature. Where timestamps exist, a local speed
   minimum adds CORNER_SPEED_WEIGHT to the score. Where they do not, the angle
   carries the decision alone and nothing breaks.

Candidates are then non-maximum suppressed: within CORNER_MIN_SEP_MM only the
strongest survives, because a real corner drawn by a human is 2-3 mm of high
curvature rather than one sharp point.

WHAT COMES OUT
--------------
A list of Segment pieces, each already fitted as a LINE or an ARC, plus the
corner positions. Where two fitted LINES meet, their infinite extensions are
intersected to recover a genuinely sharp vertex - the corner the person meant,
not the rounded one their hand drew.

    star   -> 10 lines, 10 sharp vertices
    L      ->  2 lines, 1 sharp vertex
    slot   ->  line, arc, line, arc
    circle ->  no corners; caller keeps treating it as a circle
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (CORNER_ANGLE_DEG, CORNER_ANGLE_MAX_DEG, CORNER_MAD_K,
                     CORNER_DENOISE_FRACTION, CORNER_DENOISE_MAX_MM,
                     CORNER_MIN_SEP_MM,
                     CORNER_SPEED_WEIGHT, CORNER_WINDOW_FRACTION,
                     CORNER_WINDOW_MAX_MM, CORNER_WINDOW_MM,
                     SEGMENT_LINE_RESIDUAL_MM, SEGMENT_MIN_PTS,
                     SEGMENT_RESIDUAL_FRACTION, SEGMENT_SPLIT_DEPTH,
                     VERTEX_SNAP_MM)
from .recognizer.rules import fit_arc, fit_line
from .strokes import arc_length
from .types import PrimitiveKind


@dataclass(frozen=True)
class Segment:
    """One piece of a stroke between two corners, already fitted.

    `points` is the fitted polyline in mm - a straight 2-point line for a LINE,
    a tessellated arc for an ARC. `raw` is the original sample span it came
    from, kept so a caller can re-fit differently or show the user what was
    thrown away.
    """
    kind: PrimitiveKind
    points: np.ndarray
    params: dict[str, float]
    residual: float
    raw: np.ndarray


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros(2)


def _denoise(points: np.ndarray, mm: float | None = None) -> np.ndarray:
    """Light moving average over `mm` of arc length, for SCORING ONLY.

    Never fit to the output of this. Its job is to remove mouse jitter before
    we measure turning angles, so tremor does not read as a hundred corners.
    The endpoints are left untouched, because a convolution drags them inward.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 5:
        return pts
    total = float(arc_length(pts)[-1])
    if total <= 0:
        return pts
    if mm is None:
        mm = float(np.clip(CORNER_DENOISE_FRACTION * total, 1.0,
                           CORNER_DENOISE_MAX_MM))
    spacing = total / (len(pts) - 1)
    w = int(max(1, round(mm / max(spacing, 1e-9))))
    if w < 3:
        return pts
    if w % 2 == 0:
        w += 1
    w = min(w, len(pts) // 2 * 2 - 1)
    if w < 3:
        return pts
    k = np.ones(w) / w
    out = pts.copy()
    for d in (0, 1):
        out[:, d] = np.convolve(pts[:, d], k, mode="same")
    half = w // 2
    out[:half] = pts[:half]
    out[-half:] = pts[-half:]
    return out


def tangent_window(points: np.ndarray) -> float:
    """How far to look each side when estimating a tangent, scaled to the stroke."""
    total = float(arc_length(np.asarray(points, dtype=np.float64))[-1])
    return float(np.clip(CORNER_WINDOW_FRACTION * total,
                         CORNER_WINDOW_MM, CORNER_WINDOW_MAX_MM))


def turning_angles(points: np.ndarray, window_mm: float | None = None,
                   closed: bool = False) -> np.ndarray:
    """Angle in degrees that the path turns at each point, over an arc-length window.

    On a CLOSED ring the window wraps past the seam. Without that, a corner
    sitting on the first/last point is never scored at all - which is why every
    closed polygon came back one corner short.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    n = len(pts)
    out = np.zeros(n)
    if n < 3:
        return out
    if window_mm is None:
        window_mm = tangent_window(pts)
    s = arc_length(pts)
    total = float(s[-1])

    if closed and n > 3:
        m = n - 1                                   # unique points on the ring
        P = np.vstack([pts[:-1]] * 3)
        S = np.concatenate([s[:-1], s[:-1] + total, s[:-1] + 2 * total])
        for i in range(n):
            c = i % m + m
            lo = c
            while S[c] - S[lo] < window_mm and lo > 0:
                lo -= 1
            hi = c
            while S[hi] - S[c] < window_mm and hi < len(P) - 1:
                hi += 1
            before, after = _unit(P[c] - P[lo]), _unit(P[hi] - P[c])
            if before.any() and after.any():
                out[i] = np.degrees(np.arccos(float(np.clip(before @ after, -1.0, 1.0))))
        return out

    for i in range(n):
        lo = i
        while lo > 0 and s[i] - s[lo] < window_mm:
            lo -= 1
        hi = i
        while hi < n - 1 and s[hi] - s[i] < window_mm:
            hi += 1
        if lo == i or hi == i:
            continue
        before, after = _unit(pts[i] - pts[lo]), _unit(pts[hi] - pts[i])
        if before.any() and after.any():
            out[i] = np.degrees(np.arccos(float(np.clip(before @ after, -1.0, 1.0))))
    return out


def speed_profile(points: np.ndarray, t: np.ndarray | None) -> np.ndarray | None:
    """Per-point speed, normalised to its own median. None when there are no times."""
    if t is None:
        return None
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    times = np.asarray(t, dtype=np.float64).ravel()
    if len(times) != len(pts) or len(pts) < 3:
        return None
    dt = np.gradient(times)
    if not np.all(np.isfinite(dt)) or np.allclose(dt, 0):
        return None
    step = np.gradient(pts, axis=0)
    speed = np.sqrt((step ** 2).sum(axis=1)) / np.where(np.abs(dt) < 1e-9, 1e-9, dt)
    med = float(np.median(speed))
    if med <= 0:
        return None
    return speed / med


def find_corners(points: np.ndarray, t: np.ndarray | None = None,
                 angle_deg: float = CORNER_ANGLE_DEG,
                 min_sep_mm: float = CORNER_MIN_SEP_MM,
                 closed: bool = False) -> list[int]:
    """Indices of the real corners in a RAW stroke, in order.

    Call this BEFORE resampling or smoothing. Afterwards there is nothing left
    to find.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < SEGMENT_MIN_PTS:
        return []

    # Score on a lightly denoised copy; return indices into the ORIGINAL array.
    # _denoise preserves indexing, so the two stay aligned.
    smooth = _denoise(pts)
    window = tangent_window(pts)
    angles = turning_angles(smooth, window, closed=closed)
    score = angles.copy()

    speed = speed_profile(pts, t)
    if speed is not None:
        # A dip to half the median speed adds a full CORNER_SPEED_WEIGHT of the
        # angle threshold to the score. Fast points are not penalised.
        dip = np.clip(1.0 - speed, 0.0, 1.0)
        score = score + dip * CORNER_SPEED_WEIGHT * angle_deg

    # Threshold adapts to THIS stroke's own noise, but is capped. A shaky
    # stroke turns sharply everywhere, so a fixed floor over-detects; a
    # corner-dense shape like a star has corners AS its distribution, so an
    # uncapped median + k*MAD under-detects. Both failure modes were measured.
    med = float(np.median(score))
    mad = float(np.median(np.abs(score - med))) + 1e-6
    threshold = float(np.clip(max(angle_deg, med + CORNER_MAD_K * mad),
                              angle_deg, CORNER_ANGLE_MAX_DEG))

    s = arc_length(pts)
    candidates = [i for i in range(len(pts)) if score[i] >= threshold]
    if not candidates:
        return []

    # Non-maximum suppression by arc length: strongest first, drop anything
    # within min_sep_mm of one already kept.
    # Separation must be at least the tangent window: every point WITHIN the
    # window of a corner also registers a large angle, so a smaller separation
    # returns the same corner several times.
    sep = max(min_sep_mm, window)
    kept: list[int] = []
    for i in sorted(candidates, key=lambda j: -score[j]):
        if all(abs(s[i] - s[j]) >= sep for j in kept):
            kept.append(i)
    kept.sort()

    if closed and len(kept) >= 2:
        # On a closed ring the two ends are neighbours, so a corner found near
        # both ends is one corner counted twice.
        total = s[-1]
        if total - (s[kept[-1]] - s[kept[0]]) < sep:
            kept.pop()
    return kept


def _tolerance(points: np.ndarray) -> float:
    """How well a piece has to fit, scaled to its own length.

    A 200 mm edge drawn by hand deviates more in absolute millimetres than a
    10 mm one and is no less straight for it.
    """
    total = float(arc_length(points)[-1])
    return max(SEGMENT_LINE_RESIDUAL_MM, SEGMENT_RESIDUAL_FRACTION * total)


def _fit_once(raw: np.ndarray) -> Segment:
    """Best single fit for one span: LINE if straight enough, else ARC."""
    pts = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
    a, b, line_res = fit_line(pts)
    line_seg = Segment(PrimitiveKind.LINE, np.vstack([a, b]),
                       {"x1": float(a[0]), "y1": float(a[1]),
                        "x2": float(b[0]), "y2": float(b[1])},
                       float(line_res), pts)
    if line_res <= _tolerance(pts) or len(pts) < SEGMENT_MIN_PTS:
        return line_seg
    arc = fit_arc(pts)
    if arc is not None:
        ring, params, arc_res = arc
        if arc_res < line_res:
            return Segment(PrimitiveKind.ARC, ring, params, float(arc_res), pts)
    return line_seg


def _worst_point(raw: np.ndarray) -> int:
    """Index of the point furthest from the chord. Where to split a bad fit."""
    pts = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
    a, b = pts[0], pts[-1]
    d = b - a
    n = float(np.linalg.norm(d))
    if n < 1e-9:
        return len(pts) // 2
    d = d / n
    rel = pts - a
    perp = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
    return int(np.argmax(perp))


def _fit_piece(raw: np.ndarray, depth: int = SEGMENT_SPLIT_DEPTH) -> list[Segment]:
    """Fit a span, splitting recursively when neither a line nor an arc will do.

    This is what recovers a slot. Its line-arc junctions are TANGENT: the path
    turns through zero degrees there, so no corner detector will ever find them.
    But a single line or arc cannot describe line+arc+line either, so the fit is
    poor, and splitting at the worst point lands on the junction.
    """
    pts = np.asarray(raw, dtype=np.float64).reshape(-1, 2)
    seg = _fit_once(pts)
    if depth <= 0 or seg.residual <= _tolerance(pts) or len(pts) < 2 * SEGMENT_MIN_PTS:
        return [seg]
    k = _worst_point(pts)
    if k < SEGMENT_MIN_PTS or k > len(pts) - SEGMENT_MIN_PTS:
        return [seg]
    left = _fit_piece(pts[:k + 1], depth - 1)
    right = _fit_piece(pts[k:], depth - 1)
    combined = left + right
    # Only accept the split if it genuinely fits better than the single piece.
    if max(x.residual for x in combined) < seg.residual * 0.8:
        return combined
    return [seg]


def _intersect(p1, d1, p2, d2) -> np.ndarray | None:
    """Where two infinite lines cross. None when they are near-parallel."""
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(cross) < 1e-9:
        return None
    w = p2 - p1
    t = float(w[0] * d2[1] - w[1] * d2[0]) / cross
    return p1 + t * d1


def sharpen(segments: list[Segment], closed: bool = False) -> list[Segment]:
    """Replace rounded line-line joins with the intersection of the two lines.

    This is what actually puts the point back on the star. The hand draws a
    small arc through the tip; the two straight flanks either side are what the
    person meant, and their intersection is the vertex they were aiming for.

    Only applied when the intersection lands within VERTEX_SNAP_MM of the drawn
    join, so two nearly parallel lines cannot fling a vertex off the page.
    """
    if len(segments) < 2:
        return segments
    pts = [s.points.copy() for s in segments]
    pairs = list(zip(range(len(segments) - 1), range(1, len(segments))))
    if closed and len(segments) > 2:
        pairs.append((len(segments) - 1, 0))

    for i, j in pairs:
        if segments[i].kind is not PrimitiveKind.LINE:
            continue
        if segments[j].kind is not PrimitiveKind.LINE:
            continue
        a0, a1 = pts[i][0], pts[i][-1]
        b0, b1 = pts[j][0], pts[j][-1]
        x = _intersect(a0, a1 - a0, b0, b1 - b0)
        if x is None:
            continue
        join = (a1 + b0) / 2.0
        if float(np.linalg.norm(x - join)) > VERTEX_SNAP_MM:
            continue
        pts[i][-1] = x
        pts[j][0] = x

    return [Segment(s.kind, p, s.params, s.residual, s.raw)
            for s, p in zip(segments, pts)]


def segment_stroke(points: np.ndarray, t: np.ndarray | None = None,
                   closed: bool = False,
                   seams: tuple[int, ...] = ()) -> tuple[list[Segment], list[int]]:
    """Split a RAW stroke at its corners and fit each piece. The main entry point.

    `seams` are indices where two strokes were stitched together. Those are
    corners for free - a person lifting the pen almost always does it at one -
    so they are added to whatever the detector finds.

    Returns (segments, corner_indices_into_the_input).
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < SEGMENT_MIN_PTS:
        return [], []

    corners = find_corners(pts, t, closed=closed)
    if seams:
        s_len = arc_length(pts)
        for k in seams:
            if 0 < k < len(pts) - 1 and all(abs(s_len[k] - s_len[c]) >= CORNER_MIN_SEP_MM
                                            for c in corners):
                corners.append(int(k))
        corners.sort()

    work = pts
    bounds: list[int]
    if closed and corners:
        # Roll so the ring starts AND ends at the first corner. Then every piece
        # lies between two corners and the ring closes without a stray join.
        shift = corners[0]
        work = np.vstack([pts[shift:], pts[1:shift + 1]]) if shift else pts
        rolled = [0] + [c - shift for c in corners[1:]] + [len(work) - 1]
        bounds = rolled
    else:
        bounds = [0] + corners + [len(pts) - 1]

    segments: list[Segment] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        piece = work[a:b + 1]
        if len(piece) >= 3:
            segments.extend(_fit_piece(piece))
    if not segments:
        segments = _fit_piece(work)
    return sharpen(segments, closed=closed), corners


def polyline_from_segments(segments: list[Segment], closed: bool = False) -> np.ndarray:
    """Chain fitted segments into one polyline for the kernel."""
    if not segments:
        return np.empty((0, 2))
    out = [segments[0].points]
    for s in segments[1:]:
        out.append(s.points[1:] if len(s.points) > 1 else s.points)
    ring = np.vstack(out)
    if closed and np.linalg.norm(ring[0] - ring[-1]) > 1e-9:
        ring = np.vstack([ring, ring[:1]])
    return ring
