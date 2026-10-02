"""Several strokes -> one closed contour. The "sketch contour" the prof asked for.

OWNER: Serena.  Added after the 29 Sep review.

THE ASSUMPTION WE HAD, AND WHY IT WAS WRONG
--------------------------------------------
The app assumed 1 shape = 1 stroke. Nobody draws that way. A rectangle gets
drawn as four strokes, one edge at a time. A big outline gets interrupted
because the hand runs out of room. A person lifts the pen to think.

So before anything is recognised, the strokes that belong to one outline have
to be found and joined into a single ordered path. Everything downstream is
unchanged: the stitched contour goes into ML1 exactly like a single stroke did,
and a four-stroke rectangle still comes out RECT, not POLYLINE.

HOW ENDPOINTS ARE PAIRED
------------------------
By distance, greedily, closest pair first. Each endpoint may be used once.
A stroke can be reversed if it runs the wrong way round the loop.

Distance alone is deliberate. A time-gap feature would be the obvious second
signal - people pause longer between shapes than within one - but none of our
data has real pauses between strokes, so a model trained on it would be
learning an artefact of the synthesiser. Stated in the limitations rather than
quietly used.

The join tolerance is STROKE_JOIN_MM (8 mm), deliberately looser than
CLOSE_TOL_MM (3 mm). Those answer different questions: "did one stroke close on
itself" is a claim about one person's precision returning to a point they can
see, while "do these two strokes belong together" spans a pen lift, where
people are sloppier.

SEAMS
-----
The returned Contour records where the joins landed. Two things use that:

  - the corner detector treats a seam as a free corner, because a person
    lifting the pen almost always does it at one (PK's observation, and it is
    right: it is how you draw a rectangle)
  - the seam is smoothed slightly before ML1 sees it, so the tiny kink where
    two strokes meet does not turn a circle drawn in two halves into a polyline
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import (SEAM_SMOOTH_MM, STROKE_JOIN_FRACTION, STROKE_JOIN_MM)
from .strokes import arc_length


@dataclass(frozen=True)
class Contour:
    """One shape the user drew, from one or more strokes.

    `points` is the stitched path in mm, closed (first == last) when `closed`.
    `seams` are indices into `points` where two strokes were joined.
    `n_strokes` is how many went in, so the UI can say "4 strokes joined".
    """
    points: np.ndarray
    closed: bool
    seams: tuple[int, ...] = ()
    n_strokes: int = 1

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"Contour.points must be (N, 2), got {self.points.shape}")


def _ends(stroke: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return stroke[0], stroke[-1]


def join_tolerance(strokes: list[np.ndarray]) -> float:
    """How close two endpoints have to be, scaled to the drawing's own size."""
    total = sum(float(arc_length(s)[-1]) for s in strokes if len(s) > 1)
    return max(STROKE_JOIN_MM, STROKE_JOIN_FRACTION * total)


def stitch(strokes: list[np.ndarray], tol: float | None = None) -> Contour:
    """Join strokes into one ordered path, closing it if the ends meet.

    This is the rule baseline PK named: pair endpoints by distance under a
    tolerance. A learned version has to beat it before it is worth shipping.
    """
    pieces = [np.asarray(s, dtype=np.float64).reshape(-1, 2)
              for s in strokes if len(np.asarray(s).reshape(-1, 2)) >= 2]
    if not pieces:
        return Contour(np.empty((0, 2)), False, (), 0)
    if len(pieces) == 1:
        p = pieces[0]
        return Contour(p, _meets(p[0], p[-1], tol or join_tolerance(pieces)), (), 1)

    if tol is None:
        tol = join_tolerance(pieces)

    # Greedy nearest-endpoint chaining. Start from the first stroke and keep
    # attaching whichever free end is closest to the growing path's tail.
    remaining = list(range(len(pieces)))
    order: list[tuple[int, bool]] = [(remaining.pop(0), False)]
    path = pieces[order[0][0]].copy()
    seams: list[int] = []

    while remaining:
        tail = path[-1]
        best = None
        for idx in remaining:
            a, b = _ends(pieces[idx])
            for flipped, end in ((False, a), (True, b)):
                d = float(np.linalg.norm(end - tail))
                if best is None or d < best[0]:
                    best = (d, idx, flipped)
        d, idx, flipped = best
        if d > tol:
            break                       # nothing close enough; the shape ends here
        remaining.remove(idx)
        nxt = pieces[idx][::-1] if flipped else pieces[idx]
        seams.append(len(path) - 1)
        path = np.vstack([path, nxt])
        order.append((idx, flipped))

    closed = _meets(path[0], path[-1], tol)
    if closed and np.linalg.norm(path[0] - path[-1]) > 1e-9:
        seams.append(len(path) - 1)
        path = np.vstack([path, path[:1]])

    return Contour(path, closed, tuple(seams), len(order))


def stitch_all(strokes: list[np.ndarray], tol: float | None = None) -> list[Contour]:
    """Several separate outlines drawn in one go -> one Contour EACH.

    `stitch` builds ONE path and silently drops every stroke it cannot reach, so
    a plate drawn together with the two holes inside it came back as just one of
    the three. This keeps going: chain strokes into a shape until it closes or
    nothing is near enough, then start a new shape from whatever is left.

    Two differences from `stitch`, both because shapes are now neighbours:
      * a chain STOPS as soon as it closes, so a finished circle never grabs
        the stroke next to it;
      * the join tolerance grows with the shape being built, not with the whole
        drawing, so a big outline does not swallow a small hole beside it.
    """
    pieces = [np.asarray(s, dtype=np.float64).reshape(-1, 2)
              for s in strokes if len(np.asarray(s).reshape(-1, 2)) >= 2]
    out: list[Contour] = []
    remaining = list(range(len(pieces)))
    while remaining:
        first = remaining.pop(0)
        path = pieces[first].copy()
        seams: list[int] = []
        n = 1

        def limit(p):
            if tol is not None:
                return tol
            return max(STROKE_JOIN_MM, STROKE_JOIN_FRACTION * float(arc_length(p)[-1]))

        while remaining and not (n > 1 and _meets(path[0], path[-1], limit(path))):
            if n == 1 and _meets(path[0], path[-1], limit(path)) and len(path) > 8:
                break                                   # a single stroke that already closes
            tail, best = path[-1], None
            for idx in remaining:
                a, b = _ends(pieces[idx])
                for flipped, end in ((False, a), (True, b)):
                    d = float(np.linalg.norm(end - tail))
                    if best is None or d < best[0]:
                        best = (d, idx, flipped)
            # also allow growing at the HEAD, so stroke order on the canvas does not matter
            head = path[0]
            for idx in remaining:
                a, b = _ends(pieces[idx])
                for flipped, end in ((True, a), (False, b)):
                    d = float(np.linalg.norm(end - head))
                    if d < best[0]:
                        best = (d, idx, flipped, "head")
            if best[0] > limit(np.vstack([path, pieces[best[1]]])):
                break
            idx, flipped = best[1], best[2]
            remaining.remove(idx)
            nxt = pieces[idx][::-1] if flipped else pieces[idx]
            if len(best) == 4:                          # prepend
                seams = [k + len(nxt) for k in seams] + [len(nxt) - 1]
                path = np.vstack([nxt, path])
            else:
                seams.append(len(path) - 1)
                path = np.vstack([path, nxt])
            n += 1
        closed = _meets(path[0], path[-1], limit(path))
        if closed and n > 1 and np.linalg.norm(path[0] - path[-1]) > 1e-9:
            seams.append(len(path) - 1)
            path = np.vstack([path, path[:1]])
        out.append(Contour(path, closed, tuple(seams), n))
    return out


def _meets(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    return bool(np.linalg.norm(np.asarray(a) - np.asarray(b)) <= tol)


def smooth_seams(contour: Contour, mm: float = SEAM_SMOOTH_MM) -> np.ndarray:
    """Blend a short span either side of each seam.

    A seam is a small kink where two strokes met. Left alone it reads as a
    corner, which would turn a circle drawn in two halves into a polyline. The
    blend is local: everywhere else the path is untouched, so real corners
    elsewhere survive.
    """
    pts = contour.points.copy()
    if not contour.seams or len(pts) < 5:
        return pts
    s = arc_length(pts)
    for k in contour.seams:
        if k <= 0 or k >= len(pts) - 1:
            continue
        lo = int(np.searchsorted(s, s[k] - mm))
        hi = int(np.searchsorted(s, s[k] + mm))
        lo, hi = max(lo, 0), min(hi, len(pts) - 1)
        if hi - lo < 3:
            continue
        w = hi - lo + 1
        span = pts[lo:hi + 1]
        # Straight blend between the two ends of the span.
        t = np.linspace(0.0, 1.0, w)[:, None]
        pts[lo:hi + 1] = span * 0.5 + (span[0] * (1 - t) + span[-1] * t) * 0.5
    return pts


def should_finish(last_point_time: float, now: float, idle_s: float) -> bool:
    """Has the pen been up long enough to call the shape finished?"""
    return (now - last_point_time) >= idle_s
