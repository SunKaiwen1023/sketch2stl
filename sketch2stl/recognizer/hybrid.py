"""ML1 first, geometry as the safety net for closed shapes.

OWNER: PK. ML1 (trained from scratch on synthetic strokes) is near-perfect on
lines and arcs, but on REAL mouse-drawn strokes it mixes up the closed shapes:
hand-drawn polylines come out as rectangles (macro-F1 0.77 on PK's 325 real
strokes, polyline F1 0.34).

So ML1 decides line / arc / closed shape, and among closed shapes, where it
disagrees with the least-squares circle / rectangle fits, the fit wins - a
circle only if the stroke has at most one sharp corner, so a four-sided polygon
is never rounded off. On the 325 real strokes: macro-F1 0.91.

Honest caveat for the report: the fits' thresholds were tuned on those same
325 strokes, so 0.91 is an app-quality number, not a held-out score. ML1 on its
own is the held-out number (0.77).
"""
from __future__ import annotations

from ..types import PrimitiveKind, Stroke
from .base import Recognizer


_CLOSED = (PrimitiveKind.CIRCLE, PrimitiveKind.RECT, PrimitiveKind.POLYLINE)


class HybridRecognizer(Recognizer):
    """ML1 decides line / arc / closed. Among CLOSED shapes, where the model and
    the least-squares fits disagree, the fit wins - except that the fit may only
    call something a circle if it has at most one sharp corner."""
    name = "hybrid"

    def __init__(self, learned: Recognizer, rules: Recognizer) -> None:
        self.learned, self.rules = learned, rules

    def recognize(self, stroke: Stroke):
        p = self.learned.recognize(stroke)
        if p.kind not in _CLOSED:
            return p
        q = self.rules.recognize(stroke)
        if q.kind not in _CLOSED or q.kind is p.kind:
            return p
        if q.kind is PrimitiveKind.CIRCLE and _sharp_corners(stroke) > 1:
            return p
        return q


def _sharp_corners(stroke: Stroke) -> int:
    try:
        from ..corners import find_corners
        from ..strokes import prepare
        pts = prepare(stroke)
        return 99 if pts is None else len(find_corners(pts))
    except Exception:                                # noqa: BLE001
        return 99
