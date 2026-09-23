"""The recogniser interface. Three implementations plug in here.

OWNER: Serena (interface is shared - agree before changing).

The three arms are the evaluation section of your report:

    manual  - the user picks the tool first (Circle / Rect / Line), so there is
              no recognition at all. This is the ceiling: whatever accuracy the
              other two get, this is what they are trying to reach.
    rules   - fit each primitive type by least squares, keep the best residual.
              Cheap, explainable, no training data, and often annoyingly good.
    ml      - a small learned classifier on normalised strokes.

The interesting result is NOT "our model gets 94%". It is whether the ML arm
beats the rules arm by enough to justify existing, and on which shapes. Build
the rules arm first and take it seriously, or the comparison is worthless.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import Primitive, Stroke


class Recognizer(ABC):
    """Turn one raw stroke into one recognised primitive."""

    name: str = "base"

    @abstractmethod
    def recognize(self, stroke: Stroke) -> Primitive:
        """Return a Primitive. Never raise, never return None.

        If you cannot do better than the raw path, return a POLYLINE with a low
        confidence. The geometry half only reads `points`, so a POLYLINE still
        produces a solid - the user just gets their wobbly line instead of a
        clean one. Degrading is always better than failing here.
        """
        raise NotImplementedError

    def recognize_batch(self, strokes: list[Stroke]) -> list[Primitive]:
        return [self.recognize(s) for s in strokes]
