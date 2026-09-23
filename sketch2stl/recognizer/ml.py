"""Learned recogniser. STUB - this is Serena's main build.

OWNER: Serena.

Plan, which maps directly onto what HW1 and HW2 already taught you:

  DATA   Collect strokes with a label each: line / arc / circle / rect / polyline.
         Both of you drawing 40 shapes each of 5 classes is 400 samples, which
         is enough for a small model on 64x2 normalised inputs. Save as a
         HuggingFace dataset the same way you did for HW1 - split by SESSION
         (whose hand drew it, on what day), never randomly, or the model learns
         your handwriting instead of the shape. That is the same parent_id
         grouping lesson from HW2, in a different costume.

  MODEL  Two options, try the cheap one first:
         (a) Features + classical model. Compute ~10 hand-designed features from
             the normalised stroke - closure, aspect ratio, corner count,
             curvature variance, circle-fit residual, line-fit residual - and
             throw AutoGluon TabularPredictor at them. This reuses HW2 Problem 1
             almost line for line, and it will be a strong baseline.
         (b) Small 1-D CNN over the (64, 2) resampled stroke. More impressive to
             present, more likely to overfit 400 samples.
         Report both. The comparison IS the contribution.

  PARAMS Classification alone is not enough - you still need the circle's centre
         and radius. Do NOT regress those with the network. Classify the shape,
         then run the matching fitter from rules.py. Much more accurate, and it
         means the ML arm and the rules arm share their geometry code, so the
         comparison isolates recognition.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..strokes import prepare
from ..types import Primitive, PrimitiveKind, Stroke
from .base import Recognizer
from .rules import RuleRecognizer

CLASSES = [PrimitiveKind.LINE, PrimitiveKind.ARC, PrimitiveKind.CIRCLE,
           PrimitiveKind.RECT, PrimitiveKind.POLYLINE]


def stroke_features(points_mm: np.ndarray) -> dict[str, float]:
    """Hand-designed features for the classical arm. STUB.

    Return a flat dict - it becomes one row of the training table. Suggested:
      closure_ratio    dist(first, last) / total arc length
      aspect_ratio     bbox width / height of the normalised stroke
      fill_ratio       polygon area / bbox area  (separates circle from rect)
      n_corners        curvature peaks above a threshold
      circle_residual  from rules.fit_circle
      line_residual    from rules.fit_line
      curvature_std    how much the turning rate varies (low = circle, high = rect)
    """
    raise NotImplementedError("Serena: see docs/architecture.md section 3")


class MLRecognizer(Recognizer):
    name = "ml"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model = None
        self.fallback = RuleRecognizer()
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: str | Path) -> None:
        """Load a trained model. STUB.

        If you go the AutoGluon route this is TabularPredictor.load(path).
        If you go the CNN route it is torch.load plus model.eval().
        """
        raise NotImplementedError("Serena: see recognizer/train.py")

    def recognize(self, stroke: Stroke) -> Primitive:
        # Until the model exists, degrade to rules rather than crash. This means
        # you can wire the UI to MLRecognizer from day one and swap the brain in
        # later without touching anything else.
        if self.model is None:
            p = self.fallback.recognize(stroke)
            return Primitive(p.kind, p.points, p.params, p.confidence, source="rules")

        pts = prepare(stroke)
        if pts is None:
            return Primitive(PrimitiveKind.POLYLINE, np.zeros((0, 2)), confidence=0.0, source="ml")

        # kind = self.model.predict(...)
        # then call the matching fitter from rules.py to get the parameters
        raise NotImplementedError("Serena: classify here, then fit with rules.py")
