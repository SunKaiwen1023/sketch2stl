"""The learned recogniser: classify the shape, then fit its parameters exactly.

OWNER: Serena.

THE DESIGN DECISION THAT MATTERS
--------------------------------
The model predicts the shape CLASS only. It never regresses a circle's centre or
radius. Those come from the same least-squares fitters the rules arm uses.

Why: on a few thousand samples a network will predict a radius to maybe 5 %.
`fit_circle` gets it to 0.2 % because it is solving the actual problem. And
because both arms share the fitting code, the evaluation isolates *recognition*
instead of confounding it with parameter estimation - which is what makes the
three-arm comparison a real experiment rather than a vibe.

GRACEFUL DEGRADATION
--------------------
If no model is loaded, this falls back to the rules recogniser rather than
raising. That means `app.py` can point at `MLRecognizer` from day one and the
app keeps working - the brain gets swapped in later without touching the UI.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..config import LOW_CONFIDENCE
from ..strokes import close_ring, is_closed, prepare
from ..config import CLOSE_TOL_MM
from ..types import Primitive, PrimitiveKind, Stroke
from .base import Recognizer
from .features import FEATURE_NAMES, extract
from .rules import RuleRecognizer, circle_points, fit_circle, fit_line

CLASSES = [PrimitiveKind.LINE, PrimitiveKind.ARC, PrimitiveKind.CIRCLE,
           PrimitiveKind.RECT, PrimitiveKind.POLYLINE]


def fit_params(kind: PrimitiveKind, pts: np.ndarray) -> tuple[np.ndarray, dict]:
    """Given a predicted class, fit that primitive's exact parameters.

    Shared by the ML arm and the rules arm on purpose - see the module docstring.
    """
    if kind is PrimitiveKind.CIRCLE:
        cx, cy, r, _ = fit_circle(pts)
        if r > 0:
            return circle_points(cx, cy, r), {"cx": cx, "cy": cy, "r": r}

    elif kind is PrimitiveKind.LINE:
        p0, p1, _ = fit_line(pts)
        return np.vstack([p0, p1]), {"x1": p0[0], "y1": p0[1], "x2": p1[0], "y2": p1[1]}

    elif kind is PrimitiveKind.ARC:
        cx, cy, r, _ = fit_circle(pts)
        if r > 0:
            a0 = float(np.arctan2(pts[0, 1] - cy, pts[0, 0] - cx))
            a1 = float(np.arctan2(pts[-1, 1] - cy, pts[-1, 0] - cx))
            if a1 <= a0:
                a1 += 2 * np.pi
            a = np.linspace(a0, a1, 48)
            arc = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
            return arc, {"cx": cx, "cy": cy, "r": r, "a0": a0, "a1": a1}

    elif kind is PrimitiveKind.RECT:
        rect = _min_area_rect(pts)
        if rect is not None:
            return rect

    # POLYLINE, or any fit that failed: hand back the stroke itself.
    return (close_ring(pts) if is_closed(pts, CLOSE_TOL_MM) else pts), {}


def _min_area_rect(pts: np.ndarray) -> tuple[np.ndarray, dict] | None:
    """Minimum-area enclosing rectangle, via shapely. Also used by the rules arm."""
    try:
        from shapely.geometry import MultiPoint
        rect = MultiPoint([tuple(p) for p in pts]).minimum_rotated_rectangle
        ring = np.asarray(rect.exterior.coords, dtype=np.float64)
    except Exception:
        return None
    if len(ring) < 5:
        return None
    e0 = ring[1] - ring[0]
    e1 = ring[2] - ring[1]
    w, h = float(np.linalg.norm(e0)), float(np.linalg.norm(e1))
    centre = ring[:4].mean(axis=0)
    angle = float(np.arctan2(e0[1], e0[0]))
    return ring, {"cx": float(centre[0]), "cy": float(centre[1]),
                  "w": w, "h": h, "angle": angle}


class MLRecognizer(Recognizer):
    """Feature-based classifier. Falls back to rules when no model is loaded."""

    name = "ml"

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model = None
        self.classes_: list[PrimitiveKind] = []
        self.meta: dict = {}
        self.fallback = RuleRecognizer()
        if model_path is not None and Path(model_path).exists():
            self.load(model_path)

    # --- persistence ------------------------------------------------------ #
    def load(self, model_path: str | Path) -> None:
        import joblib
        model_path = Path(model_path)
        bundle = joblib.load(model_path / "model.joblib" if model_path.is_dir() else model_path)
        self.model = bundle["model"]
        self.classes_ = [PrimitiveKind(c) for c in bundle["classes"]]
        self.meta = bundle.get("meta", {})
        if bundle.get("features") != FEATURE_NAMES:
            raise ValueError(
                "This model was trained on a different feature set. FEATURE_NAMES has "
                "changed since it was saved - retrain, or check that new features were "
                "appended at the end rather than inserted."
            )

    @staticmethod
    def save(model, classes: list[str], out_dir: str | Path, meta: dict | None = None) -> Path:
        """Save the classifier. `classes` is IGNORED if the model knows its own order.

        This matters more than it looks. `predict_proba` returns columns in the
        MODEL's class order - sklearn sorts them alphabetically - which is not
        the order you happened to list them in. Indexing your own list with
        sklearn's column index silently returns the wrong label for every
        prediction, while `predict()` keeps working, so the training report
        looks perfect and the deployed app is nonsense. Found exactly that way.
        """
        import joblib
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        model_classes = getattr(model, "classes_", None)
        if model_classes is None and hasattr(model, "steps"):       # a Pipeline
            model_classes = getattr(model.steps[-1][1], "classes_", None)
        if model_classes is None:
            model_classes = getattr(model, "class_labels", None)    # AutoGluon
        classes = [str(c) for c in model_classes] if model_classes is not None else list(classes)

        joblib.dump({"model": model, "classes": classes,
                     "features": FEATURE_NAMES, "meta": meta or {}},
                    out / "model.joblib")
        with open(out / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"classes": classes, "features": FEATURE_NAMES, **(meta or {})},
                      f, indent=2, default=str)
        return out / "model.joblib"

    # --- inference -------------------------------------------------------- #
    def predict_kind(self, pts_mm: np.ndarray) -> tuple[PrimitiveKind, float]:
        x = np.array([[extract(pts_mm)[k] for k in FEATURE_NAMES]], dtype=np.float64)
        if hasattr(self.model, "predict_proba"):
            proba = np.asarray(self.model.predict_proba(x))[0]
            i = int(proba.argmax())
            return self.classes_[i], float(proba[i])
        pred = self.model.predict(x)[0]
        return PrimitiveKind(pred), 1.0

    def recognize(self, stroke: Stroke) -> Primitive:
        if self.model is None:
            p = self.fallback.recognize(stroke)
            return Primitive(p.kind, p.points, p.params, p.confidence, source="rules")

        pts = prepare(stroke)
        if pts is None:
            return Primitive(PrimitiveKind.POLYLINE, np.zeros((0, 2)),
                             confidence=0.0, source="ml")

        kind, conf = self.predict_kind(pts)

        # A closed stroke cannot be a line, and an open one is not a circle.
        # Geometry beats the classifier on questions geometry can answer outright.
        closed = is_closed(pts, CLOSE_TOL_MM)
        if closed and kind is PrimitiveKind.LINE:
            kind, conf = PrimitiveKind.POLYLINE, min(conf, LOW_CONFIDENCE)
        if not closed and kind is PrimitiveKind.CIRCLE:
            kind, conf = PrimitiveKind.ARC, min(conf, LOW_CONFIDENCE)

        points, params = fit_params(kind, pts)
        return Primitive(kind=kind, points=points, params=params,
                         confidence=conf, source="ml")
