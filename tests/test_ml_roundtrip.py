"""The trained model must agree with itself after a save/load round trip.

This suite exists because of a real bug: `predict_proba` returns columns in the
MODEL's class order (sklearn sorts alphabetically), not the order the labels
were listed in. Indexing a hand-written list with sklearn's column index gives
the wrong label for every prediction - while `predict()` keeps working, so the
training report looked perfect and the deployed app misread every shape.
"""
import numpy as np
import pytest

from sketch2stl.recognizer.features import FEATURE_NAMES, extract
from sketch2stl.recognizer.ml import MLRecognizer
from sketch2stl.strokes import mm_to_px
from sketch2stl.types import PrimitiveKind, Stroke

rng = np.random.default_rng(20260922)


def shapes(n=40):
    """A tiny labelled set: circles and rectangles, deliberately unalphabetical."""
    X, y = [], []
    for _ in range(n):
        r = rng.uniform(8, 25)
        a = np.linspace(0, 2 * np.pi, 64)
        c = np.column_stack([80 + r * np.cos(a), 60 + r * np.sin(a)])
        X.append(c + rng.normal(0, 0.3, c.shape)); y.append("circle")

        w, h = rng.uniform(20, 60), rng.uniform(20, 60)
        ring = []
        for (x0, y0), (x1, y1) in [((0, 0), (w, 0)), ((w, 0), (w, h)),
                                   ((w, h), (0, h)), ((0, h), (0, 0))]:
            for s in np.linspace(0, 1, 16, endpoint=False):
                ring.append([80 - w / 2 + x0 + (x1 - x0) * s, 60 - h / 2 + y0 + (y1 - y0) * s])
        ring = np.asarray(ring)
        X.append(ring + rng.normal(0, 0.3, ring.shape)); y.append("rect")
    return X, np.array(y)


def test_saved_model_predicts_the_same_labels_as_the_raw_model(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    X, y = shapes()
    F = np.array([[extract(s)[k] for k in FEATURE_NAMES] for s in X])
    model = RandomForestClassifier(n_estimators=40, random_state=7).fit(F, y)

    # note the order here is NOT sklearn's alphabetical order - that is the point
    MLRecognizer.save(model, ["rect", "circle"], tmp_path)

    rec = MLRecognizer(tmp_path)
    for stroke_mm, want in zip(X, y):
        raw = model.predict([[extract(stroke_mm)[k] for k in FEATURE_NAMES]])[0]
        got, _ = rec.predict_kind(stroke_mm)
        assert got.value == raw, f"round trip disagrees: {got.value} vs {raw}"


def test_it_actually_recognises_the_shapes_it_trained_on(tmp_path):
    from sklearn.ensemble import RandomForestClassifier

    X, y = shapes()
    F = np.array([[extract(s)[k] for k in FEATURE_NAMES] for s in X])
    MLRecognizer.save(RandomForestClassifier(n_estimators=40, random_state=7).fit(F, y),
                      ["rect", "circle"], tmp_path)
    rec = MLRecognizer(tmp_path)

    correct = sum(rec.predict_kind(s)[0].value == want for s, want in zip(X, y))
    assert correct / len(X) > 0.9, f"only {correct}/{len(X)} right on its own training data"


def test_no_model_falls_back_to_rules_instead_of_crashing():
    rec = MLRecognizer(None)
    a = np.linspace(0, 2 * np.pi, 80)
    circle = np.column_stack([80 + 15 * np.cos(a), 60 + 15 * np.sin(a)])
    p = rec.recognize(Stroke(points=mm_to_px(circle)))
    assert p.kind is PrimitiveKind.CIRCLE
    assert p.source == "rules"


def test_feature_set_change_is_caught_loudly(tmp_path):
    import joblib
    joblib.dump({"model": None, "classes": ["circle"], "features": ["only_one"], "meta": {}},
                tmp_path / "model.joblib")
    with pytest.raises(ValueError, match="different feature set"):
        MLRecognizer(tmp_path)
