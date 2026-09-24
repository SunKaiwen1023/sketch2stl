import json

import numpy as np
import pytest

from sketch2stl.data.fusion360 import (curves_from_model, load_split,
                                       operations_from_model)
from sketch2stl.types import PrimitiveKind

MODEL = {
    "entities": {
        "S1": {
            "type": "Sketch",
            "transform": {"origin": {"x": 0, "y": 0, "z": 0},
                          "x_axis": {"x": 1, "y": 0, "z": 0},
                          "y_axis": {"x": 0, "y": 1, "z": 0}},
            "curves": {
                "L1": {"type": "Line3D", "start_point": {"x": 0, "y": 0, "z": 0},
                       "end_point": {"x": 8, "y": 0, "z": 0}},
                "C1": {"type": "Circle3D", "center_point": {"x": 4, "y": 3, "z": 0},
                       "radius": 1.5},
                "A1": {"type": "Arc3D", "center_point": {"x": 1, "y": 1, "z": 0},
                       "radius": 2.0, "start_angle": 0.0, "end_angle": 1.5},
            },
        },
        "E1": {"type": "ExtrudeFeature", "operation": "NewBodyFeatureOperation"},
        "E2": {"type": "ExtrudeFeature", "operation": "CutFeatureOperation"},
    },
    "timeline": [{"entity": "S1"}, {"entity": "E1"}, {"entity": "E2"}],
}


def test_all_three_curve_types_parse():
    kinds = {c.kind for c in curves_from_model(MODEL, "m1")}
    assert kinds == {PrimitiveKind.LINE, PrimitiveKind.CIRCLE, PrimitiveKind.ARC}


def test_units_are_converted_from_cm_to_mm():
    line = next(c for c in curves_from_model(MODEL) if c.kind is PrimitiveKind.LINE)
    assert abs(np.linalg.norm(line.points[-1] - line.points[0]) - 80.0) < 1e-6

    circ = next(c for c in curves_from_model(MODEL) if c.kind is PrimitiveKind.CIRCLE)
    assert abs(circ.params["r"] - 15.0) < 1e-6


def test_circle_ring_is_closed():
    circ = next(c for c in curves_from_model(MODEL) if c.kind is PrimitiveKind.CIRCLE)
    assert np.allclose(circ.points[0], circ.points[-1])


def test_operations_map_to_add_and_cut():
    assert operations_from_model(MODEL) == ["add", "cut"]


def test_empty_and_malformed_models_do_not_raise():
    assert curves_from_model({}) == []
    assert curves_from_model({"entities": {"X": {"type": "Sketch"}}}) == []
    assert curves_from_model({"entities": {"S": {"type": "Sketch",
                                                 "curves": {"B": {"type": "Line3D"}}}}}) == []


def test_load_split(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"train": ["a", "b"], "test": ["c"]}))
    s = load_split(p)
    assert s["train"] == ["a", "b"] and s["test"] == ["c"]
