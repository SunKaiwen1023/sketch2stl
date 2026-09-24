"""Parser tests built from a REAL Fusion 360 Gallery record.

The fixtures below are copied from an actual r1.0.1 file, not invented. The
first version of this suite used made-up inline coordinates, so it passed while
the parser crashed on the very first real model with

    AttributeError: 'str' object has no attribute 'get'

Two things that fixture got wrong, both represented here now:
  * `curves` references points by UUID into a separate `points` table
  * `profiles -> loops -> profile_curves` carries inline coordinates instead
"""
import json

import numpy as np
import pytest

from sketch2stl.data.fusion360 import (curves_from_model, load_split,
                                       operations_from_model)
from sketch2stl.types import PrimitiveKind

PT = lambda x, y: {"type": "Point3D", "x": x, "y": y, "z": 0.0}       # noqa: E731

# a sketch whose plane is the world YZ plane - the case that exposed the
# bogus projection, since its z_axis is (1,0,0) while curve normals are (0,0,1)
TRANSFORM = {
    "origin": PT(0.0, 0.0),
    "x_axis": {"type": "Vector3D", "x": 0.0, "y": 0.0, "z": -1.0, "length": 1.0},
    "y_axis": {"type": "Vector3D", "x": 0.0, "y": 1.0, "z": 0.0, "length": 1.0},
    "z_axis": {"type": "Vector3D", "x": 1.0, "y": 0.0, "z": 0.0, "length": 1.0},
}

MODEL = {
    "entities": {
        "sk1": {
            "type": "Sketch",
            "name": "Sketch1",
            "transform": TRANSFORM,
            "points": {
                "uuid-centre": PT(4.0, 3.0),
                "uuid-a": PT(0.0, 0.0),
                "uuid-b": PT(8.0, 0.0),
            },
            "curves": {
                "c1": {"type": "SketchCircle", "center_point": "uuid-centre",
                       "radius": 1.5, "construction_geom": False},
                "c2": {"type": "SketchLine", "start_point": "uuid-a",
                       "end_point": "uuid-b", "construction_geom": False},
                "c3": {"type": "SketchLine", "start_point": "uuid-a",
                       "end_point": "uuid-b", "construction_geom": True},
            },
            "profiles": {
                "p1": {"loops": [
                    {"is_outer": True, "profile_curves": [
                        {"type": "Circle3D", "center_point": PT(4.0, 3.0), "radius": 1.5,
                         "normal": {"type": "Vector3D", "x": 0.0, "y": 0.0, "z": 1.0}},
                    ]},
                    {"is_outer": False, "profile_curves": [
                        {"type": "Circle3D", "center_point": PT(4.0, 3.0), "radius": 0.5,
                         "normal": {"type": "Vector3D", "x": 0.0, "y": 0.0, "z": 1.0}},
                    ]},
                ]},
            },
        },
        "e1": {"type": "ExtrudeFeature", "operation": "NewBodyFeatureOperation"},
        "e2": {"type": "ExtrudeFeature", "operation": "CutFeatureOperation"},
    },
    "timeline": [{"entity": "sk1"}, {"entity": "e1"}, {"entity": "e2"}],
}

# the same sketch with no profiles, forcing the UUID-resolving fallback path
CURVES_ONLY = json.loads(json.dumps(MODEL))
CURVES_ONLY["entities"]["sk1"]["profiles"] = {}


def test_profile_curves_parse_and_keep_the_outer_flag():
    curves = curves_from_model(MODEL, "m1")
    assert len(curves) == 2
    outer = [c for c in curves if c.params["is_outer"] == 1.0]
    inner = [c for c in curves if c.params["is_outer"] == 0.0]
    assert len(outer) == 1 and len(inner) == 1
    assert abs(outer[0].params["r"] - 15.0) < 1e-6      # 1.5 cm -> 15 mm
    assert abs(inner[0].params["r"] - 5.0) < 1e-6


def test_uuid_point_references_resolve():
    """The bug that crashed on the first real file: center_point is a string."""
    curves = curves_from_model(CURVES_ONLY, "m1")
    kinds = {c.kind for c in curves}
    assert PrimitiveKind.CIRCLE in kinds and PrimitiveKind.LINE in kinds

    circ = next(c for c in curves if c.kind is PrimitiveKind.CIRCLE)
    assert abs(circ.params["cx"] - 40.0) < 1e-6        # 4 cm -> 40 mm
    assert abs(circ.params["cy"] - 30.0) < 1e-6


def test_coordinates_are_sketch_local_not_projected():
    """This sketch's z_axis is (1,0,0); projecting would move the centre to (0,0)."""
    circ = next(c for c in curves_from_model(CURVES_ONLY) if c.kind is PrimitiveKind.CIRCLE)
    assert (abs(circ.params["cx"] - 40.0) < 1e-6 and abs(circ.params["cy"] - 30.0) < 1e-6), \
        "coordinates were projected onto the sketch basis - see _to_2d"


def test_construction_geometry_is_skipped():
    lines = [c for c in curves_from_model(CURVES_ONLY) if c.kind is PrimitiveKind.LINE]
    assert len(lines) == 1, "the construction_geom line should not be a training sample"


def test_line_length_converts_cm_to_mm():
    line = next(c for c in curves_from_model(CURVES_ONLY) if c.kind is PrimitiveKind.LINE)
    assert abs(np.linalg.norm(line.points[-1] - line.points[0]) - 80.0) < 1e-6


def test_circle_ring_is_closed():
    circ = next(c for c in curves_from_model(MODEL) if c.kind is PrimitiveKind.CIRCLE)
    assert np.allclose(circ.points[0], circ.points[-1])


def test_operations_map_to_add_and_cut():
    assert operations_from_model(MODEL) == ["add", "cut"]


def test_empty_and_malformed_models_do_not_raise():
    assert curves_from_model({}) == []
    assert curves_from_model({"entities": {"X": {"type": "Sketch"}}}) == []
    assert curves_from_model({"entities": {"S": {"type": "Sketch",
                                                 "curves": {"B": {"type": "SketchLine"}}}}}) == []
    # a dangling UUID must be skipped, not crash
    bad = {"entities": {"S": {"type": "Sketch", "points": {},
                              "curves": {"c": {"type": "SketchCircle",
                                               "center_point": "missing", "radius": 1.0}}}}}
    assert curves_from_model(bad) == []


def test_load_split(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps({"train": ["a", "b"], "test": ["c"]}))
    s = load_split(p)
    assert s["train"] == ["a", "b"] and s["test"] == ["c"]
