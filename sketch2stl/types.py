"""The data contract. This file is the interface between Serena's half and PK's half.

Everything upstream of `Profile` is stroke processing and recognition (Serena).
Everything downstream of `Profile` is geometry (PK).

CHANGING ANYTHING IN THIS FILE BREAKS THE OTHER PERSON'S CODE.
Do not edit it on your own branch. Open an issue, agree, then one of you changes it
and both pull immediately. See docs/data_contract.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Literal, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Units
#
# Everything in this codebase is in MILLIMETRES in model space, and in PIXELS in
# canvas space. The conversion lives in exactly one place: config.PX_PER_MM.
# If you ever find yourself writing `* 0.1` or `/ 25.4`, stop - it belongs there.
# --------------------------------------------------------------------------- #


class PrimitiveKind(str, Enum):
    """What a single recognised stroke turned into."""
    LINE = "line"
    ARC = "arc"
    CIRCLE = "circle"
    RECT = "rect"
    POLYLINE = "polyline"      # fallback: we could not do better than the raw path


@dataclass(frozen=True)
class Stroke:
    """One raw pen-down-to-pen-up path, in canvas pixels.

    `points` is (N, 2) float32. N is whatever the canvas gave us - unresampled.
    `t` is optional per-point timestamps in seconds since stroke start; some
    recognisers use speed as a feature, most ignore it.
    """
    points: np.ndarray
    t: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.points.ndim != 2 or self.points.shape[1] != 2:
            raise ValueError(f"Stroke.points must be (N, 2), got {self.points.shape}")


@dataclass(frozen=True)
class Primitive:
    """A recognised shape, in MILLIMETRES, model space.

    `params` depends on `kind`:
      LINE     -> {"x1","y1","x2","y2"}
      CIRCLE   -> {"cx","cy","r"}
      ARC      -> {"cx","cy","r","a0","a1"}          angles in radians, CCW
      RECT     -> {"cx","cy","w","h","angle"}        angle in radians
      POLYLINE -> {}  (the geometry lives in `points`)

    `points` is always populated: a polygonal approximation of the primitive,
    (M, 2) in mm. PK's code can ignore `kind` entirely and just use `points`.
    That is deliberate - it means the geometry half never blocks on the ML half.

    `confidence` is 0..1 from the recogniser. The UI greys out anything below
    config.LOW_CONFIDENCE so the user knows to redraw.
    """
    kind: PrimitiveKind
    points: np.ndarray
    params: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0
    source: Literal["manual", "rules", "ml"] = "rules"


@dataclass(frozen=True)
class Profile:
    """A CLOSED region in the sketch plane, in millimetres. The thing you extrude.

    `outer` is (M, 2), closed (first point == last point), counter-clockwise.
    `holes` are inner rings, clockwise. Most of the time `holes` is empty,
    because in this app a hole is usually a separate CUT feature rather than an
    inner ring - but shapely gives them to us for free so the field exists.
    """
    outer: np.ndarray
    holes: tuple[np.ndarray, ...] = ()

    def __post_init__(self) -> None:
        if self.outer.ndim != 2 or self.outer.shape[1] != 2:
            raise ValueError(f"Profile.outer must be (M, 2), got {self.outer.shape}")
        if len(self.outer) < 4:
            raise ValueError("Profile.outer needs at least 4 points (3 + closing point)")


class Op(str, Enum):
    """What a feature does to the running solid.

    This is the whole modelling vocabulary. PK's constraint - "keep it at the
    extrusion level" - means we never add REVOLVE, SWEEP or LOFT. Two operations
    and a depth is enough to build the plate in the mockup.
    """
    ADD = "add"        # union: this volume becomes part of the body
    CUT = "cut"        # difference: this volume is removed from the body


@dataclass(frozen=True)
class Feature:
    """One row in the Features/Layers panel. An ordered list of these IS the model.

    `z_base` is where the extrusion starts, `depth` how far it goes, both in mm.
    A CUT with z_base=0 and depth greater than the plate thickness cuts all the
    way through, which is what you want for the corner holes in the mockup.

    `name` is what the user sees. `feature_id` is stable across edits so undo
    and the layer list can refer to it.
    """
    feature_id: str
    name: str
    op: Op
    profile: Profile
    depth: float
    z_base: float = 0.0
    visible: bool = True


@dataclass
class Document:
    """The whole project: an ordered feature stack, newest last.

    Rebuilding the solid is a pure function of this list, which is what makes
    undo trivial (drop the last feature and rebuild) and makes the app testable
    without any UI.
    """
    features: list[Feature] = field(default_factory=list)
    units: Literal["mm", "in"] = "mm"
    name: str = "Untitled Project"

    def add(self, feature: Feature) -> None:
        self.features.append(feature)

    def remove(self, feature_id: str) -> None:
        self.features = [f for f in self.features if f.feature_id != feature_id]

    def active(self) -> Sequence[Feature]:
        return [f for f in self.features if f.visible]
