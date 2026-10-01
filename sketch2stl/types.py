"""The data contract. This file is the interface between Serena's half and PK's half.

Everything upstream of `Profile` is stroke processing and recognition (Serena).
Everything downstream of `Profile` is geometry (PK).

REVISION 2026-09-29 (proposed by Serena, for PK to confirm before either of us
builds on it). Adds revolve to the feature vocabulary and a Contour type for
shapes drawn in several strokes. Everything existing keeps working: the new
fields all have defaults, so a Feature built the old way is still an extrude.

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


class FeatureKind(str, Enum):
    """HOW a profile becomes a volume. New in the 29 Sep revision.

    EXTRUDE         sweep the profile along its normal by `depth`
    REVOLVE         spin the profile around `axis` by `angle`
    MIRROR_EXTRUDE  mirror the half-profile across `axis`, then extrude

    The third one matters because a half-profile is ambiguous by construction.
    A half rectangle against a centreline is a cylinder if revolved and a block
    if mirrored and extruded, and nothing in the drawing distinguishes them.
    That is not a modelling failure to be engineered away - it is exactly where
    the human gate belongs, so the UI shows both and the person chooses.
    """
    EXTRUDE = "extrude"
    REVOLVE = "revolve"
    MIRROR_EXTRUDE = "mirror_extrude"


@dataclass(frozen=True)
class Axis:
    """A line in the sketch plane, in millimetres.

    For a REVOLVE this is the axis of revolution; for a MIRROR_EXTRUDE it is the
    mirror line. In "half + centreline" mode the user draws it, so there is no
    axis to infer and no axis error to make - which is the whole reason for
    that mode.

    `point` is any point on the line, `direction` a unit vector along it.
    """
    point: np.ndarray
    direction: np.ndarray

    def __post_init__(self) -> None:
        p = np.asarray(self.point, dtype=np.float64).reshape(-1)
        d = np.asarray(self.direction, dtype=np.float64).reshape(-1)
        if p.shape != (2,) or d.shape != (2,):
            raise ValueError("Axis.point and .direction must both be (2,)")
        n = float(np.linalg.norm(d))
        if n < 1e-9:
            raise ValueError("Axis.direction must not be zero length")
        object.__setattr__(self, "point", p)
        object.__setattr__(self, "direction", d / n)

    @classmethod
    def from_points(cls, a, b) -> "Axis":
        a = np.asarray(a, dtype=np.float64).reshape(2)
        b = np.asarray(b, dtype=np.float64).reshape(2)
        return cls(a, b - a)

    def signed_distance(self, points: np.ndarray) -> np.ndarray:
        """Perpendicular distance to the axis, signed by which side."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        rel = pts - self.point
        return rel[:, 0] * self.direction[1] - rel[:, 1] * self.direction[0]

    def crosses(self, points: np.ndarray, tol: float = 1e-6) -> bool:
        """Does this path straddle the axis? A revolve profile must not."""
        d = self.signed_distance(points)
        return bool((d > tol).any() and (d < -tol).any())


class Op(str, Enum):
    """What a feature does to the running solid.

    Op is WHAT a feature does to the running solid; FeatureKind is HOW the
    volume was made. They are independent: you can revolve-and-cut a groove
    just as easily as extrude-and-cut a hole.

    Still only two operations. Sweep and loft remain out of scope.
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

    # --- added 29 Sep. Defaults keep every existing Feature an extrude. ---
    kind: FeatureKind = FeatureKind.EXTRUDE
    axis: Axis | None = None
    angle: float = 2.0 * np.pi      # revolve sweep in radians; ignored by extrude

    def __post_init__(self) -> None:
        if self.kind is FeatureKind.EXTRUDE:
            return
        if self.axis is None:
            raise ValueError(f"{self.kind.value} needs an axis")
        # A revolve profile that straddles its own axis sweeps through itself
        # and produces a self-intersecting solid. Catch it here, at the
        # contract, rather than three layers down inside trimesh.
        if self.axis.crosses(self.profile.outer):
            raise ValueError(
                f"{self.kind.value} profile crosses its axis. A half profile "
                f"must lie entirely on one side of the centreline.")
        if self.kind is FeatureKind.REVOLVE and not (0 < self.angle <= 2 * np.pi + 1e-9):
            raise ValueError(f"revolve angle must be in (0, 2pi], got {self.angle}")


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
