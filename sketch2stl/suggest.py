"""Guess what the user meant, and say how sure we are.

OWNER: Serena (the interface), PK (the model behind it).

Two questions the app currently makes the person answer by hand:

    ADD or CUT?          is this shape more material, or a hole in it?
    EXTRUDE or REVOLVE?  in half mode, is this a block or a turned part?

Both are guessable. Neither is guessable RELIABLY, which is the whole point:
the suggestion arrives with a confidence, the UI pre-selects it, and the person
can override. That is the human gate, and a suggestion the user can ignore is
much cheaper to get wrong than a decision made for them.

THIS IS A BASELINE, NOT THE MODEL. Everything here is geometric rules. PK's
ML2 v2 (fine-tuned ResNet-18 on rendered profile + prior-part silhouette) drops
in behind `Suggestion`, and has to beat these numbers to earn its place:

    ADD/CUT     majority class on real CAD, steps after the first: 52.2%
                geometric rule ("overlaps the part -> cut"):       64.6%
    REVOLVE     majority class: unknown until PK's halves are cut, but 66% of
                the revolvable designs are plain cylinders whose half is a bare
                rectangle, so a model trained on the raw distribution will
                learn "rectangle -> revolve" and predict nothing useful.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .types import Axis, Op, Profile

# Where a trained add/cut model lives, if one has been trained. Built by
# scripts/extract_addcut.py + scripts/train_addcut.py; absent by default, and
# everything falls back to the rules below.
ADDCUT_PATH = os.environ.get("ADDCUT_PATH", "models/addcut")

# The feature vector, in the order the model was fitted on. It must match
# FEATURES in scripts/train_addcut.py exactly - `load_addcut_model` checks.
ADDCUT_FEATURES = ["step_index", "n_loops", "has_inner_loop", "is_circular",
                   "coplanar_prior", "log_area", "area_ratio", "overlap_frac",
                   "inside_frac", "centroid_dist"]


@dataclass(frozen=True)
class Suggestion:
    """What we think the user meant, and how sure we are.

    `confidence` is 0..1. Below about 0.6 the UI should present it as a genuine
    question rather than a default, because a confident wrong answer is worse
    than an obvious one.
    """
    value: str
    confidence: float
    reason: str
    source: Literal["rules", "ml"] = "rules"

    @property
    def unsure(self) -> bool:
        return self.confidence < 0.6


def _polygon(points: np.ndarray) -> Polygon | None:
    poly = Polygon(np.asarray(points, dtype=np.float64).reshape(-1, 2))
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly if (not poly.is_empty and poly.area > 0) else None


def _shape(profile: Profile) -> Polygon | None:
    """A Profile as one shapely polygon, holes punched out."""
    outer = _polygon(profile.outer)
    if outer is None:
        return None
    for hole in getattr(profile, "holes", ()) or ():
        h = _polygon(hole)
        if h is not None:
            outer = outer.difference(h)
    if not outer.is_valid:
        outer = outer.buffer(0)
    return outer if (not outer.is_empty and outer.area > 0) else None


# --------------------------------------------------------------------------- #
# The learned arm
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=1)
def load_addcut_model(path: str = ADDCUT_PATH):
    """The trained add/cut model, or None if nobody has trained one yet.

    Deliberately silent about a missing model - not having one is the normal
    state of this repo, and the app prints which arm it ended up with at
    startup. It is loud about a model it cannot trust: a file whose feature
    order does not match this module would score nonsense with total
    confidence, which is the one failure mode worth crashing over.
    """
    f = Path(path) / "model.joblib"
    if not f.exists():
        return None
    import joblib
    bundle = joblib.load(f)
    if list(bundle.get("features", [])) != ADDCUT_FEATURES:
        raise ValueError(
            f"{f} was trained on {bundle.get('features')} but this code builds "
            f"{ADDCUT_FEATURES}. Retrain with scripts/train_addcut.py.")
    return bundle


def addcut_arm(path: str = ADDCUT_PATH) -> str:
    """One line for the app header: which add/cut suggester is live."""
    try:
        return (f"learned model ({path})" if load_addcut_model(path)
                else "geometric rule (no trained model - see "
                     "scripts/train_addcut.py)")
    except Exception as exc:                         # noqa: BLE001
        return f"geometric rule (the model at {path} would not load: {exc})"


def addcut_features(new: Polygon, existing: list[Polygon],
                    n_loops: int, has_inner: bool, circular: bool) -> np.ndarray:
    """The same ten numbers `scripts/extract_addcut.py` writes per step.

    One honest difference from the training distribution, worth knowing when
    the numbers are compared: in this app everything is drawn on ONE plane, so
    `coplanar_prior` is 1 whenever anything has been built, whereas in the CAD
    data plenty of steps sit on a fresh plane. The model sees a narrower slice
    here than it was trained on.
    """
    area = new.area
    if existing:
        covered = unary_union(existing)
        overlap = new.intersection(covered).area / area
        inside = max(new.intersection(p).area for p in existing) / area
        near = min(existing, key=lambda p: new.centroid.distance(p.centroid))
        scale = float(np.sqrt(near.area)) or 1.0
        dist = new.centroid.distance(near.centroid) / scale
        ratio = area / max(p.area for p in existing)
    else:
        overlap = inside = 0.0
        dist = ratio = -1.0
    return np.array([[float(len(existing)), float(n_loops), float(has_inner),
                      float(circular), float(bool(existing)),
                      float(np.log1p(max(area, 0.0))), float(ratio),
                      float(overlap), float(inside), float(dist)]])


def _looks_circular(poly: Polygon) -> bool:
    """Isoperimetric ratio: 1.0 for a perfect circle, lower for anything else.

    A stand-in for the dataset's `is_circular`, which knows the curve types. It
    agrees on the cases that matter - a drawn hole is round - and the model
    treats it as one weak feature among ten.
    """
    if poly.length <= 0:
        return False
    return (4 * np.pi * poly.area) / (poly.length ** 2) > 0.92


# --------------------------------------------------------------------------- #
# ADD or CUT
# --------------------------------------------------------------------------- #

def suggest_op(new_profile: Profile,
               existing_profiles: list[Profile]) -> Suggestion:
    """Is this shape more material, or a hole in what is already there?

    Uses the trained model when one exists at ADDCUT_PATH, and the geometric
    rule otherwise. Both return the same `Suggestion`, so nothing downstream
    knows or cares which answered - that is the point of the object.

    THE RULE, which is also the baseline the model has to beat: how much of the
    new shape lands on top of the existing part. A hole is drawn INSIDE the
    thing it goes through; a boss or a new body is drawn beside it or on fresh
    canvas. Deliberately conservative at the edges - a shape that half overlaps
    is genuinely ambiguous, could be a notch or a wing, so the confidence drops
    and the UI asks rather than assumes.
    """
    new = _shape(new_profile)
    if new is None:
        return Suggestion(Op.ADD.value, 0.5, "Could not measure that shape.")

    if not existing_profiles:
        return Suggestion(Op.ADD.value, 0.95,
                          "Nothing to cut from yet, so this must be a new body.")

    others = [p for p in (_shape(q) for q in existing_profiles) if p is not None]
    overlap = (new.intersection(unary_union(others)).area / new.area) if others else 0.0

    learned = _suggest_op_ml(new, others, new_profile, overlap)
    if learned is not None:
        return learned

    if overlap >= 0.9:
        return Suggestion(Op.CUT.value, 0.6 + 0.35 * (overlap - 0.9) / 0.1,
                          f"{overlap:.0%} of this shape sits on the existing part, "
                          f"so it reads as a hole.")
    if overlap <= 0.1:
        return Suggestion(Op.ADD.value, 0.6 + 0.35 * (0.1 - overlap) / 0.1,
                          f"Only {overlap:.0%} of it touches the existing part, "
                          f"so it reads as new material.")
    return Suggestion(Op.CUT.value if overlap > 0.5 else Op.ADD.value,
                      0.4 + 0.2 * abs(overlap - 0.5) / 0.4,
                      f"{overlap:.0%} overlap - genuinely ambiguous. A notch and "
                      f"a wing look the same from here.")


def _suggest_op_ml(new: Polygon, others: list[Polygon], profile: Profile,
                   overlap: float) -> Suggestion | None:
    """The learned answer, or None to fall back to the rule.

    Falls back rather than raises. A suggester is an assistant, and an assistant
    that takes the app down because a model file is corrupt is worse than no
    assistant at all.
    """
    try:
        bundle = load_addcut_model()
    except Exception:                                # noqa: BLE001 - see above
        return None
    if bundle is None:
        return None

    try:
        holes = tuple(getattr(profile, "holes", ()) or ())
        x = addcut_features(new, others, 1 + len(holes), bool(holes),
                            _looks_circular(new))
        prob = bundle["model"].predict_proba(x)[0]
        i = int(np.argmax(prob))
        value = bundle["classes"][i]
        conf = float(prob[i])
    except Exception:                                # noqa: BLE001 - see above
        return None

    reason = (f"the model saw {overlap:.0%} of this shape on the existing part, "
              f"plus its size and position")
    if conf < 0.6:
        reason += " - and is not sure, so check it"
    return Suggestion(value, conf, reason, source="ml")


# --------------------------------------------------------------------------- #
# EXTRUDE or REVOLVE
# --------------------------------------------------------------------------- #

def suggest_kind(half_points: np.ndarray, axis: Axis) -> Suggestion:
    """In half mode: does this half want to be spun, or mirrored and extruded?

    A half profile is ambiguous BY CONSTRUCTION. Half a rectangle against a
    centreline is a cylinder if revolved and a block if mirrored - the drawing
    contains nothing that decides between them. So the rule only makes a
    confident call when the outline is curved, which is a shape people
    overwhelmingly draw when they mean something turned.
    """
    pts = np.asarray(half_points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 4:
        return Suggestion("extrude", 0.5, "Too few points to tell.")

    # Straightness: how far the outline strays from its own chords.
    from .corners import find_corners, turning_angles
    angles = turning_angles(pts)
    curved = float(np.mean((angles > 3) & (angles < 40)))   # gentle sustained turn
    corners = len(find_corners(pts))

    if curved > 0.45 and corners <= 2:
        conf = float(np.clip(0.6 + curved * 0.35, 0.0, 0.95))
        return Suggestion("revolve", conf,
                          f"The outline curves smoothly along {curved:.0%} of its "
                          f"length with {corners} sharp corner(s) - that is the "
                          f"shape of something turned.")
    if corners >= 3 and curved < 0.2:
        return Suggestion("extrude", 0.55,
                          f"{corners} sharp corners and almost no curvature. This "
                          f"could still be a stepped shaft, so check the preview.")
    return Suggestion("revolve", 0.45,
                      "This half is genuinely ambiguous - revolved it is a "
                      "cylinder, mirrored it is a block. Pick one and watch the "
                      "preview.")
