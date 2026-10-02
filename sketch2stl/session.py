"""Application state: the document, undo/redo, and the step-by-step flow.

OWNER: shared - Serena owns the step machine, PK owns the rebuild path.
Keep edits to separate methods and you will not collide.

The mockup's flow is: draw a shape -> choose ADD or CUT -> set a depth -> it
appears in the layer list. `Session` is that flow with no Gradio in it, which
means the whole interaction is testable without launching a UI.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal

import trimesh

from .config import DEFAULT_DEPTH_MM
from .kernel import KernelError, build
from .types import Axis, Document, Feature, FeatureKind, Op, Profile

Step = Literal["draw", "choose_op", "set_depth", "review"]


@dataclass
class Session:
    doc: Document = field(default_factory=Document)
    step: Step = "draw"
    pending_profile: Profile | None = None
    pending_op: Op | None = None

    # --- ink already turned into features, per canvas ---------------------- #
    #
    # WHY THIS EXISTS. The app used to clear the canvas after every build by
    # returning a fresh image to the Sketchpad. That is what left the canvas
    # sitting on a loading animation: any component in an event's output list
    # is marked pending when the click is dispatched, and a Sketchpad that does
    # not resolve cleanly stays that way until some other event completes.
    #
    # So the canvas is no longer written to AT ALL. Instead the ink that has
    # already been committed is remembered here, and the next read processes
    # only what is new. The user's earlier strokes stay on the canvas, which is
    # also a better answer to "show me where I drew the last one" than the pale
    # ghost was - it is the actual drawing rather than a rendering of it.
    #
    # Keyed by canvas ("free" / "half") so switching modes does not retire ink
    # on the other one.
    consumed: dict = field(default_factory=dict)

    # Corners clicked in click-to-place mode, in canvas pixels, per canvas.
    clicked: dict = field(default_factory=dict)

    _undo: list[tuple[list[Feature], dict]] = field(default_factory=list)
    _redo: list[tuple[list[Feature], dict]] = field(default_factory=list)

    # --- ink bookkeeping --------------------------------------------------- #
    def new_ink(self, key: str, mask):
        """The part of `mask` that has not already been built.

        Also forgets any consumed ink that is no longer on the canvas, so the
        Sketchpad's own bin icon and the eraser both just work: rub a shape out
        and it stops being remembered.
        """
        import numpy as np
        if mask is None:
            self.consumed.pop(key, None)
            return None
        done = self.consumed.get(key)
        if done is None or done.shape != mask.shape:
            return mask
        done = done & mask
        self.consumed[key] = done
        return mask & ~done

    def retire_ink(self, key: str, mask) -> None:
        """Everything currently on this canvas has now been dealt with."""
        if mask is not None:
            self.consumed[key] = mask.copy()

    # --- the step machine -------------------------------------------------- #
    def submit_profile(self, profile: Profile) -> Step:
        self.pending_profile = profile
        self.step = "choose_op"
        return self.step

    def choose_op(self, op: Op) -> Step:
        if self.pending_profile is None:
            raise ValueError("Draw a shape first.")
        self.pending_op = op
        self.step = "set_depth"
        return self.step

    def commit(self, depth: float = DEFAULT_DEPTH_MM, z_base: float = 0.0,
               name: str | None = None, kind: FeatureKind = FeatureKind.EXTRUDE,
               axis: Axis | None = None, angle: float | None = None) -> Feature:
        """Turn the pending profile into a feature.

        `kind`/`axis`/`angle` were added on 29 Sep for revolve. They default to
        a plain extrude, so every existing caller is unaffected.
        """
        if self.pending_profile is None or self.pending_op is None:
            raise ValueError("Nothing to commit - draw a shape and pick add or cut.")
        self._snapshot()
        extra = {} if angle is None else {"angle": float(angle)}
        feat = Feature(
            feature_id=uuid.uuid4().hex[:8],
            name=name or self._auto_name(self.pending_op),
            op=self.pending_op,
            profile=self.pending_profile,
            depth=float(depth),
            z_base=float(z_base),
            kind=kind,
            axis=axis,
            **extra,
        )
        self.doc.add(feat)
        self.pending_profile = None
        self.pending_op = None
        self.step = "draw"
        return feat

    def commit_many(self, profiles: list[Profile], op: Op, depth: float = DEFAULT_DEPTH_MM,
                    z_base: float = 0.0, kind: FeatureKind = FeatureKind.EXTRUDE,
                    axis: Axis | None = None) -> list[Feature]:
        """Several shapes drawn in one go -> several features, ONE undo step."""
        if not profiles:
            raise ValueError("Nothing to commit - draw a shape first.")
        self._snapshot()
        made = []
        for prof in profiles:
            feat = Feature(feature_id=uuid.uuid4().hex[:8], name=self._auto_name(op), op=op,
                           profile=prof, depth=float(depth), z_base=float(z_base),
                           kind=kind, axis=axis)
            self.doc.add(feat)
            made.append(feat)
        self.pending_profile = self.pending_op = None
        self.step = "draw"
        return made

    # --- the layer panel: edit any feature, not just the last ---------------- #
    # The feature list IS the model and the solid is rebuilt from it every time,
    # so editing an old layer is just replacing it in the list. Each edit is one
    # undo step.
    def _index(self, feature_id: str) -> int:
        for i, f in enumerate(self.doc.features):
            if f.feature_id == feature_id:
                return i
        raise KeyError(feature_id)

    def edit_feature(self, feature_id: str, **changes) -> Feature:
        from dataclasses import replace
        i = self._index(feature_id)
        new = replace(self.doc.features[i], **changes)
        self._snapshot()
        feats = list(self.doc.features); feats[i] = new; self.doc.features = feats
        return new

    def delete_feature(self, feature_id: str) -> None:
        self._index(feature_id)
        self._snapshot()
        self.doc.features = [f for f in self.doc.features if f.feature_id != feature_id]

    def move_feature(self, feature_id: str, delta: int) -> bool:
        i = self._index(feature_id)
        j = i + delta
        if not 0 <= j < len(self.doc.features):
            return False
        self._snapshot()
        feats = list(self.doc.features); feats[i], feats[j] = feats[j], feats[i]
        self.doc.features = feats
        return True

    def _auto_name(self, op: Op) -> str:
        n = sum(1 for f in self.doc.features if f.op is op) + 1
        return f"{'Add' if op is Op.ADD else 'Cut'} {n}"

    # --- history ----------------------------------------------------------- #
    def _snapshot(self) -> None:
        # The consumed-ink map travels with the feature list, so undoing a
        # build makes its strokes count as new again and pressing Add re-adds
        # them. Otherwise undo would leave a drawing on the canvas that the app
        # refused to look at.
        self._undo.append(self._state())
        self._redo.clear()

    def _state(self) -> tuple:
        return (list(self.doc.features), dict(self.consumed),
                {k: list(v) for k, v in self.clicked.items()})

    def _restore(self, state: tuple) -> None:
        self.doc.features, self.consumed, self.clicked = state

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._state())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._state())
        self._restore(self._redo.pop())
        return True

    # --- rebuild ----------------------------------------------------------- #
    def solid(self) -> tuple[trimesh.Trimesh | None, str | None]:
        """Returns (mesh, error_message). Never raises - the UI shows the message."""
        try:
            return build(self.doc), None
        except KernelError as exc:
            return None, str(exc)
