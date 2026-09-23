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
from .types import Document, Feature, Op, Profile

Step = Literal["draw", "choose_op", "set_depth", "review"]


@dataclass
class Session:
    doc: Document = field(default_factory=Document)
    step: Step = "draw"
    pending_profile: Profile | None = None
    pending_op: Op | None = None
    _undo: list[list[Feature]] = field(default_factory=list)
    _redo: list[list[Feature]] = field(default_factory=list)

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
               name: str | None = None) -> Feature:
        if self.pending_profile is None or self.pending_op is None:
            raise ValueError("Nothing to commit - draw a shape and pick add or cut.")
        self._snapshot()
        feat = Feature(
            feature_id=uuid.uuid4().hex[:8],
            name=name or self._auto_name(self.pending_op),
            op=self.pending_op,
            profile=self.pending_profile,
            depth=float(depth),
            z_base=float(z_base),
        )
        self.doc.add(feat)
        self.pending_profile = None
        self.pending_op = None
        self.step = "draw"
        return feat

    def _auto_name(self, op: Op) -> str:
        n = sum(1 for f in self.doc.features if f.op is op) + 1
        return f"{'Add' if op is Op.ADD else 'Cut'} {n}"

    # --- history ----------------------------------------------------------- #
    def _snapshot(self) -> None:
        self._undo.append(list(self.doc.features))
        self._redo.clear()

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(list(self.doc.features))
        self.doc.features = self._undo.pop()
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(list(self.doc.features))
        self.doc.features = self._redo.pop()
        return True

    # --- rebuild ----------------------------------------------------------- #
    def solid(self) -> tuple[trimesh.Trimesh | None, str | None]:
        """Returns (mesh, error_message). Never raises - the UI shows the message."""
        try:
            return build(self.doc), None
        except KernelError as exc:
            return None, str(exc)
