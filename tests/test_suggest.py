"""The suggester: add-or-cut, revolve-or-extrude, and how sure it is.

These are BASELINE tests. They will still be the tests when PK's model replaces
the rules behind `Suggestion`, because they check the behaviour the UI depends
on rather than the mechanism: a value, a calibrated-ish confidence, and an
honest `unsure` flag when the drawing genuinely does not decide.

The confidence numbers matter as much as the answers. A suggester that is 95%
sure and wrong is worse than one that says it does not know, because the person
stops reading it.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from sketch2stl import suggest
from sketch2stl.suggest import Suggestion, suggest_kind, suggest_op
from sketch2stl.types import Axis, Op, Profile

Y_AXIS = Axis.from_points([0.0, 0.0], [0.0, 1.0])


@pytest.fixture(autouse=True)
def _no_trained_model(monkeypatch):
    """Pin the rules arm unless a test explicitly asks for the model.

    Without this the whole file changes behaviour depending on whether somebody
    has run scripts/train_addcut.py, so it passes on one laptop and fails on the
    other. Which arm answers is a property of the deployment, not of the code
    under test, so each test has to say which one it means.
    """
    stub = lambda *a, **k: None                      # noqa: E731 - a stand-in
    stub.cache_clear = lambda: None
    monkeypatch.setattr(suggest, "load_addcut_model", stub)


def rect(cx, cy, w, h):
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    return Profile(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]))


def circle(cx, cy, r, n=64):
    a = np.linspace(0, 2 * np.pi, n)
    ring = np.column_stack([cx + r * np.cos(a), cy + r * np.sin(a)])
    return Profile(np.vstack([ring, ring[:1]]))


# --------------------------------------------------------------------------- #
# Suggestion itself
# --------------------------------------------------------------------------- #

def test_unsure_is_the_flag_the_ui_branches_on():
    assert Suggestion("add", 0.59, "").unsure
    assert not Suggestion("add", 0.61, "").unsure


def test_every_suggestion_carries_a_reason():
    """The reason is shown to the user. A suggestion without one is a machine
    telling someone what to do, which is the opposite of the point."""
    for sug in (suggest_op(rect(0, 0, 10, 10), []),
                suggest_op(circle(0, 0, 3), [rect(0, 0, 40, 40)]),
                suggest_kind(np.column_stack([np.full(40, 5.0),
                                              np.linspace(0, 20, 40)]), Y_AXIS)):
        assert sug.reason.strip()
        assert 0.0 <= sug.confidence <= 1.0


# --------------------------------------------------------------------------- #
# ADD or CUT
# --------------------------------------------------------------------------- #

def test_the_first_shape_is_always_an_add():
    sug = suggest_op(rect(0, 0, 40, 30), [])
    assert sug.value == Op.ADD.value
    assert not sug.unsure


def test_a_small_shape_inside_the_part_reads_as_a_cut():
    sug = suggest_op(circle(0, 0, 4), [rect(0, 0, 60, 40)])
    assert sug.value == Op.CUT.value
    assert not sug.unsure


def test_a_shape_beside_the_part_reads_as_new_material():
    sug = suggest_op(rect(100, 0, 20, 20), [rect(0, 0, 40, 40)])
    assert sug.value == Op.ADD.value
    assert not sug.unsure


def test_a_half_overlapping_shape_admits_it_does_not_know():
    """A notch and a wing look identical from here, so the UI must ask rather
    than pre-select. This is the case the confidence exists for."""
    sug = suggest_op(rect(20, 0, 40, 20), [rect(0, 0, 40, 40)])
    assert sug.unsure


def test_overlap_with_any_one_body_is_enough():
    """Two separate bodies: a hole through the second is still a hole."""
    sug = suggest_op(circle(100, 0, 3),
                     [rect(0, 0, 40, 40), rect(100, 0, 40, 40)])
    assert sug.value == Op.CUT.value


def test_a_degenerate_shape_does_not_crash_the_panel():
    flat = Profile(np.array([[0.0, 0], [10, 0], [20, 0], [0, 0]]))
    sug = suggest_op(flat, [rect(0, 0, 40, 40)])
    assert isinstance(sug, Suggestion)


# --------------------------------------------------------------------------- #
# EXTRUDE or REVOLVE
# --------------------------------------------------------------------------- #

def test_a_curved_half_reads_as_something_turned():
    th = np.linspace(-np.pi / 2, np.pi / 2, 120)
    arch = np.column_stack([25 * np.cos(th), 25 * np.sin(th)])
    sug = suggest_kind(arch, Y_AXIS)
    assert sug.value == "revolve"
    assert not sug.unsure


def test_a_half_rectangle_is_admitted_to_be_ambiguous():
    """Half a rectangle is a cylinder revolved and a block mirrored, and the
    drawing contains NOTHING that decides between them. Anything other than a
    low confidence here would be the suggester bluffing."""
    n = 40
    half = np.vstack([
        np.column_stack([np.zeros(n), np.linspace(0, 40, n)]),
        np.column_stack([np.linspace(0, 20, n), np.full(n, 40.0)]),
        np.column_stack([np.full(n, 20.0), np.linspace(40, 0, n)]),
    ])
    assert suggest_kind(half, Y_AXIS).unsure


def test_a_short_stroke_is_handled_not_crashed():
    sug = suggest_kind(np.array([[1.0, 1.0], [2.0, 2.0]]), Y_AXIS)
    assert isinstance(sug, Suggestion)
    assert sug.unsure


# --------------------------------------------------------------------------- #
# the learned arm
# --------------------------------------------------------------------------- #

class FakeModel:
    """Stands in for the trained classifier. Keeps the test about the WIRING."""
    def __init__(self, prob=(0.2, 0.8), explode=False):
        self.prob, self.explode = prob, explode

    def predict_proba(self, x):
        assert x.shape == (1, len(suggest.ADDCUT_FEATURES)), "wrong feature count"
        if self.explode:
            raise RuntimeError("corrupt model")
        return np.array([self.prob])



@pytest.fixture
def learned(monkeypatch):
    def install(prob=(0.2, 0.8), explode=False):
        monkeypatch.setattr(suggest, "load_addcut_model",
                            lambda *a, **k: {"model": FakeModel(prob, explode),
                                             "classes": ["add", "cut"],
                                             "features": suggest.ADDCUT_FEATURES})
    return install


def test_without_a_trained_model_the_rule_answers():
    sug = suggest_op(circle(0, 0, 4), [rect(0, 0, 60, 40)])
    assert sug.source == "rules"


def test_a_trained_model_answers_instead(learned):
    learned(prob=(0.1, 0.9))
    sug = suggest_op(circle(0, 0, 4), [rect(0, 0, 60, 40)])
    assert sug.source == "ml"
    assert sug.value == "cut"
    assert sug.confidence == pytest.approx(0.9)


def test_the_model_can_disagree_with_the_rule(learned):
    """The whole reason for having one. A cut into an existing hole overlaps
    nothing, so the rule says 'add' - the model is free to say otherwise."""
    learned(prob=(0.15, 0.85))
    assert suggest_op(rect(200, 0, 10, 10), [rect(0, 0, 40, 40)]).value == "cut"


def test_a_low_confidence_model_answer_is_still_flagged_unsure(learned):
    learned(prob=(0.45, 0.55))
    assert suggest_op(circle(0, 0, 4), [rect(0, 0, 60, 40)]).unsure


def test_a_broken_model_falls_back_instead_of_breaking_the_app(learned):
    """An assistant that takes the app down because a file is corrupt is worse
    than no assistant."""
    learned(explode=True)
    sug = suggest_op(circle(0, 0, 4), [rect(0, 0, 60, 40)])
    assert sug.source == "rules"
    assert sug.value == Op.CUT.value


def test_a_model_trained_on_different_features_is_refused(tmp_path, monkeypatch):
    monkeypatch.undo()               # this one exercises the real loader
    """Silently scoring a stale feature order would give confident nonsense."""
    joblib = pytest.importorskip("joblib")
    joblib.dump({"model": FakeModel(), "classes": ["add", "cut"],
                 "features": ["overlap_frac"]}, tmp_path / "model.joblib")
    suggest.load_addcut_model.cache_clear()
    with pytest.raises(ValueError, match="Retrain"):
        suggest.load_addcut_model(str(tmp_path))
    suggest.load_addcut_model.cache_clear()


def test_the_feature_vector_matches_what_the_training_script_writes():
    """These two lists are a contract across two files; drifting apart is the
    kind of bug that produces a model that works and is wrong."""
    script = (pathlib.Path(__file__).resolve().parents[1]
              / "scripts" / "train_addcut.py").read_text()
    for name in suggest.ADDCUT_FEATURES:
        assert f'"{name}"' in script, f"{name} missing from train_addcut.py"


def test_the_holes_in_a_profile_are_not_counted_as_material():
    """A ring drawn as one profile with an inner loop: a small shape inside the
    BORE overlaps nothing, even though it is inside the outer circle."""
    a = np.linspace(0, 2 * np.pi, 64)
    outer = np.column_stack([30 * np.cos(a), 30 * np.sin(a)])
    inner = np.column_stack([12 * np.cos(a), 12 * np.sin(a)])
    ring = Profile(np.vstack([outer, outer[:1]]), (np.vstack([inner, inner[:1]]),))
    assert suggest_op(circle(0, 0, 4), [ring]).value == Op.ADD.value


# --------------------------------------------------------------------------- #
# the baseline PK's model has to beat
# --------------------------------------------------------------------------- #

def test_the_add_cut_rule_beats_guessing_on_a_small_hand_set():
    """Documented so the comparison in the report is reproducible rather than
    a number I remembered. Six hand-labelled cases; the majority class scores
    3/6, the rule has to do better."""
    body = rect(0, 0, 60, 40)
    cases = [
        (circle(0, 0, 4), [body], Op.CUT),
        (circle(15, 10, 3), [body], Op.CUT),
        (rect(0, 0, 20, 10), [body], Op.CUT),
        (rect(120, 0, 20, 20), [body], Op.ADD),
        (rect(0, 60, 30, 20), [body], Op.ADD),
        (rect(0, 0, 40, 30), [], Op.ADD),
    ]
    right = sum(suggest_op(p, e).value == want.value for p, e, want in cases)
    assert right > len(cases) / 2, f"rule scored {right}/{len(cases)}, no better than guessing"
