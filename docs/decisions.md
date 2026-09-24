# Decisions log

Append-only. When you make a call that someone might question in three weeks — including yourselves —
write it down here. Never edit someone else's entry; add a new one that supersedes it.

Format: what was decided, when, by whom, why, and what it rules out.

---

## D1 — Extrusion only. No revolve, sweep or loft.

**2026-09-23 · PK, agreed by Serena**

Two operations (ADD, CUT) applied to extruded closed profiles. That's the whole modelling vocabulary.

**Why:** every hard problem in solid modelling lives outside this — sweeps along curves, lofts between
profiles, variable-radius fillets. Two operations and a depth is enough for the plate in the mockup and for
most printable parts, and it's achievable in three weeks by two people.

**Rules out:** lathed parts, anything organic. Say this up front in the presentation as a scope decision,
not a limitation discovered late.

---

## D2 — Gradio, not React.

**2026-09-23 · both**

**Why:** one language, no frontend build, deploys to HF Spaces in one command. Week 5 of the course covers
interactive UIs, so this is likely what gets taught. With ~2 weeks to a presentation, a React frontend
would consume one person entirely and the geometry would suffer.

**Cost:** the drawing canvas is a Gradio Sketchpad, which hands back a raster image rather than ordered pen
paths. See D5 — that's an open decision.

**Mitigation:** everything below `ui/` is plain Python with no Gradio import, so a React frontend can be
added later without touching the core.

---

## D3 — Rebuild the whole feature stack on every edit.

**2026-09-23 · PK**

`kernel.build(document)` is a pure function of the feature list.

**Why:** free undo, a preview that can't desync from the layer panel, and full testability with no UI.

**Cost:** O(n) rebuilds. Irrelevant at five features. If it bites, cache per-feature and rebuild from the
first change.

---

## D4 — Classify with ML, fit parameters with least squares.

**2026-09-23 · Serena**

The ML arm predicts the *shape class*. The circle's centre and radius come from `rules.fit_circle`, not
from a regression head.

**Why:** far more accurate on small data, and it means the rules arm and the ML arm share their geometry
code — so the evaluation isolates recognition instead of confounding it with parameter estimation.

---

## D5 — Vectorise the raster; the JS canvas stays on the shelf.

**2026-09-24 · implemented**

Gradio's Sketchpad returns an image, not pen paths. `ui/canvas.py` thresholds the ink, skeletonises it,
splits it into connected components and walks each into an ordered path.

**Cost:** stroke order and timing are gone. Clockwise and anticlockwise look identical, and there is no
speed signal.

**Why it's acceptable:** none of the twelve features in `recognizer/features.py` uses direction or time.
It works with stock Gradio, deploys to Spaces unchanged, and needs no JavaScript.

**Revisit if** recognition accuracy stalls and you have spare time. `strokes_from_json` is still there
and still works, so option (b) is a frontend job, not a rewrite.

---

## D8 — Training data is synthesised from Fusion 360 CAD, not hand-collected.

**2026-09-24 · Serena**

The Fusion 360 Gallery has 8,625 CAD models with perfectly labelled curves and no hand-drawn strokes.
`data/synth.py` distorts the clean curves into plausible hand-drawn ones; the label carries over free.

**Why:** tens of thousands of labelled strokes instead of 400, drawn from the shape distribution of real
engineering parts. It also unblocks D7 — the hand-drawn collection is no longer on the critical path.

**What it changes about D7:** hand-drawn strokes are now the TEST set, not the training set. That makes
them a domain-gap measurement, which is a better result and needs far fewer of them (~100 is plenty).

**Risk:** if the distortion model is unfaithful, the classifier learns a fantasy. Mitigated by
`synth.deviation()` — calibrate against real strokes and report both numbers.

---

# OPEN — decide these in your first hour and write the answers here



## D6 — How many shape classes?

Five (line, arc, circle, rect, polyline) is the current `PrimitiveKind`.

Fewer classes makes the rules baseline very strong and the ML arm look pointless. More classes needs more
training data than you'll realistically collect. Three (circle, rect, line) may be the honest scope — but
then say so and explain why.

**Decided:** _____________  **by:** _____________  **on:** _____________

---

## D7 — How many REAL strokes for the domain-gap test, and by when?

D8 removed this from the critical path — training data is synthetic now. What you still need is a
**test** set of real hand-drawn strokes: roughly 100 total, ~20 per class, both of you.

Record who drew each one. Put a date on it.

**Decided:** _____________  **by:** _____________  **on:** _____________
