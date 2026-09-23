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

# OPEN — decide these in your first hour and write the answers here

## D5 — How do we get ordered strokes out of the canvas?

Gradio's Sketchpad returns an image, not pen paths.

- **(a) Vectorise the raster.** Threshold → skeletonise → trace. Works with stock Gradio. Loses stroke
  order and timing, adds a scikit-image dependency.
- **(b) Custom JS canvas** via `gr.HTML` posting JSON back. True ordered paths with timestamps — much
  better input for recognition — at the cost of ~80 lines of JavaScript.

*Recommendation: ship (a) to get end-to-end working this week, then attempt (b). The ML arm is far more
interesting with (b)'s data.*

**Decided:** _____________  **by:** _____________  **on:** _____________

---

## D6 — How many shape classes?

Five (line, arc, circle, rect, polyline) is the current `PrimitiveKind`.

Fewer classes makes the rules baseline very strong and the ML arm look pointless. More classes needs more
training data than you'll realistically collect. Three (circle, rect, line) may be the honest scope — but
then say so and explain why.

**Decided:** _____________  **by:** _____________  **on:** _____________

---

## D7 — How many strokes are we each collecting, and by when?

The ML arm needs data and it doesn't exist yet. 40 shapes × 5 classes × 2 people ≈ 400 samples.

Put a **date** on this. It's the item most likely to slip, and everything in Serena's half is blocked
behind it.

**Decided:** _____________  **by:** _____________  **on:** _____________
