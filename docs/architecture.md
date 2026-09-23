# Architecture

## The one-sentence version

An ordered list of extrude-and-boolean operations, rebuilt from scratch on every edit.

## Why "rebuild from scratch" is the right call here

`kernel.build(document)` is a **pure function**: same feature list in, same mesh out, no hidden state.

That single decision buys you:

- **Undo for free.** Drop the last feature, rebuild. No inverse operations to write.
- **A preview that cannot desync.** The 3D view and the layer panel are both derived from the same list.
- **Testability with no UI.** Every test in `tests/test_kernel.py` builds documents directly.
- **Editing any feature, not just the last one.** Change the depth of feature 1 and rebuild — it just works.

It is slower than incremental modelling. At five features nobody notices. If it becomes a problem, cache
the mesh after each feature and rebuild only from the first changed one (`kernel.py` TODO 4).

## The modelling vocabulary is deliberately tiny

Two operations: **ADD** (union) and **CUT** (difference). Both are "extrude a closed 2-D profile through a
depth". That's it.

This is PK's constraint and it is a good one. Every hard problem in solid modelling — sweeps along curves,
lofts between profiles, variable-radius fillets — lives outside this vocabulary. Two operations and a depth
is enough to build the plate in the mockup, and enough for most printable parts.

**Resist adding revolve before the presentation.** The demo is stronger with two operations that always
work than four that sometimes don't.

## 1. Stroke preprocessing — `strokes.py`

Raw canvas points → millimetres → resampled to a fixed 64 points by arc length.

Resampling is what makes strokes comparable. Two people drawing the same circle produce wildly different
point counts depending on how fast they moved; afterwards they produce the same shape sampled the same way.
Normalisation (centre at origin, scale to unit box) is applied only for the ML recogniser, so it learns
shape rather than size and position.

## 2. Recognition — `recognizer/`

Three interchangeable implementations behind one interface. **This is the evaluation section of the report,
so build all three.**

| Arm | What it is | What it tells you |
|---|---|---|
| `manual` | The user picks the tool first. No recognition. | The ceiling. Whatever the other two get, this is what they're reaching for. |
| `rules` | Least-squares fit each primitive type, keep the best residual. | The baseline. Cheap, explainable, no training data, often annoyingly good. |
| `ml` | A learned classifier on normalised strokes. | The thing that has to justify existing. |

**The finding that matters is not "our model gets 94%".** It's whether the ML arm beats a serious rules
baseline by enough to be worth the complexity, and on *which shapes*. A weak baseline makes the whole
comparison worthless, so build `rules` properly first.

**Classify with ML, fit parameters with rules.** Don't regress a circle's centre and radius with a network —
classify the *shape*, then call the matching least-squares fitter. Much more accurate, and it means both
arms share their geometry code, so the comparison isolates recognition rather than confounding it with
parameter estimation.

### Suggested features for the classical ML arm

Ten numbers per stroke is enough for a few hundred samples. This also makes HW2's `TabularPredictor` work
reusable almost line for line:

`closure_ratio` · `aspect_ratio` · `fill_ratio` (polygon area / bbox area — separates circle from rect) ·
`n_corners` (curvature peaks) · `circle_fit_residual` · `line_fit_residual` · `curvature_mean` ·
`curvature_std` · `total_turning_angle` · `length_to_bbox_diagonal`

### Splitting the stroke dataset

**Split by session, never randomly.** Whoever drew a stroke, on whatever day, with whatever input device —
that's the group. A random split lets the model learn *your handwriting* and report it as shape
recognition. This is the same `parent_id` grouping lesson from HW2, wearing a different hat.

## 3. Profiles — `profiles.py`

Primitives → closed, valid, extrudable regions.

Hand-drawn strokes are never clean loops: lines overshoot, corners don't meet, rings self-intersect near
the start point. Shapely's `buffer(0)` repairs a surprising amount of that for free. The biggest gap in the
scaffold is **stitching** — joining several open strokes into one loop — which is what a rectangle drawn as
four separate lines needs.

## 4. Kernel — `kernel.py`

Profiles + depths → one mesh, via `trimesh.creation.extrude_polygon` and `trimesh.boolean`.

**Cuts overshoot deliberately.** A CUT is extruded slightly taller than requested and started slightly
lower, so it always pokes out of both faces of the body. Coplanar faces are the number one cause of boolean
failures, and `config.CUT_OVERSHOOT_MM` avoids nearly all of them for one line of code.

## 5. Export — `exporter.py`

STL is one line of trimesh. The value is everything around it: a mesh that isn't watertight slices into
garbage, and a user who discovers that in their slicer twenty minutes later won't come back. Check first,
say so plainly, export anyway with a warning.

## 6. Session — `session.py`

The step-by-step flow (draw → choose ADD/CUT → set depth → commit) as a small state machine with no Gradio
in it. That means the whole interaction is testable without launching a UI, and swapping Gradio for a React
frontend later touches nothing below this layer.
