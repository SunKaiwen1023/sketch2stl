# Proposal: closing three gaps in the plan

**2026-09-23 · PK (drafted with Claude) · for discussion with Serena before anything lands in `decisions.md`**

The scaffold's architecture stays exactly as it is: strokes → primitives → profiles → features → mesh,
three recognition arms, split by session. This proposal only adds data and one model arm around it.
Nothing here edits `types.py`, `config.py`, `app.py` or any file under `recognizer/`.

## The three gaps

| # | Gap | Why it matters | Proposed fix |
|---|---|---|---|
| 1 | The ML arm has ~400 hand-drawn strokes to learn from | Overfits; the scaffold itself warns about this for the 1-D CNN | **Synthetic strokes from Fusion 360** (29,387, labeled by the CAD history) for training |
| 2 | Both ML options in `ml.py` (AutoGluon on features, 1-D CNN) are *trained from scratch* | The brief requires **≥2 of 3 model categories** | Add a **fine-tuned ResNet-18** arm on rasterised strokes |
| 3 | D7's 400 strokes < 500 | The brief requires **≥500 manual samples** | Collect **60 per class per person = 600**, with ordered pen paths |

## What already exists (in this repo, not committed yet)

- `tools/fusion_strokes.py`: reads every sketch in Fusion 360 Gallery Reconstruction r1.0.1 and turns
  real designer primitives into strokes that look hand-drawn (wobble, tremor, overshoot/gap at closure,
  uneven sampling, slower pen at corners), in canvas pixels, with timestamps.
  Result: **29,387 strokes**, balanced (6,000 each; circle 5,387), grouped by Fusion project
  (3,629 sessions), split train 22,002 / val 3,945 / test 3,440. Stored in `data/strokes/synthetic/` (gitignored).
- `tools/stroke_collector.html`: open in any browser, no install. Prompts shapes in random order, records
  ordered pen points + timestamps at 640×480 (same space as `types.Stroke`), autosaves, downloads one JSON
  per session for `data/strokes/raw/`. Records author, device, date and session id (the grouping variable).
- `tools/strokes_io.py`: one loader for both datasets + `rasterize()` for image models.
- `notebooks/finetune_resnet_strokes.ipynb`: the fine-tuned arm. Trains on synthetic, reports per-class
  accuracy on the synthetic test **and on the hand-drawn strokes** (the number that counts).
  Smoke test on CPU, *without* ImageNet weights, 5,000 strokes, 3 epochs: 86% on the synthetic test.
  The real run on Colab (pretrained, all data) should do better; polyline vs rect is the main confusion.

## Early finding for Serena's rules arm

Running the current `RuleRecognizer` on 1,500 synthetic test strokes: lines 88% correct, circles only
38% (the rest fall to POLYLINE). Likely cause: `CLOSE_TOL_MM = 3.0` and `CIRCLE_RESIDUAL_MM = 1.2`
are tight for large, wobbly circles. Worth checking against real hand-drawn circles before tuning.
Caveat: the synthetic noise level is a guess, so calibrate it once real strokes exist.

## Suggested answers for the open decisions

**D5 — strokes from the canvas.** Use (b) a custom JS canvas: `tools/stroke_collector.html` already does
the pointer capture, so porting it into `gr.HTML` is mostly wiring. Keep (a) raster vectorising as the
fallback if the JS route is not working by Sep 28.

**D6 — shape classes.** Keep all five (line, arc, circle, rect, polyline). With 29k synthetic training
strokes the "more classes need more data" worry goes away, and five classes keep the rules baseline
honest rather than trivially strong.

**D7 — manual strokes.** 60 per class per person = **600 strokes**, collected with the stroke collector,
PK and Serena each by **Sep 26**. Each person draws in at least 2 separate sessions (different days or
devices) so the session split has something to split.

**New D8 — synthetic data.** Fusion-derived strokes are used for **training only**, clearly labeled
synthetic in the dataset card, never mixed into validation-for-reporting or test. The headline results
are on hand-drawn strokes, split by session.

**New D9 — model categories.** Arms become: manual (ceiling) · rules (baseline) · ML-scratch (Serena's
AutoGluon or 1-D CNN) · ML-fine-tuned (ResNet-18). Scratch + fine-tuned covers the two-category rule.
Score all four on the same hand-drawn test sessions, per class.

## What this changes in the roadmap

- Week 1 adds: both draw 300 strokes each (≈30–40 minutes each), upload `data/strokes` to a HF dataset.
- Week 2 adds: run the notebook on Colab; the evaluation table gets a fourth column.
- The earlier Fusion *profile* labeling (whole shapes: washer, polygon, …) is paused. It can come back
  later as a "snap to symmetric / regular polygon" helper, but it is not needed for this architecture.
