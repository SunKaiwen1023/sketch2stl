# Using the Fusion 360 Gallery dataset

## THE LICENCE — read this before you share anything

The Fusion 360 Gallery Dataset is under a **custom Autodesk licence**, not an open one. The
attached `Fusion 360 Gallery Dataset Public License.docx` is the authority; this is the summary
that matters for us.

| | |
|---|---|
| ✅ Use it for this course | Non-commercial research is explicitly permitted. Coursework qualifies. |
| ❌ **Do not commit the 5.17 GB folder to git** | "You may not redistribute or make available to others the Dataset in its entirety." Point people at the AutodeskAILab GitHub repo instead. |
| ❌ **Do not push it, or your synthesised strokes, to a public HF dataset** | Unlike HW1. A derived set is a "Modified Set" and may only be shared under terms **at least as restrictive**, clearly marked as a modification, with attribution, and restricted to non-commercial research. A public HF dataset with an open licence does not satisfy that. |
| ✅ A trained **model** is fine to share | Weights are not the dataset. Say what it was trained on. |
| ✅ Attribute it | Cite "Fusion 360 Gallery Dataset" and the papers on the dataset website in your report. |
| ❌ Never try to identify who made a model | The licence forbids re-association with creators. |

`.gitignore` already blocks `data/**`, so the dataset and everything built from it stay local by
default. **Keep it that way.** Put the dataset *outside* the repo — she has it at
`24679_DPA/r1.0.1`, beside `24679_DPA/sketch2stl`, which is exactly right.

---

## What this dataset is, and why it needs a trick

The Reconstruction subset is 8,625 real CAD models, each a construction sequence: sketches of
exact parametric curves, then extrude features that add or cut material.

**It contains no hand-drawn strokes at all.** So it cannot train a stroke recogniser directly.

What it does have is something better: **every curve carries a perfect ground-truth label.** A
`Circle3D` entry states, exactly, "circle, centre here, radius this."

So:

```
   clean CAD curve  ──handdraw()──►  synthetic hand-drawn stroke
   label known exactly               label carries over for free
```

`sketch2stl/data/synth.py` applies five distortions, each modelling something a hand actually
does: low-frequency tremor, speed-varying point spacing, endpoint overshoot, corner rounding,
and a mild global shear. Out of a dataset with zero strokes you get tens of thousands of
perfectly labelled ones — drawn from the shape distribution of **real engineering parts**,
not from shapes someone invented for a homework.

And it solves decision D7: you no longer need 400 hand-drawn samples before you can start.

## The evaluation this sets up

The strokes you and PK draw by hand become the **test set**:

| | Trained on | Tested on | What it tells you |
|---|---|---|---|
| In-domain | synthetic | synthetic (held-out models) | Is the classifier learning anything at all? |
| **Domain gap** | synthetic | **real hand-drawn** | Is the synthesis model faithful? |

That second row is the most interesting number in the project, and it is a real ML finding
rather than a leaderboard score. A small drop means the distortion model is good. A large one
tells you precisely which distortion you got wrong — and you can chase it, because each one is
a separate switch in `HandStyle`.

**Calibrate before you train.** Draw thirty shapes, fit the ideal primitive to each, and measure
the mean deviation. Tune `HandStyle.tremor_mm` until synthetic strokes have a similar spread.
Put both numbers in the report — it is the evidence that the synthetic data is not a fantasy.
`synth.deviation()` computes exactly this.

---

## Running it

You have the dataset at `24679_DPA/r1.0.1` and the repo at `24679_DPA/sketch2stl`, so all paths
below are `../r1.0.1/...` from inside the repo.

### 1. Check the parser against your actual files — do this first

```bash
python scripts/inspect_dataset.py --data ../r1.0.1/reconstruction --split ../r1.0.1/train_test.json
```

Thirty seconds, and it tells you whether the parser reads your files before you wait on a full
build. `fusion360.py` was written against the documented r1.0.x schema without your files in
hand, so if the layout differs this is where you find out. If it prints **BROKEN** or
**PARTIAL**, re-run with `--dump` and send me the output — the fix will be small.

### 2. Build the stroke dataset

```bash
python scripts/build_dataset.py \
    --data ../r1.0.1/reconstruction \
    --split ../r1.0.1/train_test.json \
    --out data/strokes \
    --per-class 4000
```

**Always pass `--split`.** Curves from one CAD model are highly self-similar — a part's four
fillet arcs have nearly identical radii. Splitting curves randomly puts siblings on both sides
and inflates the score. `train_test.json` splits by *model*, which is the correct grouping.
Same lesson as HW2's `parent_id`, wearing a CAD hat.

Try `--max-models 200` first to see it work in a minute before committing to the full run.

`--style neat|typical|shaky` controls how shaky a hand to simulate. Building all three and
comparing is a cheap, genuinely informative ablation.

### 3. Train and evaluate

```bash
python scripts/train_recognizer.py --data data/strokes --out models/recognizer
```

Writes `model.joblib`, `results.json` (the three-arm comparison) and `confusion.png`. Add
`--model autogluon` to reuse your HW2 `TabularPredictor` workflow instead of scikit-learn;
`hgb` is the better default because it keeps the HF Space build small.

Once `models/recognizer/model.joblib` exists, **`app.py` picks it up on the next start** with no
code change. The header tells you which arm is live.

### 4. The domain-gap test, once you have real strokes

Collect hand-drawn strokes, save them as an `.npz` with `strokes` (N, 64, 2) in mm and `labels`
(N,), then:

```bash
python scripts/train_recognizer.py --data data/strokes --real-test data/real_strokes.npz
```

---

## Two free extras the dataset gives you

**A real-world prior on ADD vs CUT.** `operations_from_model` reads each model's extrude
operations. `build_dataset.py` reports the split — in a trial run, roughly two thirds add and
one third cut. That is evidence that PK's two-operation vocabulary covers what designers
actually do, which is worth a line in the report.

**A sanity check on scope.** If a large share of models use operations outside `add`/`cut`
(revolve, loft, `IntersectFeatureOperation`), that quantifies what the tool cannot express —
much better than guessing at it in the limitations section.
