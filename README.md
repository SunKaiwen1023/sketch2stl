# Sketch3D — hand-drawn sketches to 3D-printable STL

**CMU 24-679 *Designing with AI*, Project 1 · Serena Sun & PK**

Draw a shape by hand. Say whether it adds material or removes it. Give it a depth. Repeat. Export an STL.

```
    draw  ──►  recognise  ──►  close  ──►  extrude  ──►  add / cut  ──►  STL
               (ML + rules)                              (boolean)
```

---

## Run it right now

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q          # expect: 47 passed
python app.py
```

Draw a rectangle → **Add volume**. Draw a circle inside it → **Cut volume**. → **Export STL**.
You get the plate from the mockup.

**That already works.** The pipeline is real end to end: circle fitting, polygon cleanup, extrusion,
boolean difference, watertight STL with printability checks. What's stubbed is clearly marked, and the
whole point of this scaffold is that you replace pieces of a running system rather than assembling a dead
one.

New to git? → **[SETUP.md](SETUP.md)**, which starts from installing it.

---

## Data and ML process (so far)

Full write-up with figures: **[docs/datasets.md](docs/datasets.md)**. Data on Hugging Face: **[pakiino/sketch2stl-datasets](https://huggingface.co/datasets/pakiino/sketch2stl-datasets)**.

| | ML 1 — stroke recognizer | ML 2 — smart suggestions |
| --- | --- | --- |
| Question | What kind of stroke is this? | ADD or CUT, and how thick? |
| Category | Trained from scratch (+ fine-tuned arm for comparison) | Fine-tuned ResNet-18 |
| Manual data | 325 hand-drawn + 197 human-reviewed = **522** | — |
| Training data | 29,387 synthetic strokes from Fusion 360 | 19,273 extrude steps from Fusion 360 |
| Baseline to beat | Rules: 3% on real circles | 64.6% ADD/CUT; thickness off ~3.6× |

![Hand-drawn strokes, 10 per class](docs/figures/hand_PK_sample.png)

---

## What works, what doesn't

| | Status |
|---|---|
| Circle recognition (least squares) | ✅ works |
| Line recognition (total least squares) | ✅ works |
| Closed-polyline fallback | ✅ works |
| Polygon repair — self-intersections, bowties | ✅ works |
| Extrude, union, difference, watertight STL | ✅ works |
| Printability checks — wall thickness, build volume | ✅ works |
| Undo / redo, feature stack, step-by-step flow | ✅ works |
| Rectangle recognition (min-area rect + fill ratio) | ✅ works |
| Arc recognition (circle fit + sweep test) | ✅ works |
| **Real canvas input** — raster → skeleton → ordered paths | ✅ works |
| **ML recogniser** — 12 features, classify then least-squares fit | ✅ works, needs training |
| Fusion 360 CAD parser + synthetic stroke generator | ✅ works |
| Three-arm evaluation (manual / rules / ml) | ✅ works |
| **Stroke stitching** (rect drawn as 4 separate lines) | 🔨 stub — `profiles.py` TODO 1 |
| Sketch planes other than Top | 🔨 stub — `kernel.py` TODO 1 |
| Extrude from an existing face | 🔨 stub — `kernel.py` TODO 2 |

The app runs on the **rules baseline** until you train a model. Once `models/recognizer/model.joblib`
exists it is picked up automatically on the next start — no code change. The header says which arm is live.

---

## Layout

```
sketch2stl/
├── app.py                    Gradio wiring. Keep it thin — logic goes below.
├── sketch2stl/
│   ├── types.py              ★ THE DATA CONTRACT — read this first
│   ├── config.py             ★ every tolerance and magic number
│   ├── strokes.py            canvas px → clean mm arrays          [Serena]
│   ├── data/
│   │   ├── fusion360.py      read Fusion 360 CAD → labelled curves [Serena]
│   │   └── synth.py          clean curve → synthetic hand stroke  [Serena]
│   ├── recognizer/
│   │   ├── base.py           the interface all three arms implement
│   │   ├── rules.py          least-squares baseline               [Serena]
│   │   ├── features.py       12 hand-designed stroke features     [Serena]
│   │   └── ml.py             learned classifier                   [Serena]
│   ├── profiles.py           primitives → closed regions          [PK]
│   ├── kernel.py             feature stack → mesh                 [PK]
│   ├── exporter.py           STL + printability                   [PK]
│   └── session.py            app state, undo, step flow           [split]
├── ui/                       canvas [Serena] · preview, layers [PK]
├── scripts/
│   ├── inspect_dataset.py    check the CAD parser against your files
│   ├── build_dataset.py      CAD → synthetic labelled strokes
│   └── train_recognizer.py   train + the three-arm evaluation
├── tests/                    47 tests, run them before every push
└── docs/
    ├── data_contract.md      ★ how the two halves connect
    ├── architecture.md       why it's built this way
    ├── dataset.md           ★ Fusion 360 licence + how to train
    ├── decisions.md          the log — 2 decisions still open
    └── roadmap.md            week by week to Oct 5
```

★ = read before writing any code.

---

## The idea behind the structure

**One ordered list of operations is the entire model.** `kernel.build(document)` is a pure function of that
list. Undo is "drop the last one and rebuild". The 3D preview can't drift out of sync with the layer panel
because both are derived from the same list. Every test builds documents directly, with no UI involved.

**Two operations only** — ADD and CUT, both extrusions. Every hard problem in solid modelling lives outside
that vocabulary, and two operations are enough for the plate in the mockup.

**The two halves never block each other.** Serena's side ends at `Profile`; PK's side starts there. The
recogniser never returns None and never raises — at worst it hands back the raw hand-drawn outline with a
low confidence, so the geometry half always gets something extrudable. That's why PK could build and demo
the whole kernel before recognition existed at all.

---

## The part that makes this an AI project

Three interchangeable recognition arms, scored on the same held-out strokes:

| Arm | What it is |
|---|---|
| `manual` | The user picks the tool first. No recognition. The ceiling. |
| `rules` | Least-squares fit each primitive type, keep the best residual. The baseline. |
| `ml` | A learned classifier on normalised strokes. |

**The result that matters is not "our model gets 94%."** It's whether the learned arm beats a *serious*
rules baseline by enough to justify existing, and on which shapes. `scripts/train_recognizer.py` prints
exactly that, per class.

### Where the training data comes from

The Fusion 360 Gallery dataset has 8,625 real CAD models and **zero hand-drawn strokes** — but every curve
in it carries a perfect label. So `data/synth.py` distorts clean CAD curves into plausible hand-drawn ones,
and the label comes along for free. Tens of thousands of labelled strokes, drawn from the shape
distribution of real engineering parts.

Your own hand-drawn strokes then become the **test set**: train on synthetic, test on real. That
domain gap is a more interesting result than any in-domain accuracy. See [`docs/dataset.md`](docs/dataset.md)
— including the licence, which is **not** open and means this data never goes near a public HF dataset.

---

## Before you write any code

1. Both of you get to **47 passed** and export an STL (SETUP.md Part 4).
2. Read [`docs/data_contract.md`](docs/data_contract.md) out loud, together. It's short.
3. Settle the ownership table in [`CONTRIBUTING.md`](CONTRIBUTING.md).
4. Answer the three open decisions at the bottom of [`docs/decisions.md`](docs/decisions.md), with dates.
   D7 — *how many strokes, by when* — is the one most likely to slip, and Serena's whole half is blocked
   behind it.
