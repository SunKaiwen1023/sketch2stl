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
pytest -q          # expect: 31 passed, 1 xfailed
python app.py
```

Click **Add volume** → **Cut volume** → **Export STL**. You get an 80 × 60 × 20 mm plate with a Ø30 hole —
the part in the mockup.

**That already works.** The pipeline is real end to end: circle fitting, polygon cleanup, extrusion,
boolean difference, watertight STL with printability checks. What's stubbed is clearly marked, and the
whole point of this scaffold is that you replace pieces of a running system rather than assembling a dead
one.

New to git? → **[SETUP.md](SETUP.md)**, which starts from installing it.

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
| **Rectangle recognition** | 🔨 stub — `recognizer/rules.py` TODO 1 |
| **Arc recognition** | 🔨 stub — `recognizer/rules.py` TODO 2 |
| **ML recogniser** | 🔨 stub — `recognizer/ml.py` |
| **Real canvas input** | 🔨 stub — `ui/canvas.py`, and see decision D5 |
| **Stroke stitching** (rect drawn as 4 lines) | 🔨 stub — `profiles.py` TODO 1 |
| Sketch planes other than Top | 🔨 stub — `kernel.py` TODO 1 |

Until `ui/canvas.py` is done, `app.py` feeds the pipeline a fixed shape so both halves can be developed and
demoed independently. Delete `_placeholder_strokes` the moment real input works.

---

## Layout

```
sketch2stl/
├── app.py                    Gradio wiring. Keep it thin — logic goes below.
├── sketch2stl/
│   ├── types.py              ★ THE DATA CONTRACT — read this first
│   ├── config.py             ★ every tolerance and magic number
│   ├── strokes.py            canvas px → clean mm arrays          [Serena]
│   ├── recognizer/
│   │   ├── base.py           the interface all three arms implement
│   │   ├── rules.py          least-squares baseline               [Serena]
│   │   ├── ml.py             learned classifier                   [Serena]
│   │   └── train.py          training + the 3-arm evaluation      [Serena]
│   ├── profiles.py           primitives → closed regions          [PK]
│   ├── kernel.py             feature stack → mesh                 [PK]
│   ├── exporter.py           STL + printability                   [PK]
│   └── session.py            app state, undo, step flow           [split]
├── ui/                       canvas [Serena] · preview, layers [PK]
├── tests/                    31 tests, run them before every push
└── docs/
    ├── data_contract.md      ★ how the two halves connect
    ├── architecture.md       why it's built this way
    ├── decisions.md          the log — 3 decisions still open
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
rules baseline by enough to justify existing, and on which shapes. Build the rules arm properly or the
comparison is worthless.

And split the stroke dataset **by session, never randomly** — otherwise the model learns your handwriting
and reports it as shape recognition. Same lesson as HW2's `parent_id` grouping, different costume.

---

## Before you write any code

1. Both of you get to **31 passed** and export an STL (SETUP.md Part 4).
2. Read [`docs/data_contract.md`](docs/data_contract.md) out loud, together. It's short.
3. Settle the ownership table in [`CONTRIBUTING.md`](CONTRIBUTING.md).
4. Answer the three open decisions at the bottom of [`docs/decisions.md`](docs/decisions.md), with dates.
   D7 — *how many strokes, by when* — is the one most likely to slip, and Serena's whole half is blocked
   behind it.
