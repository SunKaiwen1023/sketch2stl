# Working together on this repo

## Who owns what

Git merges two people editing *different files* perfectly and automatically. It struggles when you both
edit the *same lines*. So: **own whole files, not parts of files.** If you need something changed in the
other person's file, ask them rather than editing it.

| File | Owner | What it is |
|---|---|---|
| `sketch2stl/types.py` | **shared** | The data contract. See the rule below. |
| `sketch2stl/config.py` | **shared** | Every tolerance and magic number. See the rule below. |
| `sketch2stl/strokes.py` | Serena | Canvas pixels → clean millimetre point arrays |
| `sketch2stl/recognizer/*` | Serena | Rules baseline, ML arm, training script |
| `ui/canvas.py` | Serena | Drawing input |
| `sketch2stl/profiles.py` | PK | Primitives → closed extrudable regions |
| `sketch2stl/kernel.py` | PK | Feature stack → mesh. Extrude, union, difference. |
| `sketch2stl/exporter.py` | PK | STL export and printability checks |
| `ui/preview.py`, `ui/layers.py` | PK | 3D view and the layer panel |
| `sketch2stl/session.py` | split | Serena owns the step machine, PK owns the rebuild path — separate methods |
| `app.py` | **shared** | Keep it thin. Wiring only, no logic. |
| `tests/*` | whoever owns the file under test | |
| `docs/decisions.md` | both, append-only | Never edit someone else's entry; add a new one |

If this split doesn't match what you each actually want to build, change it now rather than living with it.

## The shared-file rule

`types.py`, `config.py` and `app.py` are the three files you'll both touch. To change any of them:

1. Message the other person first.
2. Make the change in its own small pull request that does nothing else.
3. Merge it quickly.
4. Both `git pull` immediately.

Never bundle a `types.py` change into a large feature branch. That's how you get a 200-line conflict.

## Branches

`yourname/what-it-does` — `serena/rect-recognition`, `pk/stroke-stitching`.

One branch per thing. When it's merged, delete it. Long-lived branches drift and become painful to merge.

## Commits

Say what changed and why, not what file you touched.

Good: `Add rectangle fitting via minimum-area enclosing rect`
Bad: `update rules.py`

Commit whenever something works, not once at the end of the day. Small commits are much easier to undo.

## Pull requests

Every change goes through one, even a one-liner. With two people you don't need formal approval — but you
do both need to *see* what changed, and the PR page is the easiest way.

Before you open one:

```bash
pytest -q
```

If tests fail, fix them first. A red `main` blocks the other person.

## Tests

If you fix a bug, add a test that would have caught it. That's how the suite gets useful rather than
decorative.

Tests that describe a feature you haven't built yet are welcome — mark them:

```python
@pytest.mark.xfail(reason="TODO: rectangle fitting - rules.py item 1")
```

They flip to passing the moment the feature lands, which is a nice signal. There's one in
`tests/test_recognizer.py` already.

## Never commit

STL output · datasets · `.venv/` · model weights · `__pycache__/` · anything over ~10 MB.

The `.gitignore` blocks the usual suspects. GitHub hard-rejects files over 100 MB, and removing one from
history afterwards is genuinely horrible — so if you're about to `git add .` and you're not sure what's in
there, run `git status` first and look.

## New files since the first commit

| File | Owner | What it is |
|---|---|---|
| `sketch2stl/data/fusion360.py` | Serena | Read Fusion 360 CAD JSONs into labelled 2-D curves |
| `sketch2stl/data/synth.py` | Serena | Clean curve → synthetic hand-drawn stroke |
| `sketch2stl/recognizer/features.py` | Serena | The 12 stroke features. **Append only** — see below |
| `scripts/inspect_dataset.py` | Serena | Check the CAD parser against the real files |
| `scripts/build_dataset.py` | Serena | CAD → labelled strokes |
| `scripts/train_recognizer.py` | Serena | Train + the three-arm evaluation |

**`FEATURE_NAMES` is append-only.** It is the column order shared between training and inference.
Insert a feature in the middle and every model you have already trained silently starts reading the
wrong columns. Add at the end, always.
