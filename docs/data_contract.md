# The data contract

**Read this together in your first hour. It is what lets you work on separate files without blocking
each other.**

The whole project is one pipeline. Each arrow is a handoff, and each handoff has exactly one type:

```
  canvas                                                                      file
    |                                                                          ^
    v                                                                          |
 Stroke  --->  Primitive  --->  Profile  --->  Feature  --->  Document  --->  Mesh
    |              |               |              |              |             |
 ui/canvas    recognizer/      profiles.py    session.py     types.py      kernel.py
                                                                          exporter.py
    \_____________________/        \_________________________________________/
          SERENA                                    PK
```

Every one of those types is defined in `sketch2stl/types.py` and nowhere else.

## Why this matters more than it looks

PK's geometry code never needs to know whether a `Profile` came from a neural network, a least-squares
circle fit, or a hard-coded test fixture. Serena's recognition code never needs a working boolean kernel to
test against. **Neither of you is ever blocked on the other.**

That only holds if the types stay fixed. The moment someone quietly adds a field, the other person's code
starts failing for reasons they cannot see.

## The rule

`types.py` and `config.py` are **shared files**. To change either:

1. Message the other person first.
2. Make the change in its own small pull request that does nothing else.
3. Merge it quickly.
4. Both `git pull` immediately.

Never bundle a `types.py` change into a big feature branch.

## The handoffs in detail

| Stage | Type | Produced by | Consumed by | The contract |
|---|---|---|---|---|
| Canvas input | `Stroke` | `ui/canvas.py` | `recognizer/` | `points` is (N, 2) float, **canvas pixels**, unresampled. `t` may be None. |
| Recognition | `Primitive` | `recognizer/` | `profiles.py` | `points` is (M, 2) float, **millimetres**, always populated even for POLYLINE. `kind` and `params` are extra information the geometry half is free to ignore. |
| Closing | `Profile` | `profiles.py` | `kernel.py` | `outer` is closed (first point == last) and counter-clockwise, in mm. Holes clockwise. Valid and non-self-intersecting — `profiles.py` guarantees this. |
| Modelling | `Feature` | `session.py` | `kernel.py` | An `Op` (ADD or CUT), a `Profile`, a `depth` and a `z_base`, all in mm. |
| The model | `Document` | `session.py` | `kernel.py` | An **ordered** list of features. Order is the build order and it matters. |
| Output | `trimesh.Trimesh` | `kernel.py` | `exporter.py`, `ui/preview.py` | A single watertight solid, in mm. |

## The two invariants that will bite you if you break them

**Units.** Canvas space is pixels. Everything past `px_to_mm` is millimetres. The conversion happens in
exactly one place — `strokes.px_to_mm` — using `config.PX_PER_MM`. If you catch yourself writing `* 0.1`
or `/ 25.4` anywhere else, that's the bug.

**Y direction.** The canvas has y growing *downward*; the model has y growing *upward*. `px_to_mm` flips
it, and that's the only flip in the codebase. If a shape ever comes out mirrored, look there first.

## `Primitive.points` is always populated — on purpose

Even when the recogniser has no idea what it's looking at, it returns a `POLYLINE` with the raw resampled
path and a low confidence. It never returns None and never raises.

This means PK's half **always** gets something extrudable. In the worst case the user gets their wobbly
hand-drawn outline instead of a clean circle, which is a far better failure than an error dialog — and it
means the geometry half can be built and demoed before recognition works at all.
