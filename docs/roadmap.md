# Three-week roadmap

Presentation **Oct 5**, report **Oct 9**. Today is **Sept 23**. That is not much time, so the plan is
ordered by *what would hurt most if it were missing on Oct 5*.

## The rule that matters

**Have something demoable at the end of every week.** A rough thing that runs beats a sophisticated thing
that doesn't compile the night before. The scaffold already runs end to end — never let it stop running.

---

## Week 1 (Sept 23–29) — make the loop real

| | Serena | PK |
|---|---|---|
| Goal | Real strokes reach the recogniser | A rectangle drawn as four lines becomes a solid |
| Build | D5 decision, then `strokes_from_image` or the JS canvas. Rectangle fitting in `rules.py`. | Stroke stitching in `profiles.py`. Trim/extend. |
| Also | Start collecting strokes — D7 | Wire the layer panel so depths are editable |

**End-of-week demo:** draw a rectangle freehand, it becomes a plate, draw a circle, it becomes a hole,
export an STL.

---

## Week 2 (Sept 30–Oct 4) — make it good, and measure it

| | Serena | PK |
|---|---|---|
| Goal | Three recognition arms, scored | Sketch planes and an honest printability check |
| Build | Finish the stroke dataset. Train the ML arm. **Score manual / rules / ml on the same held-out split.** | Front/Right planes. Extrude from a face if time. Overhang colouring in the preview. |

**This is the week the report gets written, not the week after.** Every number in it comes from the
evaluation you run here.

**End-of-week demo:** the full thing, plus a table comparing the three arms.

---

## Week 3 (Oct 5 onward) — present and write

- **Oct 3–4:** deploy to HF Spaces. Budget a whole evening — `manifold3d` occasionally fails to build on a
  Space, and that's a known-fix problem, not a mystery, but only if you hit it with time to spare.
- **Oct 4:** rehearse the demo twice, on the deployed version, on someone else's laptop. Have a recorded
  fallback video in case the Space is down.
- **Oct 5:** present.
- **Oct 6–9:** report.

---

## What to cut first if you're behind

In this order. Cutting from the top costs you the least.

1. Arc recognition — circles and rectangles carry the demo
2. Sketch planes other than Top
3. Fillets and chamfers
4. The ML arm's fancy version — a classical model on hand-designed features is a perfectly good result, and
   comparing it honestly to the rules baseline is a *better* report than an underpowered CNN

## What not to cut, ever

- The end-to-end path: draw → add → cut → export. If that breaks, you have nothing to show.
- The three-arm evaluation. It's the part that makes this a *Designing with AI* project rather than a CAD
  tool, and it's what the rubric is actually asking for.
