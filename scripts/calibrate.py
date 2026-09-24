#!/usr/bin/env python3
"""Measure how far the hand-drawing synthesis is from real strokes, and fix it.

    python scripts/calibrate.py --real data/real/real_all.npz

This is the script that makes the synthetic training data honest. Everything the
recogniser learns comes from clean CAD curves pushed through `data/synth.py` -
a GUESS about how people draw. Real strokes are the only thing that can check
that guess, and until someone records some there is no way to know.

WHAT IT FOUND THE FIRST TIME IT RAN
------------------------------------
On PK's 325 mouse-drawn strokes:

    real circles    median circle-fit residual  2.38 mm
    synthetic       median circle-fit residual  0.30 mm

Seven times too clean. The model had never seen a stroke as rough as the ones it
meets in the app. 82 % of real circles blew past the 1.2 mm acceptance threshold
and fell through to POLYLINE - the rules arm scored F1 0.213 on real circles
while scoring 0.96 on synthetic ones.

Two constants came out of that, and both are now set from measurement rather
than from taste: `config.CIRCLE_RESIDUAL_MM` and the `MOUSE` HandStyle.

RE-RUN THIS whenever you collect strokes from a new device or a new person. A
stylus on a tablet will not have a mouse's failure modes, and the numbers below
will say so.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sketch2stl.config import CIRCLE_RESIDUAL_MM, CLOSE_TOL_MM     # noqa: E402
from sketch2stl.data.synth import (STYLES, HandStyle, handdraw,     # noqa: E402
                                   to_canvas_scale)
from sketch2stl.recognizer.rules import fit_circle                  # noqa: E402
from sketch2stl.strokes import is_closed, prepare_points            # noqa: E402

CLASSES = ["line", "arc", "circle", "rect", "polyline"]


def unit_circle(r=15.0, n=64):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([r * np.cos(a), r * np.sin(a)])
    return np.vstack([p, p[:1]])


def synth_residuals(style: HandStyle, n=80, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([
        fit_circle(prepare_points(handdraw(to_canvas_scale(unit_circle(), rng), style, rng)))[3]
        for _ in range(n)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="real_all.npz from import_strokes.py")
    ap.add_argument("--target-class", default="circle",
                    help="which class to calibrate against; circle is the most sensitive")
    args = ap.parse_args()

    d = np.load(args.real, allow_pickle=True)
    X, y = d["strokes"], d["labels"].astype(str)
    authors = set(d["authors"].astype(str).tolist()) if "authors" in d else {"?"}
    print(f"{len(X)} real strokes, authors: {sorted(authors)}\n")

    # ---------------------------------------------------------------- 1
    print("=== 1. circle-fit residual by true class, REAL strokes (mm) ===")
    print(f"{'class':<10}{'p10':>7}{'median':>9}{'p90':>7}   {'% over ' + str(CIRCLE_RESIDUAL_MM):>12}")
    stats = {}
    for c in CLASSES:
        m = y == c
        if not m.any():
            continue
        res = np.array([fit_circle(s)[3] for s in X[m]])
        stats[c] = res
        print(f"{c:<10}{np.percentile(res,10):>7.2f}{np.median(res):>9.2f}"
              f"{np.percentile(res,90):>7.2f}{(res>CIRCLE_RESIDUAL_MM).mean():>12.0%}")

    target = stats.get(args.target_class)
    if target is None:
        raise SystemExit(f"No {args.target_class} strokes in the file.")

    print("\nA straight LINE fits a huge circle almost perfectly, which is why its")
    print("residual is near zero. The recogniser only tries a circle on a CLOSED")
    print("stroke, so the real competitors are rect and polyline - compare against")
    print("those when choosing the threshold.")

    # ---------------------------------------------------------------- 2
    print(f"\n=== 2. threshold sweep for CIRCLE_RESIDUAL_MM (currently {CIRCLE_RESIDUAL_MM}) ===")
    rivals = np.concatenate([stats[c] for c in ("rect", "polyline") if c in stats]) \
        if any(c in stats for c in ("rect", "polyline")) else np.array([])
    print(f"{'threshold':>10}{'circles kept':>15}{'rect/polyline admitted':>26}")
    best = None
    for t in (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0):
        keep = float((target <= t).mean())
        admit = float((rivals <= t).mean()) if len(rivals) else 0.0
        mark = ""
        if best is None or keep - admit > best[1]:
            best, mark = (t, keep - admit), ""
        print(f"{t:>10.1f}{keep:>15.0%}{admit:>26.0%}")
    print(f"\nbest separation at {best[0]:.1f} mm  (kept minus admitted = {best[1]:.0%})")
    if abs(best[0] - CIRCLE_RESIDUAL_MM) > 0.4:
        print(f"  -> consider setting CIRCLE_RESIDUAL_MM = {best[0]:.1f} in config.py")
    else:
        print("  -> the current value is already about right")

    # ---------------------------------------------------------------- 3
    print("\n=== 3. is the synthesis as rough as a real hand? ===")
    want = float(np.median(target))
    print(f"real {args.target_class} median residual: {want:.2f} mm\n")
    print(f"{'style':<10}{'tremor_mm':>11}{'median':>9}   verdict")
    closest = None
    for name, st in STYLES.items():
        got = float(np.median(synth_residuals(st)))
        ratio = want / got if got > 0 else float("inf")
        verdict = ("too clean by %.1fx" % ratio if ratio > 1.3 else
                   "too rough by %.1fx" % (1 / ratio) if ratio < 0.77 else "MATCHES")
        print(f"{name:<10}{st.tremor_mm:>11.2f}{got:>9.2f}   {verdict}")
        if closest is None or abs(np.log(ratio)) < abs(np.log(closest[1])):
            closest = (name, ratio)
    print(f"\nclosest style: '{closest[0]}'  ->  build with --style {closest[0]}")

    # ---------------------------------------------------------------- 4
    print("\n=== 4. closure: an unclosed circle is read as an ARC ===")
    for c in ("circle", "rect", "polyline"):
        if c in stats:
            frac = np.mean([is_closed(s, CLOSE_TOL_MM) for s in X[y == c]])
            print(f"  real {c:<9} detected closed: {frac:>4.0%}")
    print("\nAnything well below 100% here costs that class directly - the stroke is")
    print("classified as open and never gets a circle or rectangle fit at all.")


if __name__ == "__main__":
    main()
