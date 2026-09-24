#!/usr/bin/env python3
"""Import REAL hand-drawn strokes into the format the trainer evaluates on.

    python scripts/import_strokes.py --data data/real_strokes --out data/real

WHY THIS DATA IS THE MOST VALUABLE IN THE PROJECT
--------------------------------------------------
Everything the model trains on is synthetic - clean Fusion 360 CAD curves put
through a hand-tremor model. That is a guess about how people draw. These
strokes are the ground truth that says whether the guess was any good.

So they are the TEST set, not extra training data. Folding 325 real strokes into
14,600 synthetic ones would dilute them to 2 % of the corpus, and the one
question they can answer - does synthetic training transfer to real hands? -
would become unanswerable. Held out, they measure exactly that.

    scripts/train_recognizer.py --real-test data/real/real_all.npz

INPUT FORMAT
------------
One JSON per recording session:

    {"session_id": "pk_202609241733", "author": "PK", "device": "mouse",
     "date": "2026-09-24", "canvas": [640, 480], "per": 60,
     "strokes": [{"stroke_id": ..., "label": "circle",
                  "points": [[x, y], ...], "t": [0.0, ...]}, ...]}

Points are canvas pixels. `labels.csv` beside them is a manifest and is read for
provenance if present; the JSON labels are authoritative.

OUTPUT
------
    real_all.npz     every stroke - use this as --real-test
    real_train.npz   \\ only when there are 2+ sessions: split BY SESSION, for the
    real_test.npz    / secondary "does a little real data close the gap?" run
    meta.json        counts per class, per session, per author

SPLIT BY SESSION, NEVER RANDOMLY. Strokes from one sitting share a hand, a
mouse, a screen and a mood. A random split puts near-identical strokes on both
sides and reports memorisation as generalisation - the same lesson as HW2's
parent_id grouping and the CAD train_test.json, for the third time.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sketch2stl.config import CANVAS_H, CANVAS_W, MIN_STROKE_PTS   # noqa: E402
from sketch2stl.strokes import prepare_points, px_to_mm            # noqa: E402

CLASSES = ["line", "arc", "circle", "rect", "polyline"]


def load_session(path: Path) -> tuple[dict, list[dict]]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    meta = {k: d.get(k) for k in ("session_id", "author", "device", "date", "canvas")}
    meta["session_id"] = meta["session_id"] or path.stem
    return meta, d.get("strokes") or []


def to_mm(points_px, canvas) -> np.ndarray | None:
    """Canvas pixels -> millimetres, resampled and smoothed.

    Runs the SAME `prepare_points` the live app uses, so a stroke imported here
    and the identical stroke drawn into the running app produce identical
    features. Anything else makes the evaluation measure the importer.
    """
    pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
    if len(pts) < MIN_STROKE_PTS:
        return None
    if canvas and list(canvas[:2]) != [CANVAS_W, CANVAS_H]:
        # a different recording canvas: rescale so millimetres still mean the same
        pts = pts * np.array([CANVAS_W / float(canvas[0]), CANVAS_H / float(canvas[1])])
    return prepare_points(px_to_mm(pts))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="folder of session .json files")
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--test-session", default=None,
                    help="session_id to hold out; default is the smallest session")
    args = ap.parse_args()

    root = Path(args.data)
    files = sorted(root.glob("*.json"))
    if not files:
        raise SystemExit(f"No .json session files in {root.resolve()}")

    strokes, labels, sessions, authors, ids = [], [], [], [], []
    skipped = Counter()
    provenance = {}

    for path in files:
        meta, raw = load_session(path)
        sid = meta["session_id"]
        provenance[sid] = {k: meta[k] for k in ("author", "device", "date", "canvas")}
        n_ok = 0
        for st in raw:
            label = str(st.get("label", "")).lower()
            if label not in CLASSES:
                skipped[f"unknown label {label!r}"] += 1
                continue
            mm = to_mm(st.get("points"), meta.get("canvas"))
            if mm is None:
                skipped["too few points"] += 1
                continue
            strokes.append(mm)
            labels.append(label)
            sessions.append(sid)
            authors.append(meta.get("author") or "unknown")
            ids.append(st.get("stroke_id") or f"{sid}_{n_ok:04d}")
            n_ok += 1
        provenance[sid]["n"] = n_ok
        print(f"  {path.name:<34} {sid:<20} {n_ok:>4} strokes")

    if not strokes:
        raise SystemExit("Nothing imported. Check the JSON has a 'strokes' list with "
                         "'label' and 'points' on each entry.")

    X = np.asarray(strokes, dtype=np.float32)
    y = np.asarray(labels)
    sess = np.asarray(sessions)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(name, mask):
        np.savez_compressed(out / f"{name}.npz", strokes=X[mask], labels=y[mask],
                            model_ids=sess[mask], stroke_ids=np.asarray(ids)[mask],
                            authors=np.asarray(authors)[mask])
        return int(mask.sum())

    n_all = save("real_all", np.ones(len(X), dtype=bool))

    print(f"\n{n_all} strokes from {len(set(sessions))} session(s), "
          f"{len(set(authors))} author(s)")
    print(f"per class: {dict(Counter(y.tolist()))}")
    missing = [c for c in CLASSES if c not in set(y.tolist())]
    if missing:
        print(f"  !! no examples of: {missing} - those classes cannot be scored")
    if skipped:
        print(f"skipped: {dict(skipped)}")

    per_sess = defaultdict(Counter)
    for s, lab in zip(sessions, labels):
        per_sess[s][lab] += 1

    meta = {"n": n_all, "classes": CLASSES, "per_class": dict(Counter(y.tolist())),
            "sessions": {k: dict(v) for k, v in per_sess.items()},
            "provenance": provenance, "canvas": [CANVAS_W, CANVAS_H],
            "note": "Real hand-drawn strokes. Held out as the domain-gap test set; "
                    "the model trains on synthetic strokes only."}

    # secondary experiment: does a little real data close the gap? split BY SESSION
    if len(set(sessions)) > 1:
        test_sid = args.test_session or min(per_sess, key=lambda s: sum(per_sess[s].values()))
        te = sess == test_sid
        tr = ~te
        n_tr, n_te = save("real_train", tr), save("real_test", te)
        print(f"\nby-session split (for the optional fine-tune run):")
        print(f"  train  {n_tr:>4} strokes  sessions {sorted(set(sess[tr]))}")
        print(f"  test   {n_te:>4} strokes  session  {test_sid}")
        meta["by_session_split"] = {"train_n": n_tr, "test_n": n_te,
                                    "test_session": test_sid}
    else:
        print("\nOnly one session, so no by-session split. For the fine-tune "
              "experiment you need strokes from at least two sittings - ideally "
              "from both of you, on different days.")

    with open(out / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nwrote {out}/real_all.npz and {out}/meta.json")
    print("\nNext - the headline measurement:")
    print(f"  python scripts/train_recognizer.py --data data/strokes "
          f"--out models/recognizer --real-test {out}/real_all.npz")


if __name__ == "__main__":
    main()
