#!/usr/bin/env python3
"""Train the ADD/CUT suggester and score it against the baselines it must beat.

    python scripts/extract_addcut.py --data ~/Downloads/r1.0.1/reconstruction
    python scripts/train_addcut.py --data data/addcut.csv --out models/addcut

Produces:

    models/addcut/model.joblib     what the app loads, if it is there
    models/addcut/meta.json        features, classes, how it was trained
    models/addcut/results.json     the numbers for the report
    models/addcut/reliability.png  is the confidence honest?
    models/addcut/importance.png   which features it actually uses

THREE ARMS, same held-out split
-------------------------------
    majority   always guess the commoner class. The floor.
    rules      "overlaps the part -> cut", the geometric baseline shipping in
               `sketch2stl/suggest.py` today. THE NUMBER TO BEAT.
    ml         this classifier.

SPLIT BY PROJECT, NOT BY DESIGN. The dataset has roughly 8,600 designs across
3,700 projects, so several designs are variants of one part. A random split puts
near-duplicates on both sides and inflates the score by a lot. Grouping by
project is the difference between an honest number and a flattering one.

CALIBRATION IS PART OF THE DELIVERABLE, not a nicety. The UI shows the
confidence to the user and pre-selects a button with it, so a model that says
90% and is right 60% of the time is worse than a less accurate model that knows
what it does not know. That is what reliability.png measures.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FEATURES = ["step_index", "n_loops", "has_inner_loop", "is_circular",
            "coplanar_prior", "log_area", "area_ratio", "overlap_frac",
            "inside_frac", "centroid_dist"]
CLASSES = ["add", "cut"]


def load(path: Path):
    rows = list(csv.DictReader(open(path)))
    if not rows:
        raise SystemExit(f"{path} is empty - run scripts/extract_addcut.py first.")
    X, y, groups = [], [], []
    for r in rows:
        area = float(r["area_mm2"])
        X.append([
            float(r["step_index"]),
            float(r["n_loops"]),
            float(r["has_inner_loop"]),
            float(r["is_circular"]),
            float(r["coplanar_prior"]),
            float(np.log1p(max(area, 0.0))),
            float(r["area_ratio"]),
            float(r["overlap_frac"]),
            float(r["inside_frac"]),
            float(r["centroid_dist"]),
        ])
        y.append(CLASSES.index(r["label"]))
        groups.append(r["project"])
    return np.asarray(X, float), np.asarray(y), np.asarray(groups), rows


def rule_predict(X: np.ndarray) -> np.ndarray:
    """The geometric baseline, exactly as `suggest.suggest_op` applies it."""
    overlap = X[:, FEATURES.index("overlap_frac")]
    return (overlap >= 0.9).astype(int)


def reliability(prob: np.ndarray, correct: np.ndarray, bins: int = 10):
    """Observed accuracy against claimed confidence, bin by bin."""
    edges = np.linspace(0.5, 1.0, bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1.0 else prob <= hi)
        if m.sum() >= 5:
            xs.append(float(prob[m].mean()))
            ys.append(float(correct[m].mean()))
            ns.append(int(m.sum()))
    return xs, ys, ns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/addcut.csv")
    ap.add_argument("--out", default="models/addcut")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import joblib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import (accuracy_score, brier_score_loss,
                                 classification_report, confusion_matrix)
    from sklearn.model_selection import GroupShuffleSplit

    X, y, groups, rows = load(Path(args.data))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train, test = next(split.split(X, y, groups))
    print(f"{len(X)} steps · train {len(train)} / test {len(test)} · "
          f"{len(set(groups[train]) & set(groups[test]))} projects in both "
          f"(must be 0)")

    base = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                          random_state=args.seed)
    # Isotonic calibration on an inner split: the raw scores of a boosted tree
    # are not probabilities, and this model's whole job is to report one.
    model = CalibratedClassifierCV(base, method="isotonic", cv=4)
    model.fit(X[train], y[train])

    prob = model.predict_proba(X[test])
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    truth = y[test]

    majority = int(np.bincount(y[train]).argmax())
    arms = {
        "majority": np.full_like(truth, majority),
        "rules": rule_predict(X[test]),
        "ml": pred,
    }
    results = {name: {"accuracy": float(accuracy_score(truth, p))}
               for name, p in arms.items()}

    # The interesting slice: the first step of a design is always an ADD, which
    # every arm gets right and which flatters all of them equally.
    later = X[test][:, FEATURES.index("step_index")] > 0
    for name, p in arms.items():
        results[name]["accuracy_after_first_step"] = float(
            accuracy_score(truth[later], p[later])) if later.any() else None

    correct = (pred == truth).astype(float)
    results["ml"]["brier"] = float(brier_score_loss(truth, prob[:, 1]))
    results["ml"]["mean_confidence"] = float(conf.mean())
    results["ml"]["confident_share"] = float((conf >= 0.6).mean())
    results["ml"]["accuracy_when_confident"] = float(
        correct[conf >= 0.6].mean()) if (conf >= 0.6).any() else None
    results["ml"]["report"] = classification_report(
        truth, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    results["ml"]["confusion"] = confusion_matrix(truth, pred).tolist()
    results["n_train"], results["n_test"] = len(train), len(test)

    joblib.dump({"model": model, "features": FEATURES, "classes": CLASSES}, out / "model.joblib")
    (out / "meta.json").write_text(json.dumps(
        {"features": FEATURES, "classes": CLASSES, "source": str(args.data),
         "grouped_by": "project", "calibration": "isotonic"}, indent=2))
    (out / "results.json").write_text(json.dumps(results, indent=2))

    xs, ys, ns = reliability(conf, correct)
    fig, ax = plt.subplots(figsize=(4.4, 4.2))
    ax.plot([0.5, 1], [0.5, 1], "--", color="#999", lw=1, label="perfectly honest")
    ax.plot(xs, ys, "o-", color="#2a6fdb", label="this model")
    for x, yv, n in zip(xs, ys, ns):
        ax.annotate(str(n), (x, yv), fontsize=7, xytext=(0, 6),
                    textcoords="offset points", ha="center", color="#666")
    ax.set_xlabel("confidence the model reported")
    ax.set_ylabel("how often it was actually right")
    ax.set_title("Is the confidence honest?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "reliability.png", dpi=160)

    try:
        from sklearn.inspection import permutation_importance
        imp = permutation_importance(model, X[test], truth, n_repeats=8,
                                     random_state=args.seed)
        order = np.argsort(imp.importances_mean)
        fig, ax = plt.subplots(figsize=(5.2, 3.8))
        ax.barh([FEATURES[i] for i in order], imp.importances_mean[order],
                xerr=imp.importances_std[order], color="#2a6fdb")
        ax.set_xlabel("drop in accuracy when shuffled")
        ax.set_title("What the model relies on")
        fig.tight_layout()
        fig.savefig(out / "importance.png", dpi=160)
        results["ml"]["importance"] = {FEATURES[i]: float(imp.importances_mean[i])
                                       for i in order[::-1]}
        (out / "results.json").write_text(json.dumps(results, indent=2))
    except Exception as exc:                         # noqa: BLE001 - optional plot
        print("importance plot skipped:", exc)

    print("\n           overall   after step 1")
    for name in ("majority", "rules", "ml"):
        a = results[name]["accuracy"]
        b = results[name]["accuracy_after_first_step"]
        print(f"  {name:8s}  {a:.3f}     {b:.3f}" if b is not None
              else f"  {name:8s}  {a:.3f}")
    gap = results["ml"]["accuracy_after_first_step"] - results["rules"]["accuracy_after_first_step"]
    print(f"\n  ml - rules on the steps that matter: {gap:+.3f}")
    print(f"  Brier {results['ml']['brier']:.3f} · "
          f"{results['ml']['confident_share']:.0%} of predictions are >=60% "
          f"confident, and those are right "
          f"{results['ml']['accuracy_when_confident']:.0%} of the time")
    print(f"\nwrote {out}/  - the app picks the model up on restart")
    if gap <= 0:
        print("\nNOTE: the model does NOT beat the rule. That is a publishable "
              "result, not a failure - report it and keep the rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
