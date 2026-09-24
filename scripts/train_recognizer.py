#!/usr/bin/env python3
"""Train the stroke recogniser and score all three arms on the same held-out set.

    python scripts/train_recognizer.py --data data/strokes --out models/recognizer

THIS SCRIPT IS THE EVALUATION SECTION OF YOUR REPORT. It produces:

    models/recognizer/model.joblib     the trained classifier the app loads
    models/recognizer/meta.json        what it is, how it was trained
    models/recognizer/results.json     the three-arm comparison
    models/recognizer/confusion.png    per-class confusion matrices
    models/recognizer/importance.png   which features the model relies on

THE THREE ARMS
--------------
    manual   the user picked the tool, so recognition is perfect by definition.
             The CEILING. Reported as 1.000 to make the axis honest.
    rules    least-squares fit each primitive, keep the best residual.
             The BASELINE the learned model has to beat.
    ml       this classifier.

The headline is not the ml accuracy. It is the GAP between ml and rules, and
which classes it comes from. If ml beats rules by two points overall but by
thirty on arcs, say that - it is a far more useful finding than one number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib                                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402

from sketch2stl.recognizer.features import FEATURE_NAMES, extract_matrix   # noqa: E402
from sketch2stl.recognizer.ml import MLRecognizer                          # noqa: E402
from sketch2stl.recognizer.rules import RuleRecognizer                     # noqa: E402
from sketch2stl.strokes import mm_to_px                                    # noqa: E402
from sketch2stl.types import Stroke                                        # noqa: E402

LABELS = ["line", "arc", "circle", "rect", "polyline"]


def load(path: Path, name: str):
    f = path / f"{name}.npz"
    if not f.exists():
        raise SystemExit(f"{f} not found. Run scripts/build_dataset.py first.")
    d = np.load(f, allow_pickle=True)
    return d["strokes"], d["labels"].astype(str), d["model_ids"].astype(str)


def score_rules(strokes: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Run the rules arm over the same strokes. Returns predicted labels."""
    r = RuleRecognizer()
    return np.array([r.recognize(Stroke(points=mm_to_px(s))).kind.value for s in strokes])


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", labels=LABELS, zero_division=0)
    per = {}
    rep = classification_report(y_true, y_pred, labels=LABELS,
                                output_dict=True, zero_division=0)
    for k in LABELS:
        per[k] = round(float(rep[k]["f1-score"]), 4)
    n = len(y_true)
    ci = 1.96 * np.sqrt(max(acc * (1 - acc), 1e-12) / n)
    print(f"\n--- {name} (n={n}) ---")
    print(f"  accuracy  {acc:.4f}  (95% CI +/- {ci:.4f})")
    print(f"  macro-F1  {f1:.4f}")
    print("  per-class F1: " + "  ".join(f"{k}={v:.3f}" for k, v in per.items()))
    return {"accuracy": round(float(acc), 4), "macro_f1": round(float(f1), 4),
            "ci95": round(float(ci), 4), "per_class_f1": per, "n": int(n),
            "confusion": confusion_matrix(y_true, y_pred, labels=LABELS).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/strokes")
    ap.add_argument("--out", default="models/recognizer")
    ap.add_argument("--model", default="hgb", choices=["hgb", "rf", "logreg", "autogluon"],
                    help="hgb is the best default; autogluon reuses your HW2 workflow")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--real-test", default=None,
                    help="optional .npz of REAL hand-drawn strokes - the domain-gap test")
    args = ap.parse_args()

    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    Xs_tr, y_tr, m_tr = load(data, "train")
    Xs_te, y_te, m_te = load(data, "test")
    print(f"train {len(Xs_tr)} strokes / {len(set(m_tr))} models")
    print(f"test  {len(Xs_te)} strokes / {len(set(m_te))} models")

    overlap = set(m_tr) & set(m_te)
    if overlap:
        print(f"\n!! {len(overlap)} CAD models appear in BOTH splits. Your test score "
              f"will be inflated. Rebuild with --split pointing at train_test.json.\n")
    else:
        print("no model appears in both splits - the test set is genuinely held out")

    print("\nextracting features...")
    t0 = time.time()
    X_tr = extract_matrix(list(Xs_tr))
    X_te = extract_matrix(list(Xs_te))
    print(f"  {X_tr.shape[1]} features, {time.time() - t0:.0f}s")

    # ---------------------------------------------------------------- train
    print(f"\ntraining ({args.model})...")
    t0 = time.time()
    if args.model == "autogluon":
        import pandas as pd
        from autogluon.tabular import TabularPredictor
        df = pd.DataFrame(X_tr, columns=FEATURE_NAMES)
        df["label"] = y_tr
        predictor = TabularPredictor(label="label", eval_metric="f1_macro",
                                     path=str(out / "autogluon")).fit(df, time_limit=600)
        y_ml = predictor.predict(pd.DataFrame(X_te, columns=FEATURE_NAMES)).to_numpy().astype(str)
        model = predictor
    else:
        from sklearn.ensemble import (HistGradientBoostingClassifier,
                                      RandomForestClassifier)
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        model = {
            "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                                  random_state=args.seed),
            "rf": RandomForestClassifier(n_estimators=400, random_state=args.seed, n_jobs=-1),
            "logreg": make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=4000, random_state=args.seed)),
        }[args.model]
        model.fit(X_tr, y_tr)
        y_ml = model.predict(X_te).astype(str)
    train_s = time.time() - t0
    print(f"  {train_s:.0f}s")

    # ------------------------------------------------------- the three arms
    print("\nscoring the rules baseline on the same strokes...")
    y_rules = score_rules(Xs_te, y_te)

    results = {
        "manual": {"accuracy": 1.0, "macro_f1": 1.0, "n": int(len(y_te)),
                   "note": "the user picks the tool, so recognition cannot be wrong. "
                           "This is the ceiling, not a measurement."},
        "rules": report("rules (baseline)", y_te, y_rules),
        "ml": report(f"ml ({args.model})", y_te, y_ml),
    }
    gap = results["ml"]["macro_f1"] - results["rules"]["macro_f1"]
    print(f"\n>>> ml beats rules by {gap:+.4f} macro-F1")
    per_gap = {k: round(results["ml"]["per_class_f1"][k] - results["rules"]["per_class_f1"][k], 3)
               for k in LABELS}
    print(f">>> per class: {per_gap}")
    print(">>> THAT per-class line is the finding. Which classes did learning actually help?")
    results["gap_macro_f1"] = round(float(gap), 4)
    results["gap_per_class_f1"] = per_gap

    # ---------------------------------------------- optional domain-gap test
    if args.real_test and Path(args.real_test).exists():
        d = np.load(args.real_test, allow_pickle=True)
        Xr, yr = d["strokes"], d["labels"].astype(str)
        print(f"\n=== DOMAIN GAP: real hand-drawn strokes (n={len(Xr)}) ===")
        X_real = extract_matrix(list(Xr))
        y_real_ml = (model.predict(__import__("pandas").DataFrame(X_real, columns=FEATURE_NAMES))
                     .to_numpy().astype(str) if args.model == "autogluon"
                     else model.predict(X_real).astype(str))
        results["ml_on_real"] = report("ml on REAL strokes", yr, y_real_ml)
        results["rules_on_real"] = report("rules on REAL strokes", yr, score_rules(Xr, yr))
        drop = results["ml"]["macro_f1"] - results["ml_on_real"]["macro_f1"]
        print(f"\n>>> synthetic -> real drop: {drop:+.4f} macro-F1")
        print(">>> This is the most interesting number in the whole project. A small drop "
              "means the synthesis model is faithful; a large one tells you exactly which "
              "distortion you got wrong.")
        results["domain_gap_macro_f1"] = round(float(drop), 4)

    # ------------------------------------------------------------- figures
    from sklearn.inspection import permutation_importance
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for ax, (arm, yp) in zip(axes[:2], [("rules", y_rules), (f"ml ({args.model})", y_ml)]):
        cm = np.array(results["rules" if arm == "rules" else "ml"]["confusion"])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(LABELS))); ax.set_xticklabels(LABELS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(LABELS))); ax.set_yticklabels(LABELS, fontsize=8)
        thr = cm.max() / 2 if cm.max() else 0.5
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > thr else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(arm, fontsize=10)

    ax = axes[2]
    try:
        if args.model != "autogluon":
            imp = permutation_importance(model, X_te, y_te, n_repeats=5,
                                         random_state=args.seed, n_jobs=-1)
            order = np.argsort(imp.importances_mean)
            ax.barh([FEATURE_NAMES[i] for i in order], imp.importances_mean[order],
                    xerr=imp.importances_std[order], color="#9bbb59",
                    edgecolor="black", linewidth=0.5, capsize=3)
            ax.set_xlabel("permutation importance")
            ax.set_title("what the model relies on", fontsize=10)
        else:
            ax.text(0.5, 0.5, "see the AutoGluon leaderboard", ha="center", va="center")
            ax.axis("off")
    except Exception as e:                                   # noqa: BLE001
        ax.text(0.5, 0.5, f"unavailable\n{e}", ha="center", va="center", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out / "confusion.png", dpi=140, bbox_inches="tight")
    print(f"\nwrote {out / 'confusion.png'}")

    # ---------------------------------------------------------------- save
    meta = {"model": args.model, "seed": args.seed, "train_seconds": round(train_s, 1),
            "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
            "classes": LABELS, "features": FEATURE_NAMES,
            "source": "Fusion 360 Gallery Dataset (Reconstruction), synthesised strokes",
            "licence": "Autodesk non-commercial research"}
    if args.model != "autogluon":
        MLRecognizer.save(model, LABELS, out, meta)
        print(f"wrote {out / 'model.joblib'}")
    with open(out / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out / 'results.json'}")
    print(f"\nThe app will pick this up automatically: RECOGNIZER_PATH={out}")


if __name__ == "__main__":
    main()
