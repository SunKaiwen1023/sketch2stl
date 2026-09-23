"""Train the stroke recogniser. STUB - run as `python -m sketch2stl.recognizer.train`.

OWNER: Serena.

Keep training OUT of the app. The app loads a saved model; this script produces
it. That separation is what lets PK run the app without a GPU or a training set.

    python -m sketch2stl.recognizer.train --data data/strokes --out models/recognizer

THE EVALUATION THAT MATTERS (this is the report section, not an afterthought):

    manual   no recognition - the user picked the tool. The ceiling.
    rules    RuleRecognizer. The baseline to beat.
    ml       this model.

Score all three on the SAME held-out strokes, split by session/author so no
stroke from a training session appears in test. Report per-class accuracy, not
just overall - "94% overall" usually hides "0% on arcs", and arcs are the class
a user notices failing.
"""
from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/strokes", help="stroke dataset directory")
    ap.add_argument("--out", default="models/recognizer", help="where to save the model")
    ap.add_argument("--seed", type=int, default=20260922)
    args = ap.parse_args()

    raise NotImplementedError(
        "Serena: 1) load strokes  2) split by session  3) featurise or tensorise  "
        "4) fit  5) score manual/rules/ml on the same test split  6) save"
    )


if __name__ == "__main__":
    main()
