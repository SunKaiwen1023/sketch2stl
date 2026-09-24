#!/usr/bin/env python3
"""Check the Fusion 360 parser against YOUR files before you spend an hour building.

    python scripts/inspect_dataset.py --data ../r1.0.1/reconstruction --split ../r1.0.1/train_test.json

RUN THIS FIRST. The Fusion 360 Gallery schema has shifted between releases, and
`sketch2stl/data/fusion360.py` was written against the documented r1.0.x layout
without your actual files in hand. This tells you in thirty seconds whether the
parser reads them, and if it does not, it prints the real structure so the fix
is obvious rather than a guess.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sketch2stl.data.fusion360 import (curves_from_model, iter_models,     # noqa: E402
                                       load_split, operations_from_model)


def describe(obj, indent=0, max_depth=3, max_keys=8):
    pad = "  " * indent
    if indent >= max_depth:
        print(f"{pad}...")
        return
    if isinstance(obj, dict):
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_keys:
                print(f"{pad}... {len(obj) - max_keys} more keys")
                break
            t = type(v).__name__
            extra = f" (len {len(v)})" if isinstance(v, (list, dict)) else f" = {v!r}"[:60]
            print(f"{pad}{k}: {t}{extra}")
            if isinstance(v, (dict, list)):
                describe(v, indent + 1, max_depth, max_keys)
    elif isinstance(obj, list) and obj:
        print(f"{pad}[0]:")
        describe(obj[0], indent + 1, max_depth, max_keys)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--n", type=int, default=25, help="models to sample")
    ap.add_argument("--dump", action="store_true", help="print one file's raw structure")
    args = ap.parse_args()

    root = Path(args.data)
    print(f"looking in {root.resolve()}")
    if not root.exists():
        raise SystemExit(
            f"\n{root} does not exist.\n"
            f"You want the folder holding the *.json files - usually r1.0.1/reconstruction.\n"
            f"From inside sketch2stl/ that is probably:  --data ../r1.0.1/reconstruction"
        )

    jsons = sorted(root.glob("*.json"))
    print(f"found {len(jsons)} .json files")
    if not jsons:
        others = Counter(p.suffix for p in root.iterdir() if p.is_file())
        raise SystemExit(f"\nNo .json files here. This folder holds: {dict(others)}\n"
                         f"Check you are pointing at 'reconstruction' and not its parent.")

    if args.split:
        split = load_split(args.split)
        print(f"split file: {len(split['train'])} train, {len(split['test'])} test models")
        missing = [m for m in split["train"][:200] if not (root / f"{m}.json").exists()]
        if missing:
            print(f"  WARNING: {len(missing)}/200 sampled train ids have no .json here "
                  f"(e.g. {missing[:2]}). Wrong folder, or a partial download.")
        else:
            print("  the first 200 train ids all have files here - good")

    if args.dump:
        print(f"\n=== raw structure of {jsons[0].name} ===")
        with open(jsons[0], encoding="utf-8") as f:
            describe(json.load(f))

    print(f"\n=== parsing {args.n} models ===")
    kinds, ops = Counter(), Counter()
    per_model, empty = [], []
    for model_id, data in iter_models(root, limit=args.n):
        curves = curves_from_model(data, model_id)
        per_model.append(len(curves))
        if not curves:
            empty.append(model_id)
        kinds.update(c.kind.value for c in curves)
        ops.update(operations_from_model(data))

    if not per_model:
        raise SystemExit("Could not read any file. Are they valid JSON?")

    total = sum(per_model)
    print(f"  {len(per_model)} models -> {total} curves "
          f"({total / len(per_model):.1f} per model)")
    print(f"  curve types: {dict(kinds)}")
    print(f"  extrude ops: {dict(ops)}")
    if empty:
        print(f"  {len(empty)} models yielded nothing, e.g. {empty[:3]}")

    print("\n=== verdict ===")
    if total == 0:
        print("  BROKEN. The parser found no curves at all.")
        print("  Re-run with --dump and send me the output - the schema differs from")
        print("  what fusion360.py expects and the fix will be small.")
    elif not kinds.get("circle") and not kinds.get("arc"):
        print("  PARTIAL. Lines parse but circles/arcs do not, so three of your five")
        print("  classes will be empty. Re-run with --dump and send me the output.")
    else:
        est = int(total / len(per_model) * 6900)
        print(f"  GOOD. Extrapolating to 6,900 training models: roughly {est:,} curves.")
        print(f"  Next:  python scripts/build_dataset.py --data {args.data} "
              f"--split {args.split or '<train_test.json>'} --out data/strokes")


if __name__ == "__main__":
    main()
