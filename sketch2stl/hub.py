"""Fetch the trained models from the Hugging Face Hub when they are not on disk.

OWNER: PK. Model weights are never committed (see .gitignore), which is why a
fresh clone used to run the rules arms without saying so. The app calls
`fetch_models()` at startup: anything missing is downloaded once into models/.

    ML1  pakiino/sketch2stl-recognizer        -> models/recognizer/model.joblib
    ML2  pakiino/sketch2stl-revolve-or-mirror -> models/revolve_or_mirror/

Set SKETCH2STL_FETCH_MODELS=0 to stay offline. Never raises: no network, no
huggingface_hub, or a missing repo all just leave the rules arms in place.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

ML1_REPO = os.environ.get("ML1_REPO", "pakiino/sketch2stl-recognizer")
ML2_REPO = os.environ.get("ML2_REPO", "pakiino/sketch2stl-revolve-or-mirror")


def _get(repo: str, filename: str, dest: Path) -> str:
    if dest.exists():
        return "on disk"
    from huggingface_hub import hf_hub_download
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(hf_hub_download(repo, filename), dest)
    return "downloaded"


def fetch_models(recognizer_dir: str = "models/recognizer",
                 kind_dir: str = "models/revolve_or_mirror") -> list[str]:
    """Download whatever is missing. Returns one status line per model."""
    if os.environ.get("SKETCH2STL_FETCH_MODELS", "1") == "0" or "pytest" in sys.modules:
        return ["model download skipped"]
    lines = []
    jobs = [("ML1", ML1_REPO, ["model.joblib"], Path(recognizer_dir)),
            ("ML2", ML2_REPO, ["revolve_or_mirror.onnx", "meta.json"], Path(kind_dir))]
    for name, repo, files, d in jobs:
        try:
            got = [_get(repo, f, d / f) for f in files]
            lines.append(f"{name}: {', '.join(sorted(set(got)))} ({repo})")
        except Exception as exc:                     # noqa: BLE001
            lines.append(f"{name}: not available ({type(exc).__name__}) - using rules")
    return lines
