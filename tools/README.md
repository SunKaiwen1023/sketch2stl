# tools/

Data and training helpers that sit outside the app. Nothing here is imported by `app.py`.

| File | What it does | Run |
|---|---|---|
| `stroke_collector.html` | Collect hand-drawn strokes (the manual dataset). Ordered pen points + timestamps, 640×480 canvas pixels. | Double-click to open in a browser. Download the session JSON into `data/strokes/raw/`. |
| `fusion_strokes.py` | Synthetic training strokes from Fusion 360 Gallery sketches (labels from the CAD history). | See the docstring: three `--start/--stop` chunks, then `--merge`. Needs the dataset at `../r1.0.1`. |
| `stroke_audit_app.py` | Human spot-check of 200 synthetic strokes (CAD label kept or fixed), both of us on the same sample for agreement. | `python tools/stroke_audit_app.py` (add `--share` for a link your teammate can open). Saves to `data/strokes/audit/`. |
| `strokes_io.py` | Load synthetic + hand-drawn strokes into one format; `rasterize()` for image models. | `from tools.strokes_io import load_npz_dataset, load_hand_sessions, rasterize` |

The fine-tuned model lives in `notebooks/finetune_resnet_strokes.ipynb` (run on Colab with a T4 GPU).

Data folders (all gitignored, share through a Hugging Face dataset):

```
data/strokes/
  raw/          hand-drawn sessions from the collector   <- the manual dataset, the real test
  synthetic/    fusion_strokes.npz + labels.csv          <- training only, labeled synthetic
```
