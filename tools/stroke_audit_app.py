"""Spot-check the synthetic Fusion strokes (the human-in-the-loop check on CAD-derived labels).

    python tools/stroke_audit_app.py            (from the repo root; needs gradio + pillow)
    python tools/stroke_audit_app.py --share    (also prints a public link for your teammate)

Shows a fixed random sample of 200 synthetic strokes, 20 per page, each with the label
that came from the CAD history already selected. Change only the wrong ones:
  - pick the right class, or
  - `bad` if the stroke is not a sensible example of any class.
Both of us audit the SAME 200 strokes, so we also get inter-rater agreement.

Answers go to data/strokes/audit/audit_<name>.csv. The summary at the top shows how often
the CAD label was kept, per class; that number goes in the report.
"""
from __future__ import annotations

import csv, datetime, random, sys, time
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
SYN = ROOT / "data" / "strokes" / "synthetic"
OUT = ROOT / "data" / "strokes" / "audit"; OUT.mkdir(parents=True, exist_ok=True)
CLASSES = ["line", "arc", "circle", "rect", "polyline"]
CHOICES = CLASSES + ["bad"]
PER_CLASS = {"rect": 50, "polyline": 50, "line": 34, "arc": 33, "circle": 33}   # 200, heavier where labels are noisiest
PAGE, SEED, SIZE = 20, 20260923, 200
FIELDS = ["stroke_id", "cad_label", "human_label", "changed", "auditor", "timestamp"]

z = np.load(SYN / "fusion_strokes.npz"); P, O = z["points"], z["offsets"]
ROWS = list(csv.DictReader(open(SYN / "labels.csv")))
INDEX = {r["stroke_id"]: i for i, r in enumerate(ROWS)}


def sample_ids():
    """The same 200 strokes for everyone (fixed seed), saved once so it never changes."""
    f = OUT / "audit_sample.csv"
    if f.exists():
        return [r["stroke_id"] for r in csv.DictReader(open(f))]
    rng = random.Random(SEED); ids = []
    for k, n in PER_CLASS.items():
        pool = [r["stroke_id"] for r in ROWS if r["label"] == k]; ids += rng.sample(pool, n)
    rng.shuffle(ids)
    with open(f, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["stroke_id", "cad_label"])
        for s in ids: w.writerow([s, ROWS[INDEX[s]]["label"]])
    return ids

SAMPLE = sample_ids()


def draw(stroke_id):
    pts = P[O[INDEX[stroke_id]]:O[INDEX[stroke_id] + 1]].astype(float)
    lo, hi = pts.min(0), pts.max(0)
    q = (pts - (lo + hi) / 2) * ((SIZE - 24) / max((hi - lo).max(), 1e-6)) + SIZE / 2
    img = Image.new("RGB", (SIZE, SIZE), "white"); d = ImageDraw.Draw(img)
    d.line([tuple(p) for p in q], fill=(29, 29, 31), width=2, joint="curve")
    x, y = q[0]; d.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(232, 89, 12))   # orange dot = pen start
    return img


def out_path(name): return OUT / f"audit_{name.strip() or 'unknown'}.csv"

def done(name):
    p = out_path(name)
    return {r["stroke_id"]: r for r in csv.DictReader(open(p))} if p.exists() else {}

def summary(name):
    d = done(name)
    if not d: return ""
    per = {k: [0, 0] for k in CLASSES}
    for r in d.values(): per[r["cad_label"]][0] += r["changed"] == "0"; per[r["cad_label"]][1] += 1
    kept = sum(v[0] for v in per.values()); tot = sum(v[1] for v in per.values())
    parts = " · ".join(f"{k} {v[0]}/{v[1]}" for k, v in per.items() if v[1])
    return f"\n\nCAD label kept: **{kept}/{tot} = {100*kept/tot:.0f}%** ({parts})"

def load_page(name):
    if not name.strip():
        return [[], "Type your name first.", time.time(), False] + [gr.update(value=None, visible=False)] * (3 * PAGE)
    d = done(name); left = [s for s in SAMPLE if s not in d]; page = left[:PAGE]
    msg = f"**{name}** · {len(SAMPLE)-len(left)} / {len(SAMPLE)} checked" + (" — all done, thank you!" if not page else "") + summary(name)
    ups = []
    for i in range(PAGE):
        if i < len(page):
            s = page[i]
            ups += [gr.update(value=draw(s), visible=True), gr.update(value=f"`{s}`", visible=True),
                    gr.update(value=ROWS[INDEX[s]]["label"], visible=True)]
        else:
            ups += [gr.update(value=None, visible=False)] * 3
    return [page, msg, time.time(), False] + ups

def save_page(name, ids, t0, warned, *vals):
    changed = sum(v != ROWS[INDEX[s]]["label"] for s, v in zip(ids, vals))
    if ids and not warned and changed == 0 and time.time() - t0 < 10:
        gr.Warning("That was fast and nothing was changed. Check every stroke, then press Save again.")
        return [gr.skip()] * 2 + [t0, True] + [gr.skip()] * (3 * PAGE)
    p = out_path(name); new = not p.exists(); now = datetime.datetime.now().isoformat(timespec="seconds")
    with open(p, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new: w.writeheader()
        for s, v in zip(ids, vals):
            cad = ROWS[INDEX[s]]["label"]
            w.writerow(dict(stroke_id=s, cad_label=cad, human_label=v, changed=int(v != cad), auditor=name.strip(), timestamp=now))
    return load_page(name)

GUIDE = """
Each picture is ONE stroke, scaled to fit. The orange dot is where the pen started.
- `line` straight · `arc` part of a circle, open · `circle` closed circle
- `rect` closed, 4 straight sides at right angles · `polyline` any other closed shape (triangle, slot, L-shape, rounded rectangle…)
- `bad` not a sensible example of anything (a tiny scribble, overlapping mess)
A small gap or overshoot where a closed shape meets itself is fine: that is how people draw.
"""

with gr.Blocks(title="Synthetic stroke audit") as app:
    gr.Markdown("# Synthetic stroke audit\nThe dropdowns show the label from the CAD history. Fix only the wrong ones, then **Save page & next**.")
    with gr.Row():
        name = gr.Textbox(label="Your name", placeholder="PK", scale=3)
        start = gr.Button("Load / resume", variant="primary", scale=1)
    with gr.Accordion("How to judge", open=False):
        gr.Markdown(GUIDE)
    status = gr.Markdown()
    ids = gr.State([]); t0 = gr.State(0.0); warned = gr.State(False)
    cells, pickers = [], []
    for _ in range(PAGE // 5):
        with gr.Row():
            for _ in range(5):
                with gr.Column(min_width=150):
                    img = gr.Image(type="pil", height=170, interactive=False, visible=False, show_label=False, container=False)
                    sid = gr.Markdown(visible=False)
                    lab = gr.Dropdown(CHOICES, label="class", visible=False)
                    cells += [img, sid, lab]; pickers.append(lab)
    save = gr.Button("Save page & next", variant="primary")
    start.click(load_page, [name], [ids, status, t0, warned] + cells)
    save.click(save_page, [name, ids, t0, warned] + pickers, [ids, status, t0, warned] + cells)

if __name__ == "__main__":
    app.launch(inbrowser=True, share="--share" in sys.argv)
