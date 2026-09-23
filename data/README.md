# data/

**Nothing in here is committed** except this file. See `.gitignore`.

## What goes here

```
data/
  strokes/
    raw/            one JSON or NPZ per recording session
    processed/      resampled, normalised, ready to train on
    labels.csv      session_id, stroke_id, label, author, date, device
```

## How to share it with each other

Not through git. Options, in order of preference:

1. **A Hugging Face dataset** — same as HW1. Free, versioned, and it means the training script can pull it
   anywhere. Do this one.
2. Google Drive, if you're in a hurry.

## The one thing to get right

**Record who drew each stroke, when, and on what device.** That's your grouping variable.

Split the dataset by session, never randomly. A random split lets the model learn *your handwriting* and
report it as shape recognition — it's the same `parent_id` grouping lesson from HW2 in a different costume,
and it's the single easiest way to produce an impressive number that means nothing.
