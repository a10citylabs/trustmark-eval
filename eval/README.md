# TrustMark robustness experiment (macOS)

A small, self-contained experiment that answers one question: **how well does an
Adobe TrustMark watermark survive everyday image edits?**

You drop your own images into `images/authentic/` and run three scripts:

| Script | Reads | Writes | What it does |
|---|---|---|---|
| `01_watermark.py` | `images/authentic/` | `images/watermarked/` | Embeds a unique random payload in each image |
| `02_edit.py` | `images/watermarked/` | `images/modified/` | Applies 5 edits, one labelled file per edit |
| `03_evaluate.py` | `images/modified/` | `results/` | Decodes everything, scores it, and plots the comparison |

The edits come in two groups. Three are the mundane ones that happen to images
in the real world — JPEG re-compression, downscaling, and a crop — and are
exactly the kind of "non-editorial transformation" TrustMark is trained to
withstand. Two are deliberately aggressive sharpening, which is the interesting
case: sharpening works on exactly the high-frequency detail the watermark lives
in.

| Edit | What it does |
|---|---|
| `jpeg_q40` | JPEG re-compression at quality 40 |
| `resize_50pct` | Lanczos downscale to 50% of each side |
| `crop_80pct_area` | Centre crop keeping 80% of the area |
| `sharpen_usm_300pct` | Unsharp mask at 300%, radius 3 px — a fixed convolution, no model |
| `sharpen_ai_x4` | A super-resolution network re-renders the image, then it is resampled back to its original size |

The two sharpen edits are a matched pair. The first amplifies the detail that is
already in the pixels; the second replaces it with detail a network invented.
The watermark has to survive being *amplified* in one and *rewritten* in the
other, and they do not fail the same way.

---

## 1. Prerequisites

You need Python 3.10 or newer. macOS ships with a Python that is best left
alone, so install your own:

```sh
# Homebrew (install Homebrew first from https://brew.sh if you don't have it)
brew install python@3.11
```

Check it:

```sh
python3 --version     # 3.10 or newer
```

Everything runs on the CPU. On Apple Silicon a 12-megapixel photo takes a couple
of seconds per encode or decode; on Intel Macs expect a little longer.

## 2. Set up the environment

From this `eval/` folder:

```sh
cd eval
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

You will need to `source .venv/bin/activate` again in every new Terminal window.

> **Using the repo's copy of TrustMark instead of PyPI:** `requirements.txt`
> installs the published `trustmark` package. To test the code in this
> repository instead, run `pip install -e ../python` after the step above.

The first script run downloads the TrustMark encoder and decoder weights
(a few hundred MB) from Adobe's CDN into the installed package folder. That
happens once; later runs start immediately. If your network needs a proxy, set
`HTTPS_PROXY` before running.

The AI edit needs one more thing: a ~4.7 MB model file, downloaded on first use
from the Real-ESRGAN releases on GitHub into `eval/models/`. PyTorch is already
there as a TrustMark dependency, so nothing else to install. If the download
cannot happen — no network, a proxy in the way — `02_edit.py` says so, skips
that one edit and runs the other four.

## 3. Add your images

Put anywhere from 5 to 100 images into `images/authentic/`:

```sh
open images/authentic        # drag your photos into the Finder window
```

Notes for macOS:

- JPEG, PNG, WebP, BMP and TIFF work out of the box. For iPhone `.heic` photos,
  run `pip install pillow-heif` and the scripts will pick them up automatically.
- Hidden files such as `.DS_Store` are ignored, so drag-and-drop is safe.
- Avoid `__` (double underscore) in file names — the scripts use it to attach
  edit labels.
- Bigger images are more interesting than tiny ones. TrustMark works at any
  resolution, but a 200×200 thumbnail leaves the watermark very little to work
  with.

## 4. Run the experiment

```sh
python 01_watermark.py
python 02_edit.py
python 03_evaluate.py
```

Then look at the chart:

```sh
open results/robustness_summary.png
```

Step 2 is the slow one, because of the AI edit: it runs a network over every
pixel of every image. Expect seconds per image on a GPU and up to a minute per
12-megapixel photo on a CPU. Two flags help when you are iterating:

```sh
python 02_edit.py --edits sharpen_ai_x4 --keep   # redo one edit, keep the rest
python 02_edit.py --edits jpeg_q40,crop_80pct_area
```

### What you get

```
results/
  payloads.json             the payload embedded in each image (the ground truth)
  results.csv              one row per image x condition
  summary.json             per-edit aggregate metrics
  robustness_summary.png   the charts
```

The console prints a table like this:

```
condition                    n  detected  recovered  lost  bit acc     p10   worst
----------------------------------------------------------------------------------
no_edit_baseline           100      100%       100%     0   100.0%  100.0%  100.0%
jpeg_q40                   100      100%       100%     0    99.2%   98.6%   97.4%
resize_50pct               100      100%        99%     1    98.4%   97.2%   94.3%
crop_80pct_area            100      100%        78%    22    96.4%   93.2%   86.5%
sharpen_usm_300pct         100      100%        20%    80    90.7%   85.1%   70.6%
sharpen_ai_x4              100       77%         0%   100    68.4%   56.5%   40.0%
```

followed by the handful of images that struggled the most across the edits.

### The metrics

- **detected** — the decoder reported that a watermark is present.
- **recovered** — the payload that came back is bit-for-bit the payload that
  went in. This is the number that matters for identifying content.
- **bit acc** — agreement with the 100-bit codeword *before* error correction.
  This is the informative one when recovery fails: an image at 94% was one bit
  flip away from succeeding, an image at 60% was never close. BCH_5 error
  correction repairs up to 5 of the 100 bits, so payloads generally survive
  above ~95% bit accuracy — that limit is drawn on the chart.
- **lost** — how many images lost their payload. A rate needs a denominator:
  "88%" means something different across 8 images and across 100.
- **p10** — the 10th percentile of bit accuracy: a tenth of the images did
  worse than this. With a large set the mean hides a bad tail, and the single
  worst image is noise; p10 is the number that shows an edit is unsafe for some
  images even when the average looks fine.

The `no_edit_baseline` row decodes the untouched watermarked PNG. It is the
control: if it is not at 100%, something is wrong with the setup rather than
with the edits.

### The chart

Three stacked panels, all sharing the same conditions along the x axis:

1. **How often the watermark survived** — detection and exact-payload recovery
   rates, with the number of images that lost their payload.
2. **Raw bit accuracy** — one dot per image, the mean, and the error-correction
   limit. Past 25 images each column also gets the shape of its distribution
   and its interquartile range, because at 100 dots the cloud stops being
   countable.
3. **Per-image bit accuracy** — a row per image, sorted hardest-first, coloured
   red below the error-correction limit and blue above it, so the pass/fail
   boundary is where the colour turns. Up to 25 images every cell carries its
   number; past that the numbers stop fitting, the hardest few rows are named
   in the margin, and a marker flags the cells that actually failed. The full
   per-image numbers are always in `results.csv`.

## 5. Change the experiment

All the knobs are at the top of `common.py`:

```python
MODEL_TYPE = "Q"        # 'Q' balanced, 'P' high quality, 'B' paper model, 'C' compact
ENCODING_NAME = "BCH_5" # BCH_SUPER / BCH_5 / BCH_4 / BCH_3 - robustness vs payload size
WM_STRENGTH = 1.0       # raise for more robustness, at the cost of visibility

JPEG_QUALITY = 40       # lower is harsher
RESIZE_SCALE = 0.5      # linear scale factor
CROP_KEEP_AREA = 0.80   # centre crop keeping this fraction of the area

SHARPEN_RADIUS = 3.0    # unsharp mask radius, in pixels
SHARPEN_PERCENT = 300   # unsharp mask strength; 150% is already strong

AI_SHARPEN_INPUT_SCALE = 0.5  # the size the model sees; the speed/quality dial
AI_SHARPEN_DEVICE = "auto"    # 'auto' | 'cpu' | 'mps' | 'cuda'
```

Experiments worth trying:

- **Find the breaking point.** Drop `JPEG_QUALITY` to 20, or `CROP_KEEP_AREA` to
  0.5, and rerun steps 2 and 3. Bit accuracy degrades before recovery does,
  which is what the middle panel of the chart is for.
- **Quality against robustness.** Set `MODEL_TYPE = "P"` and rerun all three
  steps. Step 1 reports a higher PSNR (a less visible watermark) and step 3
  should show it giving up some robustness in exchange.
- **Strength.** `WM_STRENGTH = 1.5` is what Adobe suggests for surviving
  printing; `0.8` trades robustness for about 5 dB of extra PSNR.
- **Sharpen harder, or gentler.** `SHARPEN_PERCENT = 150` is a normal retouch
  and `500` is destructive; somewhere in between is where the watermark starts
  to go. Rerun steps 2 and 3.
- **Make the AI edit cheaper.** `AI_SHARPEN_INPUT_SCALE = 0.25` is four times
  faster and still sharpens, because the model's 4x output lands exactly on the
  original size. Below 0.25 the output has to be upscaled to fit, which blurs
  it — that is a different edit, not a sharpen.
- **Add a sixth edit.** Write a function in `common.py` that takes a PIL image
  and returns `(edited_image, ".png", {})`, then add it to the `EDITS` dict —
  rotation, brightness, added noise, or a screenshot-style re-encode. Both the
  editing and evaluation scripts pick it up with no other changes.

Changing `MODEL_TYPE`, `ENCODING_NAME` or `WM_STRENGTH` invalidates the images
you already generated, so rerun all three scripts. `03_evaluate.py` stops with a
warning if it notices the mismatch.

## Troubleshooting

**`No images found in images/authentic/`** — the folder is empty, or holds only
file types the scripts don't recognise. The message lists the extensions it
accepts.

**The download of the model weights fails or hangs** — the files come from
`https://cai-watermark.adobe.net/`. Retry; a partial download is detected by an
md5 check and re-fetched on the next run.

**`ModuleNotFoundError: No module named 'trustmark'`** — the virtual environment
is not active. Run `source .venv/bin/activate` from the `eval/` folder.

**A crop or resize scores badly on some images** — that is a result, not a bug.
TrustMark spreads the watermark across the whole image and tolerates roughly 20%
area loss; past that, and on very small or very low-detail images, recovery
starts to fail. The per-image heatmap in the bottom panel shows which images
struggled.

**Everything fails, including the baseline** — check that `common.py` was not
edited between running step 1 and step 3, and that
`results/payloads.json` matches the images currently in `images/watermarked/`.
Rerunning all three scripts in order fixes it.

## How the payloads work

TrustMark carries a 100-bit codeword: your payload plus BCH error-correction
bits plus 4 bits naming the schema. With the default `BCH_5` schema the payload
is 61 bits, which `01_watermark.py` fills with random bits derived from the file
name and `RANDOM_SEED` — so each image gets a distinct payload, and rerunning
the script reproduces the same one. A real deployment would put an identifier
there and look it up in a database.
