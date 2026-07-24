"""Shared configuration and helpers for the TrustMark robustness experiment.

Everything the three scripts have in common lives here: folder layout, the
watermarking parameters, the definitions of the image edits, and small I/O
helpers. Tweak the constants at the top to change the experiment.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

# ---------------------------------------------------------------------------
# Experiment parameters
# ---------------------------------------------------------------------------

# TrustMark model variant:
#   'Q' = balanced (default)      'P' = high visual quality, less robust
#   'B' = original paper model    'C' = compact decoder
MODEL_TYPE = "Q"

# Error correction schema. BCH_5 is the default and the most robust of the
# regular schemas: 61 payload bits, 5 correctable bit flips per 100 raw bits.
ENCODING_NAME = "BCH_5"

# Watermark strength passed to encode(). 1.0 is the default; raising it makes
# the watermark more robust and more visible.
WM_STRENGTH = 1.0

# Seed for the per-image random payloads, so a rerun of 01_watermark.py
# reproduces the same secrets.
RANDOM_SEED = 1234

# ---------------------------------------------------------------------------
# Folder layout
# ---------------------------------------------------------------------------

EVAL_DIR = Path(__file__).resolve().parent
AUTHENTIC_DIR = EVAL_DIR / "images" / "authentic"
WATERMARKED_DIR = EVAL_DIR / "images" / "watermarked"
MODIFIED_DIR = EVAL_DIR / "images" / "modified"
RESULTS_DIR = EVAL_DIR / "results"
# Weights for the AI sharpening model land here, not in the package folder.
MODEL_CACHE_DIR = EVAL_DIR / "models"

MANIFEST_PATH = RESULTS_DIR / "payloads.json"
RESULTS_CSV = RESULTS_DIR / "results.csv"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
SUMMARY_PLOT = RESULTS_DIR / "robustness_summary.png"

# Separator between the image stem and the edit label in modified/ filenames,
# e.g. "beach__jpeg_q40.jpg". Avoid using it in your own file names.
LABEL_SEP = "__"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

# ---------------------------------------------------------------------------
# Image edits
#
# Five edits in two groups. The first three are the mundane ones TrustMark is
# designed to survive: re-compression, rescaling, and a moderate crop. The last
# two are deliberately aggressive sharpening, which is a different kind of
# attack - it amplifies or rewrites exactly the high-frequency detail the
# watermark lives in. Each entry maps a label to a function that takes the
# watermarked PIL image and returns (edited image, file suffix, save kwargs).
# ---------------------------------------------------------------------------

JPEG_QUALITY = 40  # 1-95; lower is a harsher recompression
RESIZE_SCALE = 0.5  # linear scale factor, so 25% of the original pixel count
CROP_KEEP_AREA = 0.80  # centre crop that keeps 80% of the image area

# Aggressive unsharp mask. Photoshop's "Sharpen More" is roughly radius 1 /
# 150%; 300% at radius 3 is well past tasteful and into visible haloing, which
# is the point - this is meant to be a stress test, not a retouch.
SHARPEN_RADIUS = 3.0  # pixels of blur used to build the detail mask
SHARPEN_PERCENT = 300  # strength, in percent
SHARPEN_THRESHOLD = 0  # 0 = sharpen every pixel, including flat areas

# AI sharpening: a 4x super-resolution network re-renders the image and the
# result is resampled back to the original size (see ai_sharpen.py).
#
# AI_SHARPEN_INPUT_SCALE is the size the model sees, as a fraction of the
# original, and it is the speed/quality dial. The model's output is 4x its
# input, so:
#   0.5  the output is 2x the original and is downsampled back - the sharpest
#        result, and the default
#   0.25 the output lands exactly on the original size - 4x faster, still sharp
#   1.0  the model runs at native resolution - 4x slower than 0.5, no sharper
#   <0.25 the output has to be upscaled to fit, which blurs it: not a sharpen
# Cost scales with the square of this number. At 0.5, a 12-megapixel photo is
# roughly half a minute per image on a laptop CPU and a few seconds on a GPU.
AI_SHARPEN_INPUT_SCALE = 0.5
AI_SHARPEN_TILE = 256  # tile size, to bound memory; None = one pass
AI_SHARPEN_TILE_PAD = 16  # overlap between tiles, cropped off again
AI_SHARPEN_DEVICE = "auto"  # 'auto' | 'cpu' | 'mps' | 'cuda'


def _edit_jpeg(img):
    """Re-encode as JPEG - what every upload/share pipeline does."""
    return img, ".jpg", {"quality": JPEG_QUALITY, "subsampling": 2}


def _edit_resize(img):
    """Downscale, e.g. to fit a web page or a messaging app's size limit."""
    w, h = img.size
    new_size = (max(1, round(w * RESIZE_SCALE)), max(1, round(h * RESIZE_SCALE)))
    return img.resize(new_size, Image.LANCZOS), ".png", {}


def _edit_crop(img):
    """Centre crop, e.g. trimming borders or a light reframing."""
    w, h = img.size
    scale = math.sqrt(CROP_KEEP_AREA)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    left, top = (w - new_w) // 2, (h - new_h) // 2
    return img.crop((left, top, left + new_w, top + new_h)), ".png", {}


def _edit_sharpen(img):
    """Aggressive unsharp mask - no model, just a convolution turned up high.

    An unsharp mask boosts whatever differs between the image and a blurred
    copy of it, which is where TrustMark's residual lives. Saved as PNG so the
    only thing being tested is the sharpening.
    """
    sharpened = img.filter(
        ImageFilter.UnsharpMask(
            radius=SHARPEN_RADIUS,
            percent=SHARPEN_PERCENT,
            threshold=SHARPEN_THRESHOLD,
        )
    )
    return sharpened, ".png", {}


def _edit_ai_sharpen(img):
    """Sharpen with a super-resolution model, back at the original size.

    Imported lazily: the weights are only downloaded when this edit runs.
    """
    import ai_sharpen

    return ai_sharpen.sharpen(img), ".png", {}


SHARPEN_LABEL = f"sharpen_usm_{SHARPEN_PERCENT}pct"
AI_SHARPEN_LABEL = "sharpen_ai_x4"

EDITS = {
    f"jpeg_q{JPEG_QUALITY}": _edit_jpeg,
    f"resize_{int(RESIZE_SCALE * 100)}pct": _edit_resize,
    f"crop_{int(CROP_KEEP_AREA * 100)}pct_area": _edit_crop,
    SHARPEN_LABEL: _edit_sharpen,
    AI_SHARPEN_LABEL: _edit_ai_sharpen,
}

# Edits that need something beyond Pillow. Each maps a label to a callable
# returning True when the edit can run; 02_edit.py drops the ones that cannot
# instead of failing the whole run.
def _ai_sharpen_ready(verbose=True):
    import ai_sharpen

    return ai_sharpen.available(verbose=verbose)


EDIT_PREFLIGHT = {AI_SHARPEN_LABEL: _ai_sharpen_ready}

# Label used for the untouched watermarked image, evaluated as a control.
BASELINE_LABEL = "no_edit_baseline"

EDIT_DESCRIPTIONS = {
    BASELINE_LABEL: "Watermarked PNG, straight from the encoder (control)",
    f"jpeg_q{JPEG_QUALITY}": f"JPEG re-compression at quality {JPEG_QUALITY}",
    f"resize_{int(RESIZE_SCALE * 100)}pct": (
        f"Lanczos downscale to {int(RESIZE_SCALE * 100)}% of each side"
    ),
    f"crop_{int(CROP_KEEP_AREA * 100)}pct_area": (
        f"Centre crop keeping {int(CROP_KEEP_AREA * 100)}% of the image area"
    ),
    SHARPEN_LABEL: (
        f"Unsharp mask, radius {SHARPEN_RADIUS:g} px at {SHARPEN_PERCENT}% "
        "(aggressive, no model)"
    ),
    AI_SHARPEN_LABEL: (
        "Real-ESRGAN general x4 super-resolution, resampled back to the "
        "original size (aggressive, AI model)"
    ),
}

# Short names for chart axes, where the full label is too long to read.
EDIT_SHORT_NAMES = {
    BASELINE_LABEL: "baseline",
    f"jpeg_q{JPEG_QUALITY}": f"JPEG q{JPEG_QUALITY}",
    f"resize_{int(RESIZE_SCALE * 100)}pct": f"resize {int(RESIZE_SCALE * 100)}%",
    f"crop_{int(CROP_KEEP_AREA * 100)}pct_area": (
        f"crop {int(CROP_KEEP_AREA * 100)}%"
    ),
    SHARPEN_LABEL: f"sharpen {SHARPEN_PERCENT}%\n(no AI)",
    AI_SHARPEN_LABEL: "sharpen x4\n(AI model)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def short_name(label):
    """A chart-friendly name for an edit label."""
    return EDIT_SHORT_NAMES.get(label, label.replace("_", " "))


def select_edits(wanted=None, verbose=True):
    """The edits to run: `wanted` (or all of them), minus any that cannot run.

    `wanted` is an iterable of labels, e.g. from a --edits flag. Unknown labels
    raise SystemExit; edits whose preflight fails - the AI model with no
    weights and no network, say - are reported and dropped, so one missing
    dependency does not cost you the other four edits.
    """
    if wanted is None:
        labels = list(EDITS)
    else:
        labels = list(dict.fromkeys(wanted))
        unknown = [label for label in labels if label not in EDITS]
        if unknown:
            raise SystemExit(
                f"Unknown edit(s): {', '.join(unknown)}\n"
                f"Available: {', '.join(EDITS)}"
            )

    usable = {}
    for label in labels:
        check = EDIT_PREFLIGHT.get(label)
        if check is not None and not check(verbose=verbose):
            print(f"  ! skipping the '{label}' edit")
            continue
        usable[label] = EDITS[label]
    return usable


def register_optional_formats():
    """Enable HEIC/HEIF reading if pillow-heif happens to be installed.

    macOS photos straight off an iPhone are often .heic, which Pillow cannot
    read on its own. `pip install pillow-heif` makes them work.
    """
    try:
        import pillow_heif  # noqa: WPS433 (optional dependency)
    except ImportError:
        return False
    pillow_heif.register_heif_opener()
    IMAGE_EXTENSIONS.update({".heic", ".heif"})
    return True


def list_images(folder):
    """Image files in `folder`, sorted, skipping hidden files like .DS_Store."""
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file()
        and not p.name.startswith(".")
        and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_rgb(path):
    """Open an image, honour its EXIF orientation, and drop to RGB.

    Returns (rgb_image, alpha_or_None, source_info) so callers can restore an
    alpha channel and carry EXIF/ICC metadata over to the output file.
    """
    img = ImageOps.exif_transpose(Image.open(path))
    alpha = img.split()[-1] if img.mode in ("RGBA", "LA") else None
    return img.convert("RGB"), alpha, img.info


def psnr(img_a, img_b):
    """Peak signal-to-noise ratio in dB between two same-size PIL images."""
    a = np.asarray(img_a, dtype=np.int16)
    b = np.asarray(img_b, dtype=np.int16)
    mse = np.mean(np.square(a - b))
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0) - 10 * math.log10(mse)


def bit_accuracy(bits_a, bits_b):
    """Fraction of matching characters between two equal-length bit strings."""
    if not bits_a or not bits_b or len(bits_a) != len(bits_b):
        return float("nan")
    return sum(x == y for x, y in zip(bits_a, bits_b)) / len(bits_a)


def encoding_constant():
    """The TrustMark.Encoding value named by ENCODING_NAME."""
    from trustmark import TrustMark

    try:
        return getattr(TrustMark.Encoding, ENCODING_NAME)
    except AttributeError as exc:
        valid = [n for n in vars(TrustMark.Encoding) if not n.startswith("_")]
        raise SystemExit(
            f"ENCODING_NAME={ENCODING_NAME!r} is not valid. Pick one of: {valid}"
        ) from exc


def correctable_bits():
    """How many of the 100 raw bits this ECC schema can repair."""
    return {"BCH_SUPER": 8, "BCH_5": 5, "BCH_4": 4, "BCH_3": 3}[ENCODING_NAME]


def build_trustmark(verbose=True):
    """Construct a TrustMark instance configured for this experiment.

    The remover and bounding-box detector models are skipped: they are large
    downloads and this experiment only encodes and decodes.
    """
    from trustmark import TrustMark

    return TrustMark(
        verbose=verbose,
        model_type=MODEL_TYPE,
        encoding_type=encoding_constant(),
        loadRemover=False,
        loadBBoxDetector=False,
    )


def decode_raw_bits(tm, image):
    """Decode the 100 raw bits, bypassing error correction.

    tm.decode() returns an empty string when the ECC layer cannot recover a
    valid payload, which makes failures indistinguishable from each other.
    Turning ECC off for one call exposes the decoder's raw output, so a
    failure can be scored as "94% of bits right" rather than just "failed".
    """
    saved = tm.use_ECC
    tm.use_ECC = False
    try:
        raw_bits, _, _ = tm.decode(image, MODE="binary")
    finally:
        tm.use_ECC = saved
    return raw_bits


def expected_raw_bits(tm, secret):
    """The 100-bit codeword TrustMark embeds for `secret` (payload + ECC)."""
    packet = tm.ecc.encode_binary([secret])[0]
    return "".join(str(int(bit)) for bit in packet)


def read_manifest():
    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"No manifest at {MANIFEST_PATH}.\nRun 01_watermark.py first."
        )
    with open(MANIFEST_PATH) as handle:
        return json.load(handle)


def write_manifest(manifest):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as handle:
        json.dump(manifest, handle, indent=2)


def ensure_dirs():
    for folder in (AUTHENTIC_DIR, WATERMARKED_DIR, MODIFIED_DIR, RESULTS_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def clear_folder(folder):
    """Delete generated image files from a folder, leaving hidden files alone."""
    for path in list_images(folder):
        os.remove(path)


def rel(path):
    """Path relative to the eval/ folder, for tidier console output."""
    try:
        return str(Path(path).resolve().relative_to(EVAL_DIR))
    except ValueError:
        return str(path)
