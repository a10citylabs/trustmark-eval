"""AI-model sharpening, used by the `sharpen_ai_x4` edit in common.py.

The non-AI sharpener is a fixed convolution: it amplifies whatever is already
in the pixels, watermark included. A generative super-resolution model is a
different kind of attack. It runs the image through a network that was trained
to *invent* plausible detail, so the output is a re-rendered image rather than
a filtered one, and any signal the network was not trained to preserve - a
watermark, say - only survives by accident.

The model is Real-ESRGAN "general x4 v3" (a SRVGGNetCompact, ~4.7 MB), a
small, widely used restoration network. It upscales 4x, so to keep the edit
comparable with the other ones the result is resampled back to the source
resolution: same pixel dimensions in, same out, but every pixel repainted by
the network.

Everything is lazy. Nothing here is imported, downloaded or run unless the
`sharpen_ai_x4` edit is actually part of the run, so the rest of the
experiment keeps working with no torch weights on disk.
"""

import math
import sys

import common

# Official Real-ESRGAN release asset. Small enough (~4.7 MB) that the download
# is a few seconds, unlike the 64 MB x4plus model.
MODEL_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
    "realesr-general-x4v3.pth"
)
MODEL_NAME = "Real-ESRGAN general x4 v3"
SCALE = 4

# Cached model + device, so 100 images pay the load cost once.
_LOADED = None


class SharpenerUnavailable(RuntimeError):
    """Raised when the model cannot be imported, downloaded or built."""


# ---------------------------------------------------------------------------
# The network
#
# SRVGGNetCompact from Real-ESRGAN: a plain VGG-style stack that ends in a
# pixel shuffle, plus a nearest-neighbour skip so the convolutions only have to
# learn the residual detail. Reimplemented here (~30 lines) to avoid pulling in
# basicsr, which is a heavy dependency with a long tail of pins.
# ---------------------------------------------------------------------------


def _build_network(torch, nn):
    class SRVGGNetCompact(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64,
                     num_conv=32, upscale=4):
            super().__init__()
            self.upscale = upscale
            self.body = nn.ModuleList()
            self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
            for _ in range(num_conv):
                self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
                self.body.append(nn.PReLU(num_parameters=num_feat))
            self.body.append(
                nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1)
            )
            self.upsampler = nn.PixelShuffle(upscale)

        def forward(self, x):
            out = x
            for layer in self.body:
                out = layer(out)
            out = self.upsampler(out)
            base = torch.nn.functional.interpolate(
                x, scale_factor=self.upscale, mode="nearest"
            )
            return out + base

    return SRVGGNetCompact()


def _pick_device(torch):
    """Fastest device this machine actually has: CUDA, then Apple's MPS, then CPU."""
    requested = common.AI_SHARPEN_DEVICE
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load(verbose=True):
    """Load the model once and cache it. Raises SharpenerUnavailable on failure."""
    global _LOADED
    if _LOADED is not None:
        return _LOADED

    try:
        import torch
        from torch import nn
    except ImportError as exc:  # torch ships with trustmark, so this is rare
        raise SharpenerUnavailable(f"PyTorch is not installed ({exc})") from exc

    common.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(f"  Loading {MODEL_NAME} (first run downloads ~4.7 MB to "
              f"{common.rel(common.MODEL_CACHE_DIR)}/)")
    try:
        state = torch.hub.load_state_dict_from_url(
            MODEL_URL,
            model_dir=str(common.MODEL_CACHE_DIR),
            map_location="cpu",
            progress=verbose,
        )
    except Exception as exc:  # offline, proxy, corrupted cache
        raise SharpenerUnavailable(
            f"could not fetch the weights from {MODEL_URL} ({exc})"
        ) from exc

    weights = state.get("params", state)
    model = _build_network(torch, nn)
    try:
        model.load_state_dict(weights, strict=True)
    except Exception as exc:
        raise SharpenerUnavailable(
            f"the downloaded weights do not match the network ({exc}). "
            f"Delete {common.rel(common.MODEL_CACHE_DIR)}/ and retry."
        ) from exc
    model.eval()

    device = _pick_device(torch)
    try:
        model = model.to(device)
    except Exception as exc:  # e.g. an MPS build that cannot allocate
        print(f"  ! {device} unusable ({exc}), falling back to CPU")
        device = torch.device("cpu")
        model = model.to(device)

    if verbose:
        print(f"  {MODEL_NAME} ready on {device.type}")
    _LOADED = (torch, model, device)
    return _LOADED


def available(verbose=True):
    """True if the model can be used; prints why not otherwise."""
    try:
        load(verbose=verbose)
    except SharpenerUnavailable as exc:
        print(f"  ! AI sharpening unavailable: {exc}", file=sys.stderr)
        return False
    return True


def _to_tensor(torch, img):
    import numpy as np

    array = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def _to_image(tensor):
    import numpy as np
    from PIL import Image

    array = tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    return Image.fromarray((array * 255.0 + 0.5).astype(np.uint8))


def _infer(torch, model, device, tensor):
    """Run the network over the tensor, tile by tile to bound peak memory.

    A 4x network holds 16x the input in activations, so a full-resolution photo
    in one pass is gigabytes. Tiles are processed with a padding margin and the
    margin is cropped off again, which keeps the seams invisible.
    """
    tile = common.AI_SHARPEN_TILE
    pad = common.AI_SHARPEN_TILE_PAD
    _, _, height, width = tensor.shape

    if not tile or (height <= tile and width <= tile):
        return model(tensor.to(device)).cpu()

    out = torch.zeros(1, 3, height * SCALE, width * SCALE)
    for top in range(0, height, tile):
        for left in range(0, width, tile):
            bottom, right = min(top + tile, height), min(left + tile, width)
            # Input tile grown by `pad` on every side that has room.
            in_top, in_left = max(0, top - pad), max(0, left - pad)
            in_bottom, in_right = min(height, bottom + pad), min(width, right + pad)
            patch = tensor[:, :, in_top:in_bottom, in_left:in_right].to(device)
            sr = model(patch).cpu()
            # Drop the margin, in output coordinates.
            oy, ox = (top - in_top) * SCALE, (left - in_left) * SCALE
            oh, ow = (bottom - top) * SCALE, (right - left) * SCALE
            out[:, :, top * SCALE:bottom * SCALE, left * SCALE:right * SCALE] = (
                sr[:, :, oy:oy + oh, ox:ox + ow]
            )
    return out


def sharpen(img, verbose=False):
    """Re-render `img` through the model and return it at its original size."""
    from PIL import Image

    torch, model, device = load(verbose=verbose)
    original_size = img.size

    # The model's cost grows with the square of the side it is given, and its
    # output is 4x its input either way. Feeding it half the image and
    # downsampling the 2x output back is both the sharpest result and four
    # times cheaper than running at native resolution - see the note on
    # AI_SHARPEN_INPUT_SCALE in common.py.
    work = img
    scale = common.AI_SHARPEN_INPUT_SCALE
    if scale and scale != 1.0:
        work = img.resize(
            (max(1, math.floor(original_size[0] * scale)),
             max(1, math.floor(original_size[1] * scale))),
            Image.LANCZOS,
        )

    with torch.no_grad():
        upscaled = _to_image(_infer(torch, model, device, _to_tensor(torch, work)))

    if upscaled.size != original_size:
        upscaled = upscaled.resize(original_size, Image.LANCZOS)
    return upscaled
