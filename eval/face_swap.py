"""Face swapping, used by the `face_swap_reference` edit in common.py.

Every other edit in this experiment is a *global* transformation: recompress,
rescale, crop, sharpen. They touch every pixel a little, and TrustMark is built
to survive exactly that. A face swap is a different kind of change. It is an
editorial edit - the claim the picture makes about who is in it stops being
true - and it is local: one region is rewritten completely and the rest of the
frame is left untouched.

That makes it the interesting counterpart to the sharpening pair. The watermark
is spread across the whole image, so a swapped face destroys the signal inside
the face and nowhere else, and the decoder gets to vote with the rest of the
picture. Whether the payload still comes back tells you something a global edit
cannot: a watermark that survives says "this file came from us", not "nothing in
this file was changed".

How the swap works:

  * faces are found with YuNet, a 232 KB ONNX detector from the OpenCV model
    zoo, which returns a box and five landmarks (both eyes, nose tip, both
    mouth corners) per face;
  * the five landmarks of the reference face are mapped onto the five landmarks
    of the target face with a similarity transform, so the reference arrives at
    the right position, scale and roll;
  * the warped reference is blended in through a feathered ellipse with
    OpenCV's Poisson `seamlessClone`, which matches the surrounding skin tone
    and lighting instead of pasting a visible rectangle.

The reference face is read from `images/authentic/reference.jpg` - the same
folder as the photos, because it is an input you supply. `01_watermark.py`
holds it out, so it is never watermarked or evaluated itself.

Everything is lazy, like the AI sharpener: OpenCV is only imported and the
detector only downloaded if this edit is actually part of the run.
"""

import hashlib
import sys
import urllib.request

import numpy as np

import common

# YuNet, from the OpenCV model zoo. The zoo keeps its weights in git-lfs, so
# the download goes through media.githubusercontent.com; the raw.github URL
# returns a 131-byte pointer file instead of the model.
MODEL_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
MODEL_FILE = "face_detection_yunet_2023mar.onnx"
MODEL_NAME = "YuNet face detector"
# Checked after the download, so a truncated or proxy-mangled file is caught
# here rather than as a confusing ONNX parse error.
MODEL_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

# Cached detector, and the reference face prepared once for the whole run.
_DETECTOR = None
_REFERENCE = None

# Per-image bookkeeping, so 02_edit.py can report what was actually swapped.
_LAST = None
_STATS = []


class FaceSwapUnavailable(RuntimeError):
    """Raised when OpenCV, the detector or the reference face is missing."""


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


def _import_cv2():
    try:
        import cv2  # noqa: WPS433 (optional dependency, imported lazily)
    except ImportError as exc:
        raise FaceSwapUnavailable(
            f"OpenCV is not installed ({exc}). Run: pip install opencv-python"
        ) from exc
    if not hasattr(cv2, "FaceDetectorYN"):
        raise FaceSwapUnavailable(
            f"this OpenCV ({cv2.__version__}) has no FaceDetectorYN; "
            "4.5.4 or newer is needed"
        )
    return cv2


def _model_path(verbose=True):
    """Path to the detector weights, downloading them on first use."""
    common.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = common.MODEL_CACHE_DIR / MODEL_FILE

    if path.exists() and _sha256(path) == MODEL_SHA256:
        return path

    if verbose:
        print(f"  Downloading {MODEL_NAME} (~230 KB) to "
              f"{common.rel(common.MODEL_CACHE_DIR)}/")
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            partial.write_bytes(response.read())
    except Exception as exc:  # offline, proxy, 404
        partial.unlink(missing_ok=True)
        raise FaceSwapUnavailable(
            f"could not fetch the detector from {MODEL_URL} ({exc})"
        ) from exc

    digest = _sha256(partial)
    if digest != MODEL_SHA256:
        partial.unlink(missing_ok=True)
        raise FaceSwapUnavailable(
            f"the downloaded detector is not the expected file "
            f"(sha256 {digest[:12]}..., expected {MODEL_SHA256[:12]}...)"
        )
    partial.replace(path)
    return path


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _detector(verbose=True):
    """Load the detector once and cache it."""
    global _DETECTOR
    if _DETECTOR is None:
        cv2 = _import_cv2()
        path = _model_path(verbose=verbose)
        try:
            detector = cv2.FaceDetectorYN.create(
                str(path), "", (320, 320),
                common.FACE_SWAP_SCORE_THRESHOLD, 0.3, 5000,
            )
        except Exception as exc:
            raise FaceSwapUnavailable(
                f"OpenCV could not build the detector ({exc})"
            ) from exc
        if verbose:
            print(f"  {MODEL_NAME} ready")
        _DETECTOR = (cv2, detector)
    return _DETECTOR


def detect_faces(image_bgr, verbose=False):
    """Faces in a BGR array, biggest first, as (box, landmarks) pairs.

    Detection runs on a downscaled copy - YuNet's cost grows with the pixel
    count and a 12-megapixel photo is wasted on it - and the coordinates are
    scaled back up. `box` is (x, y, w, h) and `landmarks` is a 5x2 array of
    right eye, left eye, nose tip, right mouth corner, left mouth corner.
    """
    cv2, detector = _detector(verbose=verbose)
    height, width = image_bgr.shape[:2]

    scale = min(1.0, common.FACE_SWAP_DETECT_MAX_SIDE / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            image_bgr,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image_bgr

    detector.setInputSize((small.shape[1], small.shape[0]))
    _retval, faces = detector.detect(small)
    if faces is None or len(faces) == 0:
        return []

    found = []
    for face in faces:
        values = np.asarray(face, dtype=np.float64) / scale
        box = values[:4]
        landmarks = values[4:14].reshape(5, 2)
        found.append((box, landmarks))
    found.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)
    return found


# ---------------------------------------------------------------------------
# The reference face
# ---------------------------------------------------------------------------


def _reference(verbose=True):
    """Load images/authentic/reference.jpg and find the face in it, once.

    The reference is cropped to a generous box around its face, so the warp
    later on moves a few hundred kilopixels rather than a whole photo.
    """
    global _REFERENCE
    if _REFERENCE is not None:
        return _REFERENCE

    path = common.reference_face_path()
    if path is None:
        raise FaceSwapUnavailable(
            f"no {common.REFERENCE_FACE_NAME} in "
            f"{common.rel(common.AUTHENTIC_DIR)}/ - put a photo of the face to "
            "swap in there, next to the images being watermarked"
        )

    try:
        image, _alpha, _info = common.load_rgb(path)
    except Exception as exc:
        raise FaceSwapUnavailable(f"could not read {path.name} ({exc})") from exc

    bgr = np.asarray(image)[:, :, ::-1].copy()
    faces = detect_faces(bgr, verbose=verbose)
    if not faces:
        raise FaceSwapUnavailable(
            f"no face found in {path.name} - use a photo with one clear, "
            "roughly front-on face"
        )

    box, landmarks = faces[0]
    height, width = bgr.shape[:2]
    pad = common.FACE_SWAP_REFERENCE_PAD
    left = max(0, int(box[0] - box[2] * pad))
    top = max(0, int(box[1] - box[3] * pad))
    right = min(width, int(box[0] + box[2] * (1 + pad)))
    bottom = min(height, int(box[1] + box[3] * (1 + pad)))

    crop = bgr[top:bottom, left:right].copy()
    points = landmarks - np.array([left, top], dtype=np.float64)

    if verbose:
        print(f"  Reference face: {path.name}, "
              f"{len(faces)} face(s) found, using the largest "
              f"({int(box[2])}x{int(box[3])} px)")
    _REFERENCE = (crop, points, path)
    return _REFERENCE


# ---------------------------------------------------------------------------
# The swap
# ---------------------------------------------------------------------------


def available(verbose=True):
    """True if the edit can run; prints why not otherwise."""
    try:
        _detector(verbose=verbose)
        _reference(verbose=verbose)
    except FaceSwapUnavailable as exc:
        print(f"  ! face swapping unavailable: {exc}", file=sys.stderr)
        return False
    return True


def swap(img, verbose=False):
    """Replace the faces in `img` with the reference face.

    Raises common.EditNotApplicable when the image has no detectable face,
    which is a property of that image rather than a failure of the run.
    """
    from PIL import Image

    cv2, _detector_obj = _detector(verbose=verbose)
    reference, ref_points, _path = _reference(verbose=verbose)

    target = np.asarray(img.convert("RGB"))[:, :, ::-1].copy()
    faces = detect_faces(target, verbose=verbose)
    if not faces:
        _record(0, 0.0)
        raise common.EditNotApplicable("no face detected")

    faces = faces[: common.FACE_SWAP_MAX_FACES]
    swapped = 0
    covered = np.zeros(target.shape[:2], dtype=bool)

    for box, landmarks in faces:
        mask = _swap_one(cv2, target, reference, ref_points, box, landmarks)
        if mask is None:
            continue
        swapped += 1
        covered |= mask > 0

    if not swapped:
        _record(0, 0.0)
        raise common.EditNotApplicable("faces found but none could be blended")

    _record(swapped, float(covered.mean()))
    return Image.fromarray(target[:, :, ::-1])


def _swap_one(cv2, target, reference, ref_points, box, landmarks):
    """Blend the reference face onto one detected face, in place.

    Returns the mask that was used, or None when the face is too small or too
    close to the border to blend into.
    """
    height, width = target.shape[:2]

    # Similarity transform (rotation + uniform scale + translation) taking the
    # reference's five landmarks onto the target's. A full affine would let the
    # face shear to fit, which looks worse, not better.
    matrix, _inliers = cv2.estimateAffinePartial2D(
        ref_points.reshape(-1, 1, 2).astype(np.float32),
        landmarks.reshape(-1, 1, 2).astype(np.float32),
        method=cv2.LMEDS,
    )
    if matrix is None:
        return None

    warped = cv2.warpAffine(
        reference, matrix, (width, height),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )
    # Where the warped reference actually has pixels; without this the blend
    # can reach past the edge of the reference crop.
    valid = cv2.warpAffine(
        np.full(reference.shape[:2], 255, dtype=np.uint8), matrix,
        (width, height), flags=cv2.INTER_NEAREST,
    )

    mask = np.zeros((height, width), dtype=np.uint8)
    margin = 1.0 + common.FACE_SWAP_MARGIN
    centre = (int(round(box[0] + box[2] / 2)), int(round(box[1] + box[3] / 2)))
    axes = (max(1, int(box[2] / 2 * margin)), max(1, int(box[3] / 2 * margin)))
    cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, -1)
    mask = cv2.bitwise_and(mask, valid)

    # seamlessClone refuses a mask that touches the border, and a one-pixel
    # frame is cheaper than special-casing faces at the edge of the frame.
    mask[0, :] = mask[-1, :] = 0
    mask[:, 0] = mask[:, -1] = 0

    columns = np.flatnonzero(mask.any(axis=0))
    rows = np.flatnonzero(mask.any(axis=1))
    if len(columns) < 8 or len(rows) < 8:  # nothing worth blending
        return None
    blend_centre = (
        int((columns[0] + columns[-1]) // 2),
        int((rows[0] + rows[-1]) // 2),
    )

    if common.FACE_SWAP_SEAMLESS:
        try:
            blended = cv2.seamlessClone(
                warped, target, mask, blend_centre, cv2.NORMAL_CLONE
            )
            target[:] = blended
            return mask
        except cv2.error:
            pass  # fall through to the feathered paste

    # Fallback: feathered alpha blend. Poisson blending is the better swap -
    # it carries the target's lighting - but it is also the part most likely
    # to refuse an awkward face, and a visible seam beats a skipped image.
    feather = max(3, int(min(box[2], box[3]) * common.FACE_SWAP_FEATHER))
    feather += 1 - feather % 2  # GaussianBlur wants an odd kernel
    alpha = cv2.GaussianBlur(mask, (feather, feather), 0)
    alpha = (alpha.astype(np.float32) / 255.0)[:, :, None]
    target[:] = (warped * alpha + target * (1 - alpha)).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _record(faces, area_fraction):
    global _LAST
    _LAST = {"faces": faces, "area_fraction": area_fraction}
    _STATS.append(_LAST)


def describe_last():
    """What the last call to swap() did, for 02_edit.py's per-image line."""
    if not _LAST or not _LAST["faces"]:
        return "no face"
    plural = "" if _LAST["faces"] == 1 else "s"
    return (f"{_LAST['faces']} face{plural}, "
            f"{_LAST['area_fraction'] * 100:.1f}% of pixels")


def run_summary():
    """One line about every swap in this run, or None if there were none."""
    done = [stat for stat in _STATS if stat["faces"]]
    if not done:
        return None
    faces = sum(stat["faces"] for stat in done)
    area = sum(stat["area_fraction"] for stat in done) / len(done)
    return (f"Swapped {faces} face(s) across {len(done)} image(s); the swap "
            f"replaced {area * 100:.1f}% of the pixels on average, and "
            f"{len(_STATS) - len(done)} image(s) had no detectable face.")
