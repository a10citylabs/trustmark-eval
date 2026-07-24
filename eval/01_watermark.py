#!/usr/bin/env python3
"""Step 1 - watermark every image in images/authentic/.

Reads each image from the authentic folder, embeds a unique random payload
with TrustMark, and writes the result to images/watermarked/ as a lossless
PNG. The payloads are recorded in results/payloads.json so step 3 knows the
ground truth to compare the decoded bits against.

    python 01_watermark.py
"""

import random
import sys
from pathlib import Path

import common


def main():
    common.ensure_dirs()
    common.register_optional_formats()

    sources = common.list_images(common.AUTHENTIC_DIR)
    if not sources:
        print(f"No images found in {common.rel(common.AUTHENTIC_DIR)}/")
        print("Add a few JPEG or PNG files there, then run this script again.")
        print(f"Recognised extensions: {sorted(common.IMAGE_EXTENSIONS)}")
        return 1

    print(f"Found {len(sources)} image(s) in {common.rel(common.AUTHENTIC_DIR)}/")
    # New payloads invalidate whatever the previous run produced downstream.
    common.clear_folder(common.WATERMARKED_DIR)
    common.clear_folder(common.MODIFIED_DIR)

    tm = common.build_trustmark()
    capacity = tm.schemaCapacity()
    print(
        f"Model {common.MODEL_TYPE}, schema {common.ENCODING_NAME}, "
        f"payload capacity {capacity} bits, strength {common.WM_STRENGTH}"
    )

    manifest = {
        "model_type": common.MODEL_TYPE,
        "encoding": common.ENCODING_NAME,
        "capacity_bits": capacity,
        "wm_strength": common.WM_STRENGTH,
        "images": {},
    }

    for index, source in enumerate(sources, start=1):
        stem = source.stem
        while stem in manifest["images"]:  # two files with the same stem
            stem = f"{stem}_{index}"
        if common.LABEL_SEP in stem:
            print(f"  ! skipping {source.name}: name must not contain "
                  f"{common.LABEL_SEP!r}")
            continue

        try:
            cover, alpha, info = common.load_rgb(source)
        except Exception as exc:  # unreadable or unsupported file
            print(f"  ! skipping {source.name}: {exc}")
            continue

        # A payload that is unique per image but stable across reruns.
        rng = random.Random(f"{common.RANDOM_SEED}:{stem}")
        secret = "".join(rng.choice("01") for _ in range(capacity))

        watermarked = tm.encode(
            cover, secret, MODE="binary", WM_STRENGTH=common.WM_STRENGTH
        )
        quality_db = common.psnr(cover, watermarked)

        if alpha is not None:
            watermarked.putalpha(alpha)

        out_path = common.WATERMARKED_DIR / f"{stem}.png"
        watermarked.save(
            out_path,
            exif=info.get("exif"),
            icc_profile=info.get("icc_profile"),
            dpi=info.get("dpi", (72, 72)),
        )

        manifest["images"][stem] = {
            "authentic_file": source.name,
            "watermarked_file": out_path.name,
            "secret": secret,
            "psnr_db": round(quality_db, 3),
            "size": list(cover.size),
        }

        print(
            f"  [{index}/{len(sources)}] {source.name} -> {out_path.name}  "
            f"({cover.size[0]}x{cover.size[1]}, PSNR {quality_db:.2f} dB)"
        )

    if not manifest["images"]:
        print("Nothing was watermarked.")
        return 1

    common.write_manifest(manifest)

    scores = [entry["psnr_db"] for entry in manifest["images"].values()]
    finite = [s for s in scores if s != float("inf")]
    print(
        f"\nWatermarked {len(manifest['images'])} image(s) into "
        f"{common.rel(common.WATERMARKED_DIR)}/"
    )
    if finite:
        print(f"Mean PSNR vs the originals: {sum(finite) / len(finite):.2f} dB "
              "(higher means less visible)")
    print(f"Payloads recorded in {common.rel(common.MANIFEST_PATH)}")
    print("\nNext: python 02_edit.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
