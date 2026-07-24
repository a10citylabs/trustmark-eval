#!/usr/bin/env python3
"""Step 2 - apply the image edits to every watermarked image.

Reads images/watermarked/ and writes one edited copy per edit into
images/modified/, named "<image>__<edit label>.<ext>" so step 3 can tell which
edit produced which file. The edits are defined in common.py:

  * JPEG re-compression   - what upload and share pipelines do
  * downscale             - what a web page or messaging app does
  * centre crop           - a light reframing or border trim

    python 02_edit.py
"""

import sys

import common


def main():
    common.ensure_dirs()
    common.register_optional_formats()

    sources = common.list_images(common.WATERMARKED_DIR)
    if not sources:
        print(f"No images in {common.rel(common.WATERMARKED_DIR)}/")
        print("Run 01_watermark.py first.")
        return 1

    print(f"Editing {len(sources)} watermarked image(s) with "
          f"{len(common.EDITS)} edit(s):")
    for label in common.EDITS:
        print(f"  - {label}: {common.EDIT_DESCRIPTIONS[label]}")
    print()

    common.clear_folder(common.MODIFIED_DIR)

    written = 0
    for index, source in enumerate(sources, start=1):
        # Alpha is dropped here: TrustMark works on RGB, and JPEG cannot hold
        # an alpha channel anyway.
        image, _alpha, info = common.load_rgb(source)

        for label, edit in common.EDITS.items():
            edited, suffix, save_kwargs = edit(image)
            out_path = (
                common.MODIFIED_DIR
                / f"{source.stem}{common.LABEL_SEP}{label}{suffix}"
            )
            edited.save(out_path, icc_profile=info.get("icc_profile"),
                        **save_kwargs)
            written += 1
            print(
                f"  [{index}/{len(sources)}] {out_path.name}  "
                f"({edited.size[0]}x{edited.size[1]})"
            )

    print(f"\nWrote {written} edited image(s) to "
          f"{common.rel(common.MODIFIED_DIR)}/")
    print("\nNext: python 03_evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
