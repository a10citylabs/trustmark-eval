#!/usr/bin/env python3
"""Step 2 - apply the image edits to every watermarked image.

Reads images/watermarked/ and writes one edited copy per edit into
images/modified/, named "<image>__<edit label>.<ext>" so step 3 can tell which
edit produced which file. The edits are defined in common.py:

  * JPEG re-compression   - what upload and share pipelines do
  * downscale             - what a web page or messaging app does
  * centre crop           - a light reframing or border trim
  * unsharp mask, 300%    - aggressive sharpening, no model
  * AI x4 sharpening      - aggressive sharpening by a super-resolution model
  * face swap             - the faces replaced with the reference face

    python 02_edit.py                      # every edit
    python 02_edit.py --edits sharpen_ai_x4 --keep   # redo just the slow one

The AI edit downloads a small model on first use and is by far the slowest
step; --edits and --keep exist so you can rerun it on its own.

The face swap is the one edit that does not apply to every image: a photo with
no face in it is skipped, no file is written, and step 3 scores that edit over
the images it did apply to.
"""

import argparse
import sys
import time
from collections import defaultdict

import common


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--edits",
        help="comma-separated subset of edits to apply "
             f"(default: all of {','.join(common.EDITS)})",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="keep the existing files in images/modified/ instead of clearing "
             "it first; use with --edits to redo one edit",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    common.ensure_dirs()
    common.register_optional_formats()

    sources = common.list_images(common.WATERMARKED_DIR)
    if not sources:
        print(f"No images in {common.rel(common.WATERMARKED_DIR)}/")
        print("Run 01_watermark.py first.")
        return 1

    wanted = [e.strip() for e in args.edits.split(",")] if args.edits else None
    edits = common.select_edits(wanted)
    if not edits:
        print("No edits left to apply.")
        return 1

    print(f"Editing {len(sources)} watermarked image(s) with "
          f"{len(edits)} edit(s):")
    for label in edits:
        print(f"  - {label}: {common.EDIT_DESCRIPTIONS[label]}")
    if common.AI_SHARPEN_LABEL in edits:
        print("\nThe AI edit runs a network over every pixel: seconds per image "
              "on a GPU,\nand up to a minute per 12-megapixel photo on a CPU. "
              "Lower AI_SHARPEN_INPUT_SCALE\nin common.py to trade sharpness "
              "for speed.")
    print()

    if args.keep:
        print(f"Keeping the existing files in {common.rel(common.MODIFIED_DIR)}/")
    else:
        common.clear_folder(common.MODIFIED_DIR)

    written = 0
    failures = 0
    skipped = defaultdict(int)
    elapsed = defaultdict(float)
    started_run = time.perf_counter()

    for index, source in enumerate(sources, start=1):
        # Alpha is dropped here: TrustMark works on RGB, and JPEG cannot hold
        # an alpha channel anyway.
        image, _alpha, info = common.load_rgb(source)
        done = []

        for label, edit in edits.items():
            started = time.perf_counter()
            try:
                edited, suffix, save_kwargs = edit(image)
            except common.EditNotApplicable as reason:
                # Not a failure: this image has nothing for the edit to act on,
                # e.g. a face swap on a photo with no face. No file is written,
                # so step 3 scores this edit over the other images.
                elapsed[label] += time.perf_counter() - started
                skipped[label] += 1
                done.append(f"{label} n/a ({reason})")
                continue
            except Exception as exc:  # one bad edit should not sink the run
                failures += 1
                print(f"  ! {source.name} / {label}: {exc}")
                continue
            elapsed[label] += time.perf_counter() - started

            out_path = (
                common.MODIFIED_DIR
                / f"{source.stem}{common.LABEL_SEP}{label}{suffix}"
            )
            edited.save(out_path, icc_profile=info.get("icc_profile"),
                        **save_kwargs)
            written += 1

            note = common.EDIT_NOTES.get(label)
            detail = note() if note else f"{edited.size[0]}x{edited.size[1]}"
            done.append(f"{label} {detail}")

        # An ETA, because with 100 images the AI edit turns this into a step
        # you walk away from.
        spent = time.perf_counter() - started_run
        eta = ""
        if len(sources) > 1 and index < len(sources) and spent > 20:
            remaining = spent / index * (len(sources) - index)
            eta = f"  [~{remaining / 60:.0f} min left]"
        print(f"  [{index}/{len(sources)}] {source.name}  ->  "
              + ", ".join(done) + eta)

    print(f"\nWrote {written} edited image(s) to "
          f"{common.rel(common.MODIFIED_DIR)}/")
    if failures:
        print(f"{failures} edit(s) failed; the rest are usable.")
    for label, count in skipped.items():
        print(f"{count} of {len(sources)} image(s) had nothing for the "
              f"'{label}' edit to do, so it was scored over "
              f"{len(sources) - count}.")

    if common.FACE_SWAP_LABEL in edits:
        import face_swap

        line = face_swap.run_summary()
        if line:
            print(line)

    if written:
        print("\nTime spent per edit:")
        for label, seconds in elapsed.items():
            print(f"  {label:<24}{seconds:7.1f}s total"
                  f"{seconds / len(sources):8.2f}s per image")

    print("\nNext: python 03_evaluate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
