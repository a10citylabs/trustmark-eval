#!/usr/bin/env python3
"""Step 3 - decode everything and score how well the watermark survived.

Decodes the untouched watermarked images (the control) and every edited copy
in images/modified/, compares what comes back against the payloads recorded in
step 1, and writes:

    results/results.csv               one row per image x condition
    results/summary.json              per-edit aggregate metrics
    results/robustness_summary.png    the charts

Three metrics are reported per condition:

  * detection rate  - the decoder reported a watermark is present
  * payload recovery - the recovered bits are exactly the payload embedded
  * raw bit accuracy - agreement with the 100-bit codeword before error
    correction, which is the metric that keeps meaning when recovery fails

    python 03_evaluate.py
"""

import csv
import json
import sys
import textwrap
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # write PNG files, never try to open a window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke

import common

# Palette: categorical slots 1 and 2, ink and surface tokens, and a
# single-hue blue ramp for the heatmap.
C_SERIES_1 = "#2a78d6"
C_SERIES_2 = "#eb6834"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_GRID = "#dedcd6"
C_SURFACE = "#fcfcfb"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95",
            "#0d366b"]


def collect_records(tm, manifest):
    """Decode every image and return one record per image x condition."""
    records = []
    images = manifest["images"]

    # The control: the watermarked PNG, untouched.
    for stem, entry in images.items():
        path = common.WATERMARKED_DIR / entry["watermarked_file"]
        if not path.exists():
            print(f"  ! missing {common.rel(path)}, skipping control for {stem}")
            continue
        records.append(score(tm, path, stem, common.BASELINE_LABEL, entry))

    # The edited copies.
    for path in common.list_images(common.MODIFIED_DIR):
        if common.LABEL_SEP not in path.stem:
            print(f"  ! {path.name} has no '{common.LABEL_SEP}<edit>' label, "
                  "skipping")
            continue
        stem, label = path.stem.rsplit(common.LABEL_SEP, 1)
        entry = images.get(stem)
        if entry is None:
            print(f"  ! {path.name} has no payload in the manifest, skipping")
            continue
        records.append(score(tm, path, stem, label, entry))

    return records


def score(tm, path, stem, label, entry):
    """Decode one file and compare it against the payload that was embedded."""
    image, _alpha, _info = common.load_rgb(path)
    secret = entry["secret"]

    decoded, present, schema = tm.decode(image, MODE="binary")
    raw_bits = common.decode_raw_bits(tm, image)
    accuracy = common.bit_accuracy(common.expected_raw_bits(tm, secret), raw_bits)

    record = {
        "image": stem,
        "condition": label,
        "file": path.name,
        "detected": bool(present),
        "payload_recovered": bool(present) and decoded == secret,
        "raw_bit_accuracy": round(float(accuracy), 4),
        "raw_bit_errors": int(round((1 - accuracy) * len(raw_bits))),
        "schema": int(schema),
        "decoded_secret": decoded,
    }
    status = "OK " if record["payload_recovered"] else (
        "wm only" if present else "FAIL")
    print(f"  {status:8s} {path.name}  bit acc {accuracy * 100:.1f}%")
    return record


def summarise(records):
    """Aggregate the per-image records into per-condition metrics."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)

    summary = {}
    for label in condition_order(grouped):
        rows = grouped[label]
        summary[label] = {
            "description": common.EDIT_DESCRIPTIONS.get(label, label),
            "n_images": len(rows),
            "detection_rate": mean(r["detected"] for r in rows),
            "payload_recovery_rate": mean(r["payload_recovered"] for r in rows),
            "mean_raw_bit_accuracy": mean(r["raw_bit_accuracy"] for r in rows),
            "min_raw_bit_accuracy": min(r["raw_bit_accuracy"] for r in rows),
        }
    return summary


def condition_order(grouped):
    """Baseline first, then the edits in the order common.py defines them."""
    preferred = [common.BASELINE_LABEL] + list(common.EDITS)
    ordered = [label for label in preferred if label in grouped]
    return ordered + sorted(set(grouped) - set(ordered))


def mean(values):
    values = [float(v) for v in values]
    return round(sum(values) / len(values), 4) if values else float("nan")


def write_tables(records, summary):
    common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(common.RESULTS_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    with open(common.SUMMARY_JSON, "w") as handle:
        json.dump(summary, handle, indent=2)


def print_table(summary, threshold):
    header = f"{'condition':<26}{'n':>4}{'detected':>10}{'recovered':>11}{'bit acc':>10}"
    print("\n" + header)
    print("-" * len(header))
    for label, stats in summary.items():
        print(
            f"{label:<26}{stats['n_images']:>4}"
            f"{stats['detection_rate'] * 100:>9.0f}%"
            f"{stats['payload_recovery_rate'] * 100:>10.0f}%"
            f"{stats['mean_raw_bit_accuracy'] * 100:>9.1f}%"
        )
    print(
        f"\nError correction repairs up to {round((1 - threshold) * 100)} of the "
        f"100 raw bits, so payloads survive above ~{threshold * 100:.0f}% bit "
        "accuracy."
    )


def style_axes(ax):
    ax.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_INK_SECONDARY, labelsize=9, length=0)


def plot(records, summary, threshold, capacity):
    labels = list(summary)
    short = [textwrap.fill(label.replace("_", " "), 14) for label in labels]
    images = sorted({r["image"] for r in records})

    heat_ratio = 0.2 + 0.14 * len(images)
    fig_height = 7.4 + 3.6 * heat_ratio
    fig = plt.figure(figsize=(11, fig_height), facecolor=C_SURFACE)
    grid = fig.add_gridspec(3, 1, height_ratios=[1, 1, heat_ratio], hspace=0.6)
    fig.suptitle(
        f"TrustMark robustness  ·  model {common.MODEL_TYPE}  ·  "
        f"{common.ENCODING_NAME} ({capacity}-bit payload)  ·  "
        f"strength {common.WM_STRENGTH}  ·  {len(images)} images",
        color=C_INK_SECONDARY, fontsize=10, x=0.02, ha="left", y=0.995,
    )

    # --- Panel 1: recovery rates -------------------------------------------
    ax = fig.add_subplot(grid[0])
    style_axes(ax)
    x = np.arange(len(labels))
    width = 0.36
    detected = [summary[l]["detection_rate"] * 100 for l in labels]
    recovered = [summary[l]["payload_recovery_rate"] * 100 for l in labels]

    bars_d = ax.bar(x - width / 2 - 0.01, detected, width, label="Watermark detected",
                    color=C_SERIES_1)
    bars_r = ax.bar(x + width / 2 + 0.01, recovered, width,
                    label="Payload recovered exactly", color=C_SERIES_2)
    for bars in (bars_d, bars_r):
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9, color=C_INK)

    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(x, short)
    ax.yaxis.grid(True, color=C_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title("How often the watermark survived", color=C_INK, fontsize=13,
                 pad=34, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=C_INK_SECONDARY, ncols=2,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))

    # --- Panel 2: bit accuracy with per-image points ------------------------
    ax = fig.add_subplot(grid[1])
    style_axes(ax)
    # A dot plot rather than bars: the interesting range sits just below 100%,
    # and bars on a clipped axis would misrepresent the differences.
    rng = np.random.default_rng(0)
    for i, label in enumerate(labels):
        points = [r["raw_bit_accuracy"] * 100 for r in records
                  if r["condition"] == label]
        jitter = rng.uniform(-0.13, 0.13, size=len(points))
        dots = ax.scatter(np.full(len(points), i) + jitter, points, s=34,
                          facecolor=C_SERIES_1, edgecolor=C_SURFACE,
                          linewidth=1.5, alpha=0.9, zorder=3)
        mean_value = summary[label]["mean_raw_bit_accuracy"] * 100
        marker = ax.plot([i - 0.26, i + 0.26], [mean_value] * 2, color=C_INK,
                         linewidth=2.5, solid_capstyle="round", zorder=4)[0]
        ax.annotate(f"{mean_value:.1f}%", (i + 0.3, mean_value), fontsize=9,
                    color=C_INK, va="center", zorder=4,
                    path_effects=[withStroke(linewidth=3, foreground=C_SURFACE)])

    limit = ax.axhline(threshold * 100, color=C_SERIES_2, linewidth=2,
                       linestyle=(0, (4, 3)), zorder=2)

    lowest = min([r["raw_bit_accuracy"] for r in records] + [threshold])
    ax.set_ylim(max(0, lowest * 100 - 6), 101.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xticks(x, short)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")
    ax.yaxis.grid(True, color=C_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title("Raw bit accuracy, before error correction", color=C_INK,
                 fontsize=13, pad=34, loc="left")
    ax.legend(
        [dots, marker, limit],
        ["one image", "mean", f"error correction limit "
                              f"({threshold * 100:.0f}%)"],
        frameon=False, fontsize=9, labelcolor=C_INK_SECONDARY, ncols=3,
        loc="lower left", bbox_to_anchor=(0.0, 1.005),
    )

    # --- Panel 3: per-image heatmap ----------------------------------------
    ax = fig.add_subplot(grid[2])
    lookup = {(r["image"], r["condition"]): r for r in records}
    matrix = np.full((len(images), len(labels)), np.nan)
    for row, image in enumerate(images):
        for col, label in enumerate(labels):
            record = lookup.get((image, label))
            if record:
                matrix[row, col] = record["raw_bit_accuracy"] * 100

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    cmap.set_bad(C_GRID)
    floor = min(90.0, float(np.nanmin(matrix)))
    mesh = ax.imshow(matrix, cmap=cmap, vmin=floor, vmax=100, aspect="auto")

    for row, image in enumerate(images):
        for col, label in enumerate(labels):
            record = lookup.get((image, label))
            if record is None:
                continue
            value = matrix[row, col]
            shade = (value - floor) / max(1e-6, 100 - floor)
            mark = "✓ " if record["payload_recovered"] else (
                "~ " if record["detected"] else "✗ ")
            ax.text(col, row, f"{mark}{value:.0f}%", ha="center", va="center",
                    fontsize=8,
                    color=C_SURFACE if shade > 0.65 else C_INK)

    ax.set_xticks(np.arange(len(labels)), short, fontsize=9)
    ax.set_yticks(np.arange(len(images)), images, fontsize=9)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(images) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=C_SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=C_INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Per-image bit accuracy", color=C_INK, fontsize=13, pad=34,
                 loc="left")
    ax.text(0, 1.012,
            "✓ payload recovered      ~ watermark detected only      "
            "✗ not detected",
            transform=ax.transAxes, fontsize=9, color=C_INK_SECONDARY,
            va="bottom")
    bar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=C_INK_SECONDARY, labelsize=8, length=0)
    bar.ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")

    fig.savefig(common.SUMMARY_PLOT, dpi=160, bbox_inches="tight",
                facecolor=C_SURFACE)
    plt.close(fig)


def main():
    common.ensure_dirs()
    common.register_optional_formats()
    manifest = common.read_manifest()

    if manifest["model_type"] != common.MODEL_TYPE or (
        manifest["encoding"] != common.ENCODING_NAME
    ):
        print(
            f"! The manifest was written with model {manifest['model_type']}/"
            f"{manifest['encoding']} but common.py now says "
            f"{common.MODEL_TYPE}/{common.ENCODING_NAME}. Rerun 01_watermark.py "
            "and 02_edit.py to keep them in step."
        )
        return 1

    tm = common.build_trustmark()
    print(f"\nDecoding {len(manifest['images'])} control image(s) and the "
          f"edited copies in {common.rel(common.MODIFIED_DIR)}/\n")

    records = collect_records(tm, manifest)
    if not records:
        print("Nothing to evaluate. Run 01_watermark.py and 02_edit.py first.")
        return 1

    summary = summarise(records)
    threshold = (100 - common.correctable_bits()) / 100
    write_tables(records, summary)
    print_table(summary, threshold)
    plot(records, summary, threshold, manifest["capacity_bits"])

    print(f"\nWrote {common.rel(common.RESULTS_CSV)}, "
          f"{common.rel(common.SUMMARY_JSON)} and "
          f"{common.rel(common.SUMMARY_PLOT)}")
    print(f"Open the chart with:  open {common.rel(common.SUMMARY_PLOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
