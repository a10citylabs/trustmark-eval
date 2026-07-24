#!/usr/bin/env python3
"""Step 3 - decode everything and score how well the watermark survived.

Decodes the untouched watermarked images (the control) and every edited copy
in images/modified/, compares what comes back against the payloads recorded in
step 1, and writes:

    results/results.csv               one row per image x condition
    results/summary.json              per-edit aggregate metrics
    results/robustness_summary.png    the charts

Reported per condition:

  * detection rate  - the decoder reported a watermark is present
  * payload recovery - the recovered bits are exactly the payload embedded
  * raw bit accuracy - agreement with the 100-bit codeword before error
    correction, which is the metric that keeps meaning when recovery fails,
    as a mean, a 10th percentile and a worst case
  * payloads lost - the count behind the recovery rate

The charts adapt to the size of the run: up to 25 images every image is a dot
and every heatmap cell carries its number; past that the columns show
distributions and the heatmap is read as a pattern, sorted hardest-first.

    python 03_evaluate.py
"""

import csv
import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # write PNG files, never try to open a window

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patheffects import withStroke

import common

# Palette: categorical slots 1 and 2, ink and surface tokens, the status red,
# and a diverging red-grey-blue ramp for the heatmap. The heatmap diverges
# rather than ramps because its midpoint means something: the error-correction
# limit separates "payload survives" from "payload is lost".
C_SERIES_1 = "#2a78d6"
C_SERIES_2 = "#eb6834"
C_CRITICAL = "#d03b3b"
C_INK = "#0b0b0b"
C_INK_SECONDARY = "#52514e"
C_MUTED = "#898781"
C_GRID = "#dedcd6"
C_SURFACE = "#fcfcfb"
DIV_BELOW = ["#7d1d1d", "#a82c2c", "#d03b3b", "#e58080", "#f2cccc"]
DIV_NEUTRAL = "#f0efec"
DIV_ABOVE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#0d366b"]

# Above this many images the per-cell numbers stop fitting, so the charts swap
# to their large-sample forms: distributions instead of a dot per image, and a
# heatmap read as a pattern rather than as a table.
LARGE_SAMPLE = 25


def collect_records(tm, manifest):
    """Decode every image and return one record per image x condition."""
    images = manifest["images"]
    jobs = []

    # The control: the watermarked PNG, untouched.
    for stem, entry in images.items():
        path = common.WATERMARKED_DIR / entry["watermarked_file"]
        if not path.exists():
            print(f"  ! missing {common.rel(path)}, skipping control for {stem}")
            continue
        jobs.append((path, stem, common.BASELINE_LABEL, entry))

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
        jobs.append((path, stem, label, entry))

    # 100 images x 6 conditions is 600 decodes, so only the interesting lines
    # are printed in full: anything that did not fully recover, plus a progress
    # line often enough to show the run is alive.
    records = []
    step = max(5, -(-len(jobs) // 10))  # about ten progress lines, whatever n is
    for index, (path, stem, label, entry) in enumerate(jobs, start=1):
        record = score(tm, path, stem, label, entry)
        records.append(record)
        if not record["payload_recovered"]:
            status = "wm only" if record["detected"] else "FAIL"
            print(f"  {status:8s} {path.name}  bit acc "
                  f"{record['raw_bit_accuracy'] * 100:.1f}%")
        elif index % step == 0 or index == len(jobs):
            lost = sum(1 for r in records if not r["payload_recovered"])
            print(f"  ... {index}/{len(jobs)} decoded, {lost} payload(s) lost")
    return records


def score(tm, path, stem, label, entry):
    """Decode one file and compare it against the payload that was embedded."""
    image, _alpha, _info = common.load_rgb(path)
    secret = entry["secret"]

    decoded, present, schema = tm.decode(image, MODE="binary")
    raw_bits = common.decode_raw_bits(tm, image)
    accuracy = common.bit_accuracy(common.expected_raw_bits(tm, secret), raw_bits)

    return {
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


def summarise(records):
    """Aggregate the per-image records into per-condition metrics."""
    grouped = defaultdict(list)
    for record in records:
        grouped[record["condition"]].append(record)

    summary = {}
    for label in condition_order(grouped):
        rows = grouped[label]
        accuracies = sorted(r["raw_bit_accuracy"] for r in rows)
        lost = [r for r in rows if not r["payload_recovered"]]
        summary[label] = {
            "description": common.EDIT_DESCRIPTIONS.get(label, label),
            "n_images": len(rows),
            "detection_rate": mean(r["detected"] for r in rows),
            "payload_recovery_rate": mean(r["payload_recovered"] for r in rows),
            "mean_raw_bit_accuracy": mean(r["raw_bit_accuracy"] for r in rows),
            "median_raw_bit_accuracy": round(float(np.median(accuracies)), 4),
            # The 10th percentile is the number that matters at 100 images: the
            # mean hides a bad tail, and the single worst image is noise.
            "p10_raw_bit_accuracy": round(float(np.percentile(accuracies, 10)), 4),
            "min_raw_bit_accuracy": accuracies[0],
            "n_payloads_lost": len(lost),
            "worst_images": [
                r["image"] for r in sorted(rows, key=lambda r: r["raw_bit_accuracy"])
            ][:5],
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


def print_table(records, summary, threshold):
    header = (f"{'condition':<26}{'n':>4}{'detected':>10}{'recovered':>11}"
              f"{'lost':>6}{'bit acc':>9}{'p10':>8}{'worst':>8}")
    print("\n" + header)
    print("-" * len(header))
    for label, stats in summary.items():
        print(
            f"{label:<26}{stats['n_images']:>4}"
            f"{stats['detection_rate'] * 100:>9.0f}%"
            f"{stats['payload_recovery_rate'] * 100:>10.0f}%"
            f"{stats['n_payloads_lost']:>6}"
            f"{stats['mean_raw_bit_accuracy'] * 100:>8.1f}%"
            f"{stats['p10_raw_bit_accuracy'] * 100:>7.1f}%"
            f"{stats['min_raw_bit_accuracy'] * 100:>7.1f}%"
        )
    print(
        f"\nError correction repairs up to {round((1 - threshold) * 100)} of the "
        f"100 raw bits, so payloads survive above ~{threshold * 100:.0f}% bit "
        "accuracy.\n'p10' is the 10th percentile: a tenth of the images did "
        "worse than that."
    )

    # Which images struggle is the question a 100-image run raises and a
    # per-condition table cannot answer.
    per_image = defaultdict(list)
    for record in records:
        if record["condition"] != common.BASELINE_LABEL:
            per_image[record["image"]].append(record)
    ranked = sorted(
        per_image.items(),
        key=lambda item: sum(r["raw_bit_accuracy"] for r in item[1]) / len(item[1]),
    )
    if len(ranked) > 5:
        print("\nHardest images (mean bit accuracy across the edits):")
        for image, rows in ranked[:5]:
            score_pct = 100 * sum(r["raw_bit_accuracy"] for r in rows) / len(rows)
            lost = sum(1 for r in rows if not r["payload_recovered"])
            print(f"  {image:<40}{score_pct:>6.1f}%   "
                  f"{lost}/{len(rows)} payload(s) lost")


def style_axes(ax):
    ax.set_facecolor(C_SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(C_GRID)
    ax.tick_params(colors=C_INK_SECONDARY, labelsize=9, length=0)


def rank_images(records):
    """Images ordered worst-first by mean bit accuracy across the edits.

    Sorting is what makes a 100-row heatmap readable: unsorted it is noise,
    sorted it shows at a glance whether failures are spread across the set or
    concentrated in a handful of difficult images.
    """
    scores = defaultdict(list)
    for record in records:
        if record["condition"] != common.BASELINE_LABEL:
            scores[record["image"]].append(record["raw_bit_accuracy"])
    for record in records:  # images that only have a baseline row
        scores.setdefault(record["image"], [record["raw_bit_accuracy"]])
    return sorted(scores, key=lambda image: (sum(scores[image]) / len(scores[image]),
                                             image))


def panel_rates(ax, labels, short, summary):
    """Headline rates: how often the watermark was found and read back."""
    style_axes(ax)
    x = np.arange(len(labels))
    width = 0.30
    detected = [summary[l]["detection_rate"] * 100 for l in labels]
    recovered = [summary[l]["payload_recovery_rate"] * 100 for l in labels]

    bars_d = ax.bar(x - width / 2 - 0.01, detected, width,
                    label="Watermark detected", color=C_SERIES_1)
    bars_r = ax.bar(x + width / 2 + 0.01, recovered, width,
                    label="Payload recovered exactly", color=C_SERIES_2)
    ax.bar_label(bars_d, fmt="%.0f%%", padding=3, fontsize=9, color=C_INK)
    # The recovery bar carries a count as well as a rate, so that "78%" reads
    # as "22 images lost their payload" rather than as a rate with no
    # denominator - which is the difference between 8 images and 100.
    ax.bar_label(
        bars_r, padding=3, fontsize=9, color=C_INK,
        labels=[
            f"{lost} lost\n{rate:.0f}%" if lost else f"{rate:.0f}%"
            for rate, lost in zip(
                recovered, (summary[l]["n_payloads_lost"] for l in labels))
        ],
    )

    ax.set_ylim(0, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(x, short)
    ax.yaxis.grid(True, color=C_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title("How often the watermark survived", color=C_INK, fontsize=13,
                 pad=34, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=C_INK_SECONDARY, ncols=2,
              loc="lower left", bbox_to_anchor=(0.0, 1.005))


def panel_accuracy(ax, labels, short, records, summary, threshold, large):
    """Distribution of raw bit accuracy, one column per condition.

    Under ~25 images every image is a dot. Past that the dots stop being
    countable, so each column also gets the shape of its distribution and its
    interquartile range - which is where the answer lives once the mean has
    stopped being informative.
    """
    style_axes(ax)
    x = np.arange(len(labels))
    columns = [[r["raw_bit_accuracy"] * 100 for r in records
                if r["condition"] == label] for label in labels]

    if large:
        # gaussian_kde needs some spread; a column where every image scored
        # the same (the baseline, usually) has none, so it is skipped.
        spread = [(i, values) for i, values in enumerate(columns)
                  if len(values) > 1 and np.ptp(values) > 1e-9]
        if spread:
            bodies = ax.violinplot([values for _i, values in spread],
                                   positions=[i for i, _v in spread],
                                   widths=0.74, showextrema=False,
                                   showmedians=False)
            for body in bodies["bodies"]:
                body.set_facecolor(C_SERIES_1)
                body.set_edgecolor("none")
                body.set_alpha(0.16)
                body.set_zorder(1)

    rng = np.random.default_rng(0)
    dot_size, dot_alpha = (10, 0.5) if large else (34, 0.9)
    for i, values in enumerate(columns):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        dots = ax.scatter(np.full(len(values), i) + jitter, values, s=dot_size,
                          facecolor=C_SERIES_1, edgecolor=C_SURFACE,
                          linewidth=1.5 if not large else 0.4, alpha=dot_alpha,
                          zorder=3)
        if large:
            q1, q3 = np.percentile(values, [25, 75])
            spread_bar = ax.plot([i, i], [q1, q3], color=C_SERIES_1, linewidth=5,
                                 alpha=0.55, solid_capstyle="round", zorder=4)[0]
        mean_value = summary[labels[i]]["mean_raw_bit_accuracy"] * 100
        marker = ax.plot([i - 0.26, i + 0.26], [mean_value] * 2, color=C_INK,
                         linewidth=2.5, solid_capstyle="round", zorder=5)[0]
        ax.annotate(f"{mean_value:.1f}%", (i + 0.3, mean_value), fontsize=9,
                    color=C_INK, va="center", zorder=5,
                    path_effects=[withStroke(linewidth=3, foreground=C_SURFACE)])

    limit = ax.axhline(threshold * 100, color=C_CRITICAL, linewidth=2,
                       linestyle=(0, (4, 3)), zorder=2)

    lowest = min([v for values in columns for v in values] + [threshold * 100])
    ax.set_ylim(max(0, lowest - 6), 101.5)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xticks(x, short)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")
    ax.yaxis.grid(True, color=C_GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.set_title("Raw bit accuracy, before error correction", color=C_INK,
                 fontsize=13, pad=34, loc="left")

    handles = [dots, marker, limit]
    names = ["one image", "mean",
             f"error correction limit ({threshold * 100:.0f}%)"]
    if large:
        handles.insert(1, spread_bar)
        names.insert(1, "middle 50% of images")
    ax.legend(handles, names, frameon=False, fontsize=9,
              labelcolor=C_INK_SECONDARY, ncols=len(handles), loc="lower left",
              bbox_to_anchor=(0.0, 1.005))


def heat_colormap():
    """Diverging ramp: red below the error-correction limit, blue above it.

    Equal numbers of steps each side of a neutral midpoint, so a TwoSlopeNorm
    centred on the limit puts the grey exactly at the pass/fail boundary.
    """
    return matplotlib.colors.LinearSegmentedColormap.from_list(
        "ecc_diverging", DIV_BELOW + [DIV_NEUTRAL] + DIV_ABOVE
    )


def mark_ink(shade):
    """Ink for text or a marker sitting on a cell at this point of the ramp."""
    return C_SURFACE if shade > 0.82 or shade < 0.12 else C_INK


def panel_heatmap(fig, ax, labels, short, records, images, threshold, large,
                  row_height):
    """One row per image, one column per condition, sorted worst-first."""
    lookup = {(r["image"], r["condition"]): r for r in records}
    matrix = np.full((len(images), len(labels)), np.nan)
    for row, image in enumerate(images):
        for col, label in enumerate(labels):
            record = lookup.get((image, label))
            if record:
                matrix[row, col] = record["raw_bit_accuracy"] * 100

    centre = threshold * 100
    cmap = heat_colormap()
    cmap.set_bad(C_GRID)
    norm = matplotlib.colors.TwoSlopeNorm(
        vcenter=centre,
        vmin=min(float(np.nanmin(matrix)), centre - 5),
        vmax=max(float(np.nanmax(matrix)), centre + 1),
    )
    mesh = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    if large:
        # Numbers in 100 x 6 cells are unreadable; the colour carries the
        # value and a marker calls out the cells that actually failed. Marker
        # ink flips with the cell it sits on, the same way the numbers do.
        lost, gone = [], []
        for row, image in enumerate(images):
            for col, label in enumerate(labels):
                record = lookup.get((image, label))
                if record is None or record["payload_recovered"]:
                    continue
                ink = mark_ink(norm(matrix[row, col]))
                (lost if record["detected"] else gone).append((col, row, ink))
        if lost:
            cols, rows, inks = zip(*lost)
            ax.scatter(cols, rows, marker="o", s=9, facecolor="none",
                       edgecolor=inks, linewidth=1.0, zorder=3)
        if gone:
            cols, rows, inks = zip(*gone)
            ax.scatter(cols, rows, marker="x", s=12, color=inks, linewidth=1.0,
                       zorder=3)
        legend = ("colour = bit accuracy      ○ payload lost, watermark still "
                  "detected      ✕ watermark not detected")
    else:
        for row, image in enumerate(images):
            for col, label in enumerate(labels):
                record = lookup.get((image, label))
                if record is None:
                    continue
                value = matrix[row, col]
                mark = "✓ " if record["payload_recovered"] else (
                    "~ " if record["detected"] else "✗ ")
                ax.text(col, row, f"{mark}{value:.0f}%", ha="center",
                        va="center", fontsize=8, color=mark_ink(norm(value)))
        legend = ("✓ payload recovered      ~ watermark detected only      "
                  "✗ not detected")

    ax.set_xticks(np.arange(len(labels)), short, fontsize=9)
    ax.set_xticks(np.arange(len(labels) + 1) - 0.5, minor=True)
    if row_height * 72 >= 10:  # rows tall enough for a label on each
        ax.set_yticks(np.arange(len(images)), images, fontsize=9)
    else:
        # Row labels do not fit; the worst handful get named instead, fanned
        # out into the margin with a leader line back to their row.
        ax.set_yticks([])
        label_rows(ax, images, 8, row_height)
    if large:
        ax.grid(which="minor", axis="x", color=C_SURFACE, linewidth=2)
    else:
        ax.set_yticks(np.arange(len(images) + 1) - 0.5, minor=True)
        ax.grid(which="minor", color=C_SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0, colors=C_INK_SECONDARY)
    for spine in ax.spines.values():
        spine.set_visible(False)

    title = (f"Per-image bit accuracy  ·  {len(images)} images, "
             "hardest at the top")
    ax.set_title(title, color=C_INK, fontsize=13, pad=34, loc="left")
    ax.text(0, 1.012, legend, transform=ax.transAxes, fontsize=9,
            color=C_INK_SECONDARY, va="bottom")

    bar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    bar.outline.set_visible(False)
    bar.ax.tick_params(colors=C_INK_SECONDARY, labelsize=8, length=0)
    bar.ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:.0f}%")
    bar.ax.axhline(centre, color=C_INK, linewidth=1.2)
    bar.ax.annotate("ECC limit", (0.5, centre), xycoords=("axes fraction", "data"),
                    xytext=(0, 5), textcoords="offset points", ha="center",
                    fontsize=7, color=C_INK)


def label_rows(ax, images, count, row_height):
    """Name the `count` worst rows in the left margin, with leader lines.

    Rows are only a couple of points tall at 100 images, so the labels are
    fanned downwards to a legible spacing and each one keeps a leader line
    back to the row it belongs to.
    """
    count = min(count, len(images))
    if count <= 0:
        return
    row_points = row_height * 72
    fan = max(0.0, 10.0 - row_points)  # extra points of spacing per label
    for rank, image in enumerate(images[:count]):
        ax.annotate(
            image, xy=(-0.5, rank), xytext=(-58, -rank * fan),
            textcoords="offset points", fontsize=8, ha="right", va="center",
            color=C_INK_SECONDARY, annotation_clip=False,
            arrowprops=dict(arrowstyle="-", color=C_GRID, linewidth=0.8,
                            shrinkA=2, shrinkB=1),
        )
    ax.annotate(f"↑ the {count} hardest images", xy=(-0.5, count - 1),
                xytext=(-58, -(count - 1) * fan - 13),
                textcoords="offset points", fontsize=8, ha="right", va="top",
                color=C_MUTED, annotation_clip=False)


def plot(records, summary, threshold, capacity):
    labels = list(summary)
    short = [common.short_name(label) for label in labels]
    images = rank_images(records)
    large = len(images) > LARGE_SAMPLE

    # Panel heights in inches, so the figure grows with the number of images
    # instead of the aspect ratio running away with it: at 8 images each row is
    # a comfortable 0.3", at 100 the block is capped and rows get thin.
    rates_h, dist_h = 3.2, 3.8
    row_h = 0.30 if not large else max(0.06, min(0.30, 7.5 / len(images)))
    heat_h = 1.0 + row_h * len(images)
    gap = 1.4  # room between panels for the next one's title and legend
    top_h, bottom_h = 0.95, 0.6  # the suptitle, and the last x tick labels
    panels = [rates_h, dist_h, heat_h]

    fig_w = max(11.0, 1.7 * len(labels) + 3.4)
    fig_h = sum(panels) + 2 * gap + top_h + bottom_h
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=C_SURFACE)
    grid = fig.add_gridspec(
        3, 1, height_ratios=panels,
        hspace=gap * len(panels) / sum(panels),
        top=1 - top_h / fig_h, bottom=bottom_h / fig_h,
        left=0.8 / fig_w, right=1 - 0.3 / fig_w,
    )
    fig.suptitle(
        f"TrustMark robustness  ·  model {common.MODEL_TYPE}  ·  "
        f"{common.ENCODING_NAME} ({capacity}-bit payload)  ·  "
        f"strength {common.WM_STRENGTH}  ·  {len(images)} images",
        color=C_INK_SECONDARY, fontsize=10, x=0.02, ha="left", y=0.998,
    )

    panel_rates(fig.add_subplot(grid[0]), labels, short, summary)
    panel_accuracy(fig.add_subplot(grid[1]), labels, short, records, summary,
                   threshold, large)
    panel_heatmap(fig, fig.add_subplot(grid[2]), labels, short, records, images,
                  threshold, large, row_h)

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
    print_table(records, summary, threshold)
    plot(records, summary, threshold, manifest["capacity_bits"])

    print(f"\nWrote {common.rel(common.RESULTS_CSV)}, "
          f"{common.rel(common.SUMMARY_JSON)} and "
          f"{common.rel(common.SUMMARY_PLOT)}")
    print(f"Open the chart with:  open {common.rel(common.SUMMARY_PLOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
