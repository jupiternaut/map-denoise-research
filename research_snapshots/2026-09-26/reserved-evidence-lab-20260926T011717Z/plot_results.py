"""Render the locked balanced comparisons from sealed evaluator aggregates.

Reads aggregate results only; does not load geometry, masks, labels, or models.
No uncertainty is inferred from point counts or repeated ROI measurements.
Run with the existing open3d-019 Python and -B. All writes stay in figures/.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

sys.dont_write_bytecode = True
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures"
SCENES = (55, 65, 69)
CONDITIONS = ("native", "minus1", "plus1", "minus3", "plus3")
CONDITION_LABELS = ("Native", "−1 mm", "+1 mm", "−3 mm", "+3 mm")
ARMS = ("identity", "previous_normalized", "fit_aug__balanced",
        "reserved_aug__balanced", "both_aug__balanced")
LABELS = ("Identity", "Previous normalized", "Fit-source pairs",
          "Reserved-source pairs", "Both source groups")
SHORT_LABELS = ("Identity", "Previous", "Fit", "Reserved", "Both")
COLORS = ("#272727", "#767676", "#B64342", "#0F4D92", "#42949E")
MARKERS = ("+", "D", "s", "o", "^")
NUMERIC = ("source_MSE_mm2", "source_MAE_mm", "source_p95_mm",
           "improved_fraction", "harmed_fraction", "accepted_fraction",
           "accepted_support_fraction", "moved_fraction", "move_RMS_mm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_and_validate():
    """Independently recompute ROI -> scene means and check SUMMARY exactly."""
    metrics = ROOT / "evaluation/METRICS.csv"
    summary_path = ROOT / "evaluation/SUMMARY.json"
    seal_path = ROOT / "evaluation/SEALED.json"
    seal = json.loads(seal_path.read_text())
    for path in (metrics, summary_path):
        if sha256(path) != seal["files"][path.name]:
            raise ValueError(f"Sealed evaluator artifact changed: {path}")
    summary = json.loads(summary_path.read_text())
    with metrics.open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["arm"] in ARMS]
    index = {}
    for row in rows:
        key = (row["arm"], row["condition"], int(row["scene"]), row["roi"])
        if key in index:
            raise ValueError(f"Duplicate evaluator row: {key}")
        if key[1] not in CONDITIONS or key[2] not in SCENES:
            raise ValueError(f"Unexpected evaluator case: {key}")
        record = {name: float(row[name]) for name in NUMERIC}
        if not np.isfinite(list(record.values())).all():
            raise ValueError(f"Nonfinite evaluator row: {key}")
        index[key] = record
    expected = {(c, s, roi) for arm, c, s, roi in index if arm == "identity"}
    if len(expected) != 60:
        raise ValueError("Expected 60 identity cases")
    for scene in SCENES:
        rois = {roi for c, s, roi in expected if c == "native" and s == scene}
        if len(rois) != 4:
            raise ValueError("Expected four native ROIs per scene")
        for condition in CONDITIONS:
            if {roi for c, s, roi in expected if c == condition and s == scene} != rois:
                raise ValueError("ROI membership changed across conditions")
    for arm in ARMS:
        if {(c, s, roi) for a, c, s, roi in index if a == arm} != expected:
            raise ValueError(f"Case membership differs: {arm}")

    output, checks = [], []
    for arm in ARMS:
        for condition in CONDITIONS:
            item = {"arm": arm, "condition": condition, "n_scenes": 3, "n_rois": 12}
            scenes = {}
            for scene in SCENES:
                rr = [r for (a, c, s, _), r in index.items()
                      if (a, c, s) == (arm, condition, scene)]
                scenes[scene] = {k: float(np.mean([r[k] for r in rr])) for k in NUMERIC}
                for name in NUMERIC:
                    reference = summary["per_scene"][str(scene)][condition][arm][name]
                    difference = abs(scenes[scene][name] - reference)
                    if difference > 1e-12:
                        raise ValueError(f"CSV/SUMMARY scene mismatch: {arm}/{condition}/{scene}/{name}")
                    checks.append(difference)
            for name in NUMERIC:
                item[name] = float(np.mean([scenes[s][name] for s in SCENES]))
            baseline_scenes = [np.mean([r["source_MSE_mm2"] for (a, c, s, _), r in index.items()
                                       if (a, c, s) == ("identity", condition, scene)])
                               for scene in SCENES]
            baseline = float(np.mean(baseline_scenes))
            if baseline <= 0:
                raise ValueError("Identity aggregate MSE must be positive")
            item["MSE_gain_percent"] = 100 * (1 - item["source_MSE_mm2"] / baseline)
            for name in (*NUMERIC, "MSE_gain_percent"):
                reference = summary["exposed_replay"][condition][arm][name]
                difference = abs(item[name] - reference)
                if difference > 1e-10:
                    raise ValueError(f"CSV/SUMMARY aggregate mismatch: {arm}/{condition}/{name}")
                checks.append(difference)
            # Plot the exact sealed aggregate after independent recomputation.
            for name in (*NUMERIC, "MSE_gain_percent"):
                item[name] = summary["exposed_replay"][condition][arm][name]
            output.append(item)
    return output, {"status": "PASS", "numeric_checks": len(checks),
                    "maximum_absolute_difference": max(checks),
                    "aggregation": "four ROI means per scene, then three equally weighted scenes"}


def render(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle

    plt.rcParams.update({
        "font.family": ["DejaVu Sans", "sans-serif"], "font.size": 12,
        "axes.titlesize": 14, "axes.labelsize": 12,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.5, "legend.frameon": False,
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    data = {(r["arm"], r["condition"]): r for r in rows}
    matrix = np.array([[data[(a, c)]["MSE_gain_percent"] for c in CONDITIONS] for a in ARMS])
    fig, (heat, trade) = plt.subplots(1, 2, figsize=(13.8, 6.2),
                                    gridspec_kw={"width_ratios": [1.55, 1], "wspace": 0.37})
    scale = max(1.0, float(np.max(np.abs(matrix))))
    cmap = LinearSegmentedColormap.from_list("harm_gain", ["#B64342", "white", "#3775BA"])
    im = heat.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale), aspect="auto")
    heat.set_xticks(range(5), CONDITION_LABELS)
    heat.set_yticks(range(5), LABELS)
    heat.tick_params(length=0, pad=8)
    for r in range(5):
        for c in range(5):
            value = matrix[r, c]
            heat.text(c, r, f"{value:+.2f}" if value else "0.00", ha="center", va="center",
                      color="white" if abs(value) > scale * 0.59 else "#202020", fontsize=12)
    heat.set_xticks(np.arange(-.5, 5, 1), minor=True)
    heat.set_yticks(np.arange(-.5, 5, 1), minor=True)
    heat.grid(which="minor", color="#DDDDDD", lw=.7)
    heat.tick_params(which="minor", bottom=False, left=False)
    for spine in heat.spines.values():
        spine.set_visible(False)
    heat.add_patch(Rectangle((-.5, 2.5), 5, 1, fill=False, edgecolor=COLORS[3], lw=2))
    heat.get_yticklabels()[3].set_color(COLORS[3])
    heat.get_yticklabels()[3].set_fontweight("bold")
    heat.set_title("A  MSE reduction versus identity (%)", loc="left", pad=18)
    cbar = fig.colorbar(im, ax=heat, orientation="horizontal", fraction=.055, pad=.18, aspect=30)
    cbar.set_label("Negative: higher error     |     Positive: lower error", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    native = [data[(a, "native")] for a in ARMS]
    xs = np.array([100 * r["moved_fraction"] for r in native])
    ys = np.array([100 * r["harmed_fraction"] for r in native])
    for i in range(5):
        trade.scatter(xs[i], ys[i], c=COLORS[i], marker=MARKERS[i], s=95 if i else 110,
                      lw=1.5, label=SHORT_LABELS[i], zorder=4 + i)
    trade.set_xlim(-max(.05, xs.max() * .07), max(.2, xs.max() * 1.15))
    trade.set_ylim(-max(.02, ys.max() * .07), max(.1, ys.max() * 1.15))
    trade.axhline(0, color="#999999", lw=.8, zorder=1)
    trade.axvline(0, color="#999999", lw=.8, zorder=1)
    trade.grid(color="#EEEEEE", lw=.6)
    trade.set_title("B  Native edits and geometric harm", loc="left", pad=18)
    trade.set_xlabel("Moved input rows (%)")
    trade.set_ylabel("Native-support rows harmed by >0.1 mm (%)")
    trade.legend(loc="upper center", bbox_to_anchor=(.5, -.18), ncol=3, fontsize=10,
                 handletextpad=.3, columnspacing=.8)
    fig.suptitle("Reserved evidence for fixed A / KEEP selection", x=.55, y=.98,
                 fontsize=19, fontweight="bold")
    fig.text(.55, .916, "Balanced thresholds · 3 exposed replay scenes × 4 ROIs · ROI → scene equal weight",
             ha="center", fontsize=11, color="#444444")
    fig.text(.55, .043,
             "Primary comparison: reserved-source pairs versus fit-source pairs; same 32 added features.\n"
             "Shared reference and fixed candidates; no per-condition winner selection or point-count uncertainty bars.",
             ha="center", va="center", fontsize=10, color="#444444")
    fig.subplots_adjust(left=.18, right=.975, top=.825, bottom=.29)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"reserved_evidence_balanced.{extension}"
        with path.open("xb") as stream:
            fig.savefig(stream, format=extension, dpi=300, bbox_inches="tight", pad_inches=.15)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main():
    if socket.gethostname() != "liekkas":
        raise RuntimeError("The exact target host is liekkas")
    rows, validation = load_and_validate()
    sources = [ROOT / "plot_results.py", ROOT / "PROTOCOL.md",
               ROOT / "evaluation/METRICS.csv", ROOT / "evaluation/SUMMARY.json",
               ROOT / "evaluation/SEALED.json", ROOT / "training/MODEL_LOCK.json",
               ROOT / "training/THRESHOLDS.json", ROOT / "training/SEALED.json"]
    source_hashes = {str(p): sha256(p) for p in sources}
    OUT.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(OUT / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(OUT / "cache")
    with (OUT / "PLOTTED_VALUES.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    data = {(r["arm"], r["condition"]): r for r in rows}
    contrast = {condition: data[("reserved_aug__balanced", condition)]["MSE_gain_percent"]
                - data[("fit_aug__balanced", condition)]["MSE_gain_percent"] for condition in CONDITIONS}
    write_json(OUT / "VALIDATION.json", {**validation, "reserved_minus_fit_gain_percentage_points": contrast})
    outputs = render(rows)
    for path in sources:
        if sha256(path) != source_hashes[str(path)]:
            raise RuntimeError(f"Source changed while plotting: {path}")
    products = outputs + [OUT / "PLOTTED_VALUES.csv", OUT / "VALIDATION.json"]
    write_json(OUT / "MANIFEST.json", {
        "source_sha256": source_hashes,
        "output_sha256": {p.name: sha256(p) for p in products},
        "arms_fixed_before_reading_outcomes": list(ARMS),
        "aggregation": validation["aggregation"],
        "data_role": "EXPOSED_REPLAY_NOT_NEW_CONFIRMATION",
        "uncertainty": "None inferred from point count; no confidence intervals shown",
        "native_plot_denominators": {"x": "all input rows", "y": "fixed native support rows"},
        "native_edit_definition": "squared displacement > 1e-14 mm²",
        "native_harm_definition": "reference distance increase > 0.1 mm",
        "style_skill": "/home/grf/.codex/skills/scientific-figure-making/SKILL.md",
    })
    print(json.dumps({"figures": [str(p) for p in outputs], "validation": validation,
                      "reserved_minus_fit_pp": contrast}, indent=2))


if __name__ == "__main__":
    main()
