"""Render the predeclared replay figure from aggregate evaluator output.

This script reads no reference geometry, models, scores, or decision masks.
It reads locked threshold status solely to label infeasibility; it neither fits
models nor changes thresholds. All plotted arms are fixed below.
"""

from __future__ import annotations

import argparse
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
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "evaluation" / "METRICS.csv"
THRESHOLDS = ROOT / "training" / "THRESHOLDS.json"
OUT = ROOT / "figures"
CONDITIONS = ("native", "minus1", "plus1", "minus3", "plus3")
CONDITION_LABELS = ("Native", "−1 mm", "+1 mm", "−3 mm", "+3 mm")
SCENES = (55, 65, 69)
METHODS = (
    "photo_cost", "mode_gap", "source_agreement", "small_displacement",
    "absolute_confidence_hgb", "candidate_error_hgb", "gain_sign_hgb",
    "direct_gain", "normalized_gain", "benefit_harm", "hurdle_gain",
)
REFERENCES = ("identity", "A_all", "frozen_gain")
MAIN_ARMS = REFERENCES + tuple(name + "__balanced" for name in METHODS)
FOCUS_METHODS = (
    "absolute_confidence_hgb", "direct_gain", "normalized_gain", "benefit_harm", "hurdle_gain",
)
LABELS = {
    "identity": "Identity", "A_all": "A: move all", "frozen_gain": "Frozen gain",
    "photo_cost": "Photo cost", "mode_gap": "Mode gap", "source_agreement": "Source agreement",
    "small_displacement": "Small displacement", "absolute_confidence_hgb": "Absolute confidence",
    "candidate_error_hgb": "Candidate error", "gain_sign_hgb": "Gain sign",
    "direct_gain": "Direct gain", "normalized_gain": "Normalized gain · lead",
    "benefit_harm": "Benefit / harm", "hurdle_gain": "Hurdle gain",
}
PALETTE = {
    "blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_3": "#8BCF8B",
    "red_strong": "#B64342", "neutral": "#CFCECE", "teal": "#42949E", "violet": "#9A4D8E",
}
FOCUS_COLORS = {
    "absolute_confidence_hgb": PALETTE["red_strong"],
    "direct_gain": PALETTE["teal"],
    "normalized_gain": PALETTE["blue_main"],
    "benefit_harm": "#4A8B43", "hurdle_gain": PALETTE["violet"],
    "frozen_gain": "#4D4D4D",
}
FOCUS_MARKERS = {"absolute_confidence_hgb": "s", "direct_gain": "^", "normalized_gain": "o",
                 "benefit_harm": "P", "hurdle_gain": "v", "frozen_gain": "D"}
OPTIONAL_METRICS = (
    "source_MAE_mm", "improved_fraction", "harmed_fraction", "moved_fraction",
    "benefit_sum_mm2", "harm_sum_mm2",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_publication_style() -> None:
    plt.rcParams.update({
        # DejaVu is bundled in this environment; missing commercial fonts create
        # per-label lookup noise without changing the intended sans-serif style.
        "font.family": ["DejaVu Sans", "sans-serif"],
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 1.5, "legend.frameon": False, "svg.fonttype": "none",
        "pdf.fonttype": 42, "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def load_records(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"scene", "roi", "condition", "arm", "source_MSE_mm2"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing metric columns: {sorted(required - set(reader.fieldnames or []))}")
        records = []
        for row in reader:
            record = {
                "scene": int(row["scene"].removeprefix("scan")), "roi": str(row["roi"]),
                "condition": row["condition"], "arm": row["arm"],
                "source_MSE_mm2": float(row["source_MSE_mm2"]),
            }
            for name in OPTIONAL_METRICS:
                if row.get(name, "") != "":
                    record[name] = float(row[name])
            records.append(record)
    return records


def index_and_validate(records: list[dict]) -> tuple[dict, list[str]]:
    if not records:
        raise ValueError("The evaluation CSV is empty")
    index = {}
    for record in records:
        key = (record["arm"], record["scene"], record["roi"], record["condition"])
        if key in index:
            raise ValueError(f"Duplicate evaluator row: {key}")
        if record["scene"] not in SCENES or record["condition"] not in CONDITIONS:
            raise ValueError(f"Unexpected replay case: {key}")
        if not record["roi"] or not np.isfinite(record["source_MSE_mm2"]) or record["source_MSE_mm2"] < 0:
            raise ValueError(f"Invalid MSE/ROI: {key}")
        for field in OPTIONAL_METRICS:
            if field in record and not np.isfinite(record[field]):
                raise ValueError(f"Nonfinite optional metric {field}: {key}")
        index[key] = record
    arms = sorted({record["arm"] for record in records})
    required = set(MAIN_ARMS) | {name + "__native_priority" for name in METHODS}
    if missing := required - set(arms):
        raise ValueError(f"Missing predeclared arms: {sorted(missing)}")
    identity_cases = {(s, roi, c) for arm, s, roi, c in index if arm == "identity"}
    if len(identity_cases) != 60:
        raise ValueError("Identity must contain exactly 60 replay cases")
    for scene in SCENES:
        native_rois = {roi for s, roi, c in identity_cases if s == scene and c == "native"}
        if len(native_rois) != 4:
            raise ValueError(f"Scene {scene} must have four native ROIs")
        for condition in CONDITIONS:
            if {roi for s, roi, c in identity_cases if s == scene and c == condition} != native_rois:
                raise ValueError("Each scene must reuse its four ROIs across conditions")
    for arm in arms:
        cases = {(s, roi, c) for a, s, roi, c in index if a == arm}
        if cases != identity_cases:
            raise ValueError(f"Incomplete or mismatched cases for {arm}: {len(cases)}")
    return index, arms


def aggregate(records: list[dict]) -> tuple[dict, list[dict]]:
    index, arms = index_and_validate(records)
    result, rows = {}, []
    for arm in arms:
        for condition in CONDITIONS:
            by_scene, baseline_by_scene, case_relative, wins, ties, optional = [], [], [], [], [], {}
            for scene in SCENES:
                rois = sorted(roi for a, s, roi, c in index if a == "identity" and s == scene and c == condition)
                actual, baseline = [], []
                for roi in rois:
                    record = index[(arm, scene, roi, condition)]
                    value = record["source_MSE_mm2"]
                    ref = index[("identity", scene, roi, condition)]["source_MSE_mm2"]
                    actual.append(value)
                    baseline.append(ref)
                    case_relative.append(100.0 * (ref - value) / max(ref, 1e-6))
                    wins.append(value < ref)
                    ties.append(value == ref)
                    for name in OPTIONAL_METRICS:
                        if name in record:
                            optional.setdefault(name, []).append(record[name])
                by_scene.append(float(np.mean(actual)))
                baseline_by_scene.append(float(np.mean(baseline)))
            mean_mse, reference_mse = float(np.mean(by_scene)), float(np.mean(baseline_by_scene))
            # Full balanced design: ROI-then-scene averaging equals case averaging.
            summary = {
                "arm": arm, "condition": condition, "scene_count": len(SCENES), "roi_case_count": len(wins),
                "mean_MSE_mm2": mean_mse, "identity_mean_MSE_mm2": reference_mse,
                "relative_gain_pct": 100.0 * (reference_mse - mean_mse) / max(reference_mse, 1e-6),
                "mean_case_relative_gain_pct": float(np.mean(case_relative)),
                "roi_win_count": int(sum(wins)), "roi_tie_count": int(sum(ties)),
                "roi_loss_count": int(len(wins) - sum(wins) - sum(ties)),
                "roi_win_fraction": float(np.mean(wins)),
                "scene_win_count": int(np.sum(np.array(by_scene) < baseline_by_scene)),
            }
            for name in OPTIONAL_METRICS:
                if name in optional:
                    if len(optional[name]) != len(wins):
                        raise ValueError(f"Partial optional metric {name} for {arm}/{condition}")
                    summary["mean_" + name] = float(np.mean(optional[name]))
            rows.append(summary)
            result[(arm, condition)] = summary
    return result, rows


def display_label(arm: str) -> str:
    return LABELS[arm.split("__", 1)[0]]


def make_report(summary: dict):
    apply_publication_style()
    fig = plt.figure(figsize=(22.5, 10.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.13], width_ratios=[1.42, 0.82, 1.45],
                           hspace=0.18, wspace=0.29)
    heat = fig.add_subplot(grid[0, 0])
    wins_ax = fig.add_subplot(grid[0, 1])
    trade = fig.add_subplot(grid[0, 2])
    legend_ax = fig.add_subplot(grid[1, 2])
    legend_ax.set_axis_off()
    n = len(MAIN_ARMS)
    matrix = np.array([[summary[(arm, condition)]["relative_gain_pct"] for condition in CONDITIONS]
                       for arm in MAIN_ARMS])
    # Symmetric limits retain every failure without clipping the color scale.
    scale = max(1.0, float(np.max(np.abs(matrix))))
    cmap = LinearSegmentedColormap.from_list("harm_white_gain", ["#B64342", "#FFFFFF", "#3775BA"])
    image = heat.imshow(matrix, cmap=cmap, norm=TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale), aspect="auto")
    heat.set_xticks(np.arange(len(CONDITIONS)), CONDITION_LABELS)
    heat.set_yticks(np.arange(n), [display_label(arm) for arm in MAIN_ARMS])
    heat.tick_params(length=0, pad=8)
    heat.set_title("A   MSE reduction versus identity (%)\nAll methods · balanced thresholds", loc="left", pad=18)
    for row in range(n):
        for col in range(len(CONDITIONS)):
            value = matrix[row, col]
            ink = "white" if abs(value) > 0.58 * scale else "#202020"
            label = "0.0" if value == 0 else (f"{value:+.1e}" if abs(value) < 0.05 else f"{value:+.1f}")
            heat.text(col, row, label, ha="center", va="center", color=ink, fontsize=11)
    heat.set_xticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
    heat.set_yticks(np.arange(-0.5, n, 1), minor=True)
    heat.grid(which="minor", color="#DDDDDD", linewidth=0.65)
    heat.tick_params(which="minor", bottom=False, left=False)
    for spine in heat.spines.values():
        spine.set_visible(False)
    lead_row = MAIN_ARMS.index("normalized_gain__balanced")
    heat.add_patch(Rectangle((-0.5, lead_row - 0.5), 5, 1, fill=False, edgecolor=PALETTE["blue_main"], linewidth=2.2))
    heat.get_yticklabels()[lead_row].set_color(PALETTE["blue_main"])
    heat.get_yticklabels()[lead_row].set_fontweight("bold")
    colorbar_ax = fig.add_subplot(grid[1, 0])
    cbar = fig.colorbar(image, cax=colorbar_ax, orientation="horizontal")
    cbar.set_label("Negative: harm    |    Positive: improvement", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    y = np.arange(n)
    for condition, marker, offset, color, label in (
        ("native", "o", -0.13, PALETTE["blue_main"], "Native"),
        ("plus3", "s", 0.13, PALETTE["red_strong"], "+3 mm"),
    ):
        fractions = np.array([summary[(arm, condition)]["roi_win_fraction"] for arm in MAIN_ARMS])
        wins_ax.scatter(fractions * 100.0, y + offset, marker=marker, s=45, color=color,
                        edgecolors="white", linewidths=0.6, label=label, zorder=3)
    wins_ax.set_ylim(n - 0.5, -0.5)
    wins_ax.set_xlim(-7, 107)
    wins_ax.set_yticks(y, [""] * n)
    wins_ax.tick_params(axis="y", length=0)
    wins_ax.set_xticks([0, 25, 50, 75, 100])
    wins_ax.grid(axis="x", color="#DDDDDD", linewidth=0.75)
    for row in (2.5, 6.5, 9.5):
        heat.axhline(row, color="#888888", linewidth=1.2)
        wins_ax.axhline(row, color="#BBBBBB", linewidth=0.8)
    wins_ax.set_title("B   ROI win fraction\n12 scene–ROI pairs", loc="left", pad=18)
    wins_ax.set_xlabel("Strict MSE wins versus identity (%)")
    wins_ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=11)
    wins_ax.spines["left"].set_visible(False)

    coordinates = []
    for method in FOCUS_METHODS:
        values = []
        for setting in ("balanced", "native_priority"):
            arm = method + "__" + setting
            xy = (summary[(arm, "native")]["relative_gain_pct"], summary[(arm, "plus3")]["relative_gain_pct"])
            values.append(xy)
            coordinates.append(xy)
        color, marker = FOCUS_COLORS[method], FOCUS_MARKERS[method]
        if not np.array_equal(values[0], values[1]):
            trade.annotate("", xy=values[1], xytext=values[0],
                           arrowprops={"arrowstyle": "->", "color": color, "lw": 1.2, "alpha": 0.8}, zorder=2)
        # The smaller filled balanced point stays visible when both settings coincide.
        trade.scatter(*values[1], marker=marker, s=135, facecolors="white", edgecolors=color,
                      linewidths=1.8, zorder=4)
        trade.scatter(*values[0], marker=marker, s=48, facecolors=color, edgecolors=color,
                      linewidths=0.8, zorder=5)
    frozen = (summary[("frozen_gain", "native")]["relative_gain_pct"],
              summary[("frozen_gain", "plus3")]["relative_gain_pct"])
    coordinates.append(frozen)
    trade.scatter(*frozen, marker="D", s=70, color=FOCUS_COLORS["frozen_gain"], zorder=6)
    trade.axhline(0, color="#777777", linewidth=1.0, linestyle="--", zorder=1)
    trade.axvline(0, color="#777777", linewidth=1.0, linestyle="--", zorder=1)
    points = np.vstack([coordinates, [0.0, 0.0]])
    for axis, column in ((trade.set_xlim, 0), (trade.set_ylim, 1)):
        lo, hi = float(np.min(points[:, column])), float(np.max(points[:, column]))
        margin = max((hi - lo) * 0.12, 1.5)
        axis(lo - margin, hi + margin)
    trade.set_title("C   Native risk and +3 mm recovery\nFixed gain family and confidence comparator", loc="left", pad=18)
    trade.set_xlabel("Native MSE reduction versus identity (%)")
    trade.set_ylabel("+3 mm MSE reduction versus identity (%)")
    trade.text(0.03, 0.98, "Filled: balanced     Open: native priority\nArrow: balanced → native priority",
               transform=trade.transAxes, ha="left", va="top", fontsize=10, color="#454545")
    trade.text(0.03, 0.03, "All native-priority constraints infeasible → KEEP\nThe origin is no recovery, not a safe improvement.",
               transform=trade.transAxes, ha="left", va="bottom", fontsize=10, color="#454545",
               bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 3})
    handles = [Line2D([], [], color=FOCUS_COLORS[name], marker=FOCUS_MARKERS[name], linestyle="none",
                      markersize=7, label=LABELS[name]) for name in (*FOCUS_METHODS, "frozen_gain")]
    legend_ax.legend(handles=handles, loc="center", ncol=2, fontsize=11, columnspacing=1.2, handletextpad=0.5)
    note_ax = fig.add_subplot(grid[1, 1])
    note_ax.set_axis_off()
    note_ax.text(0.0, 0.18, "No per-condition winner selection.\nAll negative results retained.", fontsize=10,
                 color="#454545", ha="left", va="center")
    fig.suptitle("Exposed replay · 3 scenes · fixed A candidate", fontsize=22, fontweight="bold", y=0.982)
    fig.text(0.5, 0.938, "ROI then scene equal weight · native and injected conditions remain separate · thresholds fixed on development scenes",
             ha="center", fontsize=12, color="#444444")
    fig.subplots_adjust(left=0.135, right=0.985, top=0.855, bottom=0.105)
    return fig


def finalize_figure(fig, out_path: Path, formats=("png", "pdf", "svg"), dpi=300) -> list[Path]:
    result = []
    for extension in formats:
        path = out_path.with_suffix("." + extension)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.14)
        result.append(path)
    plt.close(fig)
    return result


def caption(summary: dict, source_hash: str) -> str:
    balanced_native = summary[("normalized_gain__balanced", "native")]["relative_gain_pct"]
    balanced_plus3 = summary[("normalized_gain__balanced", "plus3")]["relative_gain_pct"]
    priority_native = summary[("normalized_gain__native_priority", "native")]["relative_gain_pct"]
    priority_plus3 = summary[("normalized_gain__native_priority", "plus3")]["relative_gain_pct"]
    return f"""# Exposed replay: fixed-A selection study

This is historical, exposed replay of scenes 55, 65 and 69, with four ROIs per
scene and five input conditions (60 cases per arm). It is not independent
confirmation, and 12 scene–ROI pairs are not 12 independent scenes.

**A.** All 11 predeclared selectors use their development-calibrated balanced
threshold, alongside identity, moving every A candidate, and the old frozen gain
policy. Each cell is 100 × (mean identity MSE − mean method MSE) / mean identity
MSE, with a denominator floor of 1e-6 mm². MSE is averaged over ROIs within each
scene and then equally over scenes. Positive values mean improvement, negative
values mean harm. The symmetric color scale includes the full observed range;
no negative value is clipped or omitted. The outlined normalized-gain row is the
predeclared exploratory lead, not a winner chosen from replay.

**B.** Fractions of the 12 matched scene–ROI cases with strictly smaller MSE than
identity, for native input and +3 mm. Rows align with panel A. Ties, including an
identity/KEEP output, are not wins. This is a descriptive fraction, not an
independent-sample significance analysis or a point-level benefit fraction.

**C.** Native and +3 mm relative MSE reductions for the four predeclared gain
constructions and the absolute-confidence comparator. Filled symbols are balanced
operating points; open symbols are the predeclared native-priority points; arrows
connect these two settings of the same method. Coincident settings show a filled
center within the open marker. The gray diamond is the old frozen policy with
its existing threshold. **All 11 native-priority constraints were infeasible on
development data, so all returned KEEP.** Their overlapping origin points are
identity outcomes with no recovery, not successful safe policies. The locked
threshold-status JSON is checked to substantiate this annotation. There is no condition-wise model or
threshold selection. Per-head capacity is fixed, but benefit/harm uses up to two
heads and hurdle gain up to three, versus one for direct and normalized gain.

The predeclared lead's displayed values are: balanced native {balanced_native:+.3f}%
and +3 mm {balanced_plus3:+.3f}%; native-priority native {priority_native:+.3f}%
and +3 mm {priority_plus3:+.3f}%. These values are descriptions of exposed replay.
They do not select a deployment policy or establish an external generalization
claim. Native outcomes are never averaged into injected conditions.

`all_arm_condition_summary.csv` retains every supplied arm and condition,
including natural-threshold, random-control, and oracle rows when present.
It separately reports the mean of per-case percentage gains because that
quantity differs from the ratio of equally weighted mean MSEs used in the figure.
No raw reference geometry, learned score, or decision mask is read by this script.
The locked threshold file is read solely to verify infeasible/KEEP status.

Source: `{METRICS}`

Source SHA-256: `{source_hash}`

Figure style follows the scientific-figure-making skill: sans-serif editable
vector text, a restrained blue/red/neutral palette, minimal spines, explicit
zero references, and PNG at 300 dpi plus PDF/SVG exports.
"""


def self_test() -> None:
    # Deliberately heterogeneous identity errors distinguish aggregation formulas.
    arms = list(MAIN_ARMS) + [name + "__native_priority" for name in METHODS]
    records = []
    for arm in arms:
        for scene in SCENES:
            for roi in range(4):
                for condition in CONDITIONS:
                    ref = float(roi + 1)
                    actual = ref if arm == "identity" else ref + (0.2 if roi == 0 else -0.1)
                    records.append(dict(arm=arm, scene=scene, roi=str(roi), condition=condition, source_MSE_mm2=actual))
    summary, rows = aggregate(records)
    target = summary[("normalized_gain__balanced", "native")]
    np.testing.assert_allclose(target["relative_gain_pct"], 1.0)
    np.testing.assert_allclose(target["roi_win_fraction"], 0.75)
    assert target["roi_win_count"] == 9 and target["roi_loss_count"] == 3
    assert target["scene_win_count"] == 3
    assert summary[("identity", "native")]["roi_tie_count"] == 12
    assert len(rows) == len(arms) * 5
    assert not np.isclose(target["mean_case_relative_gain_pct"], target["relative_gain_pct"])
    for invalid in (records[:-1], records + [records[0]]):
        try:
            aggregate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed evaluator records were accepted")
    print("SELF-TEST PASSED: aggregation formulas, win counts, complete panels, missing/duplicate rejection")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="Check only synthetic aggregate rows; write nothing")
    parser.add_argument("--replace", action="store_true", help="Replace this script's existing figure artifacts")
    args = parser.parse_args()
    if socket.gethostname() != "liekkas":
        raise RuntimeError("The authorized host is liekkas")
    if args.self_test:
        self_test()
        return
    if not METRICS.is_file():
        raise SystemExit(f"Waiting for the completed evaluator CSV: {METRICS}")
    source_hash = sha256(METRICS)
    records = load_records(METRICS)
    summary, rows = aggregate(records)
    locked_thresholds = json.loads(THRESHOLDS.read_text())
    for method in METHODS:
        status = locked_thresholds[method]["native_priority"]
        if status["feasible"] is not False or status["action"] != "keep":
            raise ValueError("Update the explicit infeasible/KEEP figure annotation to match locked status")
        for condition in CONDITIONS:
            entry = summary[(method + "__native_priority", condition)]
            if entry["mean_MSE_mm2"] != entry["identity_mean_MSE_mm2"] or entry["roi_tie_count"] != 12:
                raise ValueError("A locked infeasible/KEEP arm is not identity in evaluator results")
    names = ("replay_report.png", "replay_report.pdf", "replay_report.svg",
             "all_arm_condition_summary.csv", "figcaption.md", "manifest.json")
    if not args.replace and any((OUT / name).exists() for name in names):
        raise FileExistsError("Figure artifacts already exist; use --replace for a deliberate rendering revision")
    OUT.mkdir(exist_ok=True)
    figure = make_report(summary)
    files = finalize_figure(figure, OUT / "replay_report")
    with (OUT / "all_arm_condition_summary.csv").open("w", newline="") as handle:
        columns = list(rows[0]) + sorted(set().union(*(set(row) for row in rows)) - set(rows[0]))
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "figcaption.md").write_text(caption(summary, source_hash))
    files += [OUT / "all_arm_condition_summary.csv", OUT / "figcaption.md"]
    if sha256(METRICS) != source_hash:
        raise RuntimeError("Evaluation CSV changed during rendering; figure is not a final snapshot")
    source_paths = [METRICS, THRESHOLDS, Path(__file__).resolve(), ROOT / "PROTOCOL.md"]
    for optional in (ROOT / "evaluation" / "SEALED.json", ROOT / "training" / "MODEL_LOCK.json"):
        if optional.is_file():
            source_paths.append(optional)
    manifest = {
        "data_role": "EXPOSED_REPLAY_NOT_INDEPENDENT_CONFIRMATION", "host": socket.gethostname(),
        "sources_sha256": {str(path): sha256(path) for path in source_paths},
        "outputs_sha256": {str(path.relative_to(OUT)): sha256(path) for path in files},
        "source_rows": len(records), "source_arms": sorted({row["arm"] for row in records}),
        "main_arm_order": list(MAIN_ARMS), "scatter_methods": list(FOCUS_METHODS) + ["frozen_gain"],
        "scatter_settings": ["balanced", "native_priority"], "conditions": list(CONDITIONS),
        "aggregation": "MSE: ROIs equal within scene, then scenes equal; percent: ratio of these mean MSEs",
        "wins": "strict per-case MSE < identity; ties are not wins", "raw_GT_access": False,
        "threshold_changes": False, "per_condition_selection": False, "png_dpi": 300,
        "all_native_priority_development_infeasible_and_keep": True,
        "matplotlib_version": matplotlib.__version__, "numpy_version": np.__version__,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"outputs": [str(path) for path in files] + [str(OUT / "manifest.json")],
                      "source_rows": len(records), "source_sha256": source_hash}, indent=2))


if __name__ == "__main__":
    main()
