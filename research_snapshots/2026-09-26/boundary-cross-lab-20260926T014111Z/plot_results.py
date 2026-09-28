"""Plot the predeclared candidate × evidence cross from sealed evaluator rows.

No geometry, decision masks, point labels or models are loaded. Recompute every
plotted statistic as ROI -> scene equal-weight means and verify SUMMARY.json.
The evaluator-only oracle panel is explicitly separate from deployable arms.
Run once with the existing open3d-019 interpreter and -B; writes only figures/.
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
BALANCED = ("A_F__balanced", "A_R__balanced", "B_F__balanced", "B_R__balanced")
ARM_LABELS = ("A_F · fitted", "A_R · reserved", "B_F · fitted", "B_R · reserved")
ORACLES = ("oracle_A", "oracle_B", "oracle_AB")
ORACLE_LABELS = ("A / KEEP", "B / KEEP", "A / B / KEEP")
ARMS = ("identity", *BALANCED, *ORACLES)
COLORS = ("#B64342", "#9A4D8E", "#42949E", "#0F4D92")
MARKERS = ("s", "^", "D", "o")
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
    """Independently calculate all values from CSV, accepting only sealed parity."""
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
            baseline = float(np.mean([
                np.mean([r["source_MSE_mm2"] for (a, c, s, _), r in index.items()
                         if (a, c, s) == ("identity", condition, scene)])
                for scene in SCENES]))
            if baseline <= 0:
                raise ValueError("Identity aggregate MSE must be positive")
            item["MSE_gain_percent"] = 100 * (1 - item["source_MSE_mm2"] / baseline)
            for name in (*NUMERIC, "MSE_gain_percent"):
                reference = summary["exposed_replay"][condition][arm][name]
                difference = abs(item[name] - reference)
                if difference > 1e-10:
                    raise ValueError(f"CSV/SUMMARY aggregate mismatch: {arm}/{condition}/{name}")
                checks.append(difference)
            output.append(item)
    data = {(r["arm"], r["condition"]): r for r in output}
    derived = []
    for condition in CONDITIONS:
        gain = {arm: data[(arm, condition)]["MSE_gain_percent"] for arm in ARMS}
        effect_a = gain["A_R__balanced"] - gain["A_F__balanced"]
        effect_b = gain["B_R__balanced"] - gain["B_F__balanced"]
        for oracle in ORACLES:
            if gain[oracle] < -1e-10:
                raise ValueError(f"A KEEP oracle cannot increase MSE: {condition}/{oracle}")
        if gain["oracle_AB"] + 1e-10 < max(gain["oracle_A"], gain["oracle_B"]):
            raise ValueError(f"Joint oracle is not a bound: {condition}")
        item = {"condition": condition,
                        "reserved_minus_fitted_A_gain_pp": effect_a,
                        "reserved_minus_fitted_B_gain_pp": effect_b,
                        "interaction_gain_pp": effect_b - effect_a,
                        "oracle_AB_minus_A_gain_pp": gain["oracle_AB"] - gain["oracle_A"],
                        "oracle_AB_minus_B_gain_pp": gain["oracle_AB"] - gain["oracle_B"]}
        for field, reference_field in (
            ("reserved_minus_fitted_A_gain_pp", "A_R_minus_A_F_gain_pp"),
            ("reserved_minus_fitted_B_gain_pp", "B_R_minus_B_F_gain_pp"),
            ("interaction_gain_pp", "candidate_by_evidence_interaction_gain_pp"),
        ):
            reference = summary["interactions"][condition]["balanced"][reference_field]
            difference = abs(item[field] - reference)
            if difference > 1e-10:
                raise ValueError(f"CSV/SUMMARY interaction mismatch: {condition}/{field}")
            checks.append(difference)
        derived.append(item)
    return output, derived, {"status": "PASS", "numeric_checks": len(checks),
                            "maximum_absolute_difference": max(checks),
                            "aggregation": "four ROI means per scene, then three equally weighted scenes",
                            "plotted_values": "independently recomputed from METRICS.csv"}


def render(rows, derived):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
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
    primary = np.array([[data[(a, c)]["MSE_gain_percent"] for c in CONDITIONS] for a in BALANCED])
    oracle = np.array([[data[(a, c)]["MSE_gain_percent"] for c in CONDITIONS] for a in ORACLES])
    fig = plt.figure(figsize=(16.4, 10.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], width_ratios=[1.1, 1],
                           left=.13, right=.97, top=.865, bottom=.22, hspace=.82, wspace=.41)
    heat, bounds = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])
    trade, interaction = fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])

    def heatmap(ax, matrix, labels, cmap, norm):
        im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        ax.set_xticks(range(5), CONDITION_LABELS)
        ax.set_yticks(range(len(labels)), labels)
        ax.tick_params(length=0, pad=8)
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                value = matrix[r, c]
                rgb = im.cmap(im.norm(value))[:3]
                luminance = .2126 * rgb[0] + .7152 * rgb[1] + .0722 * rgb[2]
                ax.text(c, r, f"{value:+.2f}" if value else "0.00", ha="center", va="center",
                        color="white" if luminance < .53 else "#202020", fontsize=12)
        ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(labels), 1), minor=True)
        ax.grid(which="minor", color="#DDDDDD", lw=.65)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axvline(.5, color="#272727", lw=1.6)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return im

    scale = max(1.0, float(np.max(np.abs(primary))))
    diverging = LinearSegmentedColormap.from_list("harm_gain", ["#B64342", "white", "#3775BA"])
    heatmap(heat, primary, ARM_LABELS, diverging, TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale))
    heat.add_patch(Rectangle((-.5, 2.5), 5, 1, fill=False, edgecolor=COLORS[3], lw=2.5, clip_on=False))
    heat.get_yticklabels()[3].set_color(COLORS[3])
    heat.get_yticklabels()[3].set_fontweight("bold")
    heat.set_title("A  Balanced selection: MSE reduction (%)", loc="left", pad=16)
    heat.text(.5, -.30, "Native protection  |  Four repair conditions shown separately\n"
              "Negative: higher error; positive: lower error versus identity",
              transform=heat.transAxes, ha="center", va="top", fontsize=10, color="#444444")

    sequential = LinearSegmentedColormap.from_list("oracle_potential", ["white", "#AADCA9", "#215F42"])
    heatmap(bounds, oracle, ORACLE_LABELS, sequential, Normalize(vmin=0, vmax=max(1., oracle.max())))
    bounds.set_title("B  Candidate potential: oracle reduction (%)", loc="left", pad=16)
    extra = [r["oracle_AB_minus_A_gain_pp"] for r in derived]
    bounds.text(-.04, -.32, "AB − A\ngain (pp)", transform=bounds.transAxes,
                ha="right", va="top", fontsize=10, color="#215F42")
    for i, value in enumerate(extra):
        bounds.text((i + .5) / 5, -.32, f"+{value:.2f}", transform=bounds.transAxes,
                    ha="center", va="top", fontsize=11, color="#215F42")

    native = [data[(a, "native")] for a in BALANCED]
    xs = np.array([100 * r["moved_fraction"] for r in native])
    ys = np.array([100 * r["harmed_fraction"] for r in native])
    for i in range(4):
        label = f"{BALANCED[i].split('__')[0]}: moved {xs[i]:.2f}%, harm {ys[i]:.2f}%"
        trade.scatter(xs[i], ys[i], c=COLORS[i], marker=MARKERS[i], s=95 if i < 3 else 120,
                      lw=1.5, label=label, zorder=4 + i, edgecolors="white")
    trade.scatter(0, 0, c="#272727", marker="+", s=90, lw=1.5, zorder=3)
    trade.annotate("Identity", (0, 0), xytext=(7, 5), textcoords="offset points", fontsize=9)
    trade.set_xlim(-max(.05, xs.max() * .06), max(.2, xs.max() * 1.1))
    trade.set_ylim(-max(.02, ys.max() * .07), max(.1, ys.max() * 1.13))
    trade.axhline(0, color="#AAAAAA", lw=.7, zorder=1)
    trade.axvline(0, color="#AAAAAA", lw=.7, zorder=1)
    trade.grid(color="#EEEEEE", lw=.6)
    trade.set_title("C  Native edits and geometric harm", loc="left", pad=16)
    trade.set_xlabel("Moved input rows (%)")
    trade.set_ylabel("Native-support rows harmed >0.1 mm (%)")
    trade.legend(loc="upper left", bbox_to_anchor=(-.01, -.24), ncol=1, fontsize=10,
                 handletextpad=.5, labelspacing=.4)

    effects = np.array([r["interaction_gain_pp"] for r in derived])
    xpos = np.arange(5)
    interaction.bar(xpos, effects, width=.62, color=["#3775BA" if x >= 0 else "#B64342" for x in effects],
                    edgecolor="#272727", linewidth=.8)
    span = max(1., np.max(np.abs(effects)))
    for x, effect in zip(xpos, effects):
        interaction.text(x, effect + (.04 * span if effect >= 0 else -.04 * span), f"{effect:+.2f}",
                         ha="center", va="bottom" if effect >= 0 else "top", fontsize=11)
    interaction.set_ylim(min(0, effects.min()) - .21 * span, max(0, effects.max()) + .23 * span)
    interaction.set_xticks(xpos, CONDITION_LABELS)
    interaction.set_ylabel("Difference in MSE gains (percentage points)")
    interaction.axhline(0, color="#777777", lw=.8)
    interaction.set_title("D  Candidate × evidence interaction", loc="left", pad=16)
    interaction.text(.5, -.25, "(B_R − B_F) − (A_R − A_F)\n"
                     "Positive: reserved-source gain is larger for B than for A",
                     transform=interaction.transAxes, ha="center", va="top", fontsize=10, color="#444444")

    fig.suptitle("Candidate geometry × paired verification evidence", x=.54, y=.985,
                 fontsize=21, fontweight="bold")
    fig.text(.54, .934, "Predeclared primary: B_R balanced · 3 exposed replay scenes × 4 ROIs · ROI → scene equal weight",
             ha="center", fontsize=12, color="#444444")
    fig.text(.54, .060,
             "A: single PCA / full patch; B: archived multi-normal / half patch. F: fitted sources; R: correction-reserved sources.\n"
             "Panel B uses evaluator-only choices as fixed-candidate bounds; no oracle is deployed. Shared native support and identity denominator.\n"
             "Descriptive exposed replay, not new confirmation. All five conditions retained; no point-count confidence intervals or per-condition winner selection.",
             ha="center", va="center", fontsize=10, color="#444444", linespacing=1.55)
    outputs = []
    for extension in ("png", "pdf", "svg"):
        path = OUT / f"candidate_evidence_cross.{extension}"
        with path.open("xb") as stream:
            fig.savefig(stream, format=extension, dpi=300, bbox_inches="tight", pad_inches=.18)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main():
    if socket.gethostname() != "liekkas":
        raise RuntimeError("The exact target host is liekkas")
    rows, derived, validation = load_and_validate()
    sources = [ROOT / "plot_results.py", ROOT / "PROTOCOL.md",
               ROOT / "evaluation/METRICS.csv", ROOT / "evaluation/SUMMARY.json",
               ROOT / "evaluation/SEALED.json", ROOT / "training/MODEL_LOCK.json",
               ROOT / "training/THRESHOLDS.json", ROOT / "training/SEALED.json"]
    source_hashes = {str(p): sha256(p) for p in sources}
    OUT.mkdir(exist_ok=False)
    os.environ["MPLCONFIGDIR"] = str(OUT / "mplconfig")
    os.environ["XDG_CACHE_HOME"] = str(OUT / "cache")
    for name, values in (("PLOTTED_VALUES.csv", rows), ("DERIVED_VALUES.csv", derived)):
        with (OUT / name).open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    write_json(OUT / "VALIDATION.json", validation)
    outputs = render(rows, derived)
    for path in sources:
        if sha256(path) != source_hashes[str(path)]:
            raise RuntimeError(f"Source changed while plotting: {path}")
    products = outputs + [OUT / "PLOTTED_VALUES.csv", OUT / "DERIVED_VALUES.csv", OUT / "VALIDATION.json"]
    write_json(OUT / "MANIFEST.json", {
        "source_sha256": source_hashes,
        "output_sha256": {p.name: sha256(p) for p in products},
        "balanced_arms_fixed_before_reading_outcomes": list(BALANCED),
        "primary": "B_R__balanced",
        "oracle_arms_evaluator_only": list(ORACLES),
        "aggregation": validation["aggregation"],
        "data_role": "EXPOSED_REPLAY_NOT_NEW_CONFIRMATION",
        "uncertainty": "None inferred from point count; no confidence intervals shown",
        "native_plot_denominators": {"x": "all input rows", "y": "fixed native support rows"},
        "native_edit_definition": "squared displacement > 1e-14 mm²",
        "native_harm_definition": "reference distance increase > 0.1 mm",
        "native_is_not_assumed_all_valid": True,
        "style_skill": "/home/grf/.codex/skills/scientific-figure-making/SKILL.md",
    })
    print(json.dumps({"figures": [str(p) for p in outputs], "validation": validation,
                      "derived": derived}, indent=2))


if __name__ == "__main__":
    main()
