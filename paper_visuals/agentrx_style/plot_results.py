"""Rebuild Fig. 2 and exact derived data from committed V22 evidence.

Run from any directory: python /path/to/plot_results.py
Dependencies: numpy, matplotlib. No geometry, model, network, or experiment run.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
RUN = ROOT / "evidence/progress_v22/runs/reconstruction-v22-qayc8gft"
DATA = OUT / "data"
METHODS = ["identity", "v18", "apss2", "rimls2", "local_plane64", "quadratic64",
           "multiscale_full", "multiscale_consensus", "multiscale_matched"]
LABELS = ["Identity", "Frozen V18", "APSS2", "RIMLS2", "Local plane 64", "Quadratic 64",
          "Multiscale full", "Consensus (primary)", "Matched damping"]
PRIMARY, MATCHED = METHODS[-2:]
BLUE, ORANGE, GREY, INK, GUIDE = "#3778A8", "#CB7736", "#737373", "#292929", "#D9D9D9"
METRICS = ["accuracy_mm", "completeness_mm", "recall", "fscore", "displacement_rms_mm"]


def write_csv(name: str, rows: list[dict]):
    with (DATA / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_and_validate():
    raw = (RUN / "confirmation_RESULTS.json").read_bytes()
    rows = json.loads(raw)
    stored = json.loads((RUN / "SUMMARY.json").read_text(encoding="utf-8"))["results"]["confirmation"]
    audit = json.loads((RUN / "FINAL_AUDIT.json").read_text(encoding="utf-8"))
    assert len(rows) == 216
    assert all(r["status"] == "OK" and r["phase"] == "confirmation" for r in rows)
    cases = sorted({r["case"] for r in rows})
    assert len(cases) == 24 and cases == [f"s37_confirm{i:02d}" for i in range(24)]
    keyed = {(r["case"], r["method"]): r for r in rows}
    assert len(keyed) == 216 and set(r["method"] for r in rows) == set(METHODS)
    assert all((c, m) in keyed for c in cases for m in METHODS)
    assert all(np.isfinite(r[k]) for r in rows for k in METRICS)
    assert all(keyed[c, m][k] == keyed[c, "identity"][k]
               for c in cases for m in METHODS for k in ("n_input", "n_reference"))
    means, comparison, table = {}, {}, []
    max_diff = 0.
    for method, label in zip(METHODS, LABELS):
        rr = [keyed[c, method] for c in cases]
        means[method] = {k: float(np.mean([r[k] for r in rr])) for k in METRICS}
        assert stored["summary"][method]["n"] == stored["summary"][method]["valid"] == 24
        for key in METRICS:
            diff = abs(means[method][key] - stored["summary"][method][key])
            assert diff < 5e-13, (method, key, diff)
            max_diff = max(max_diff, diff)
        table.append(dict(method=method, label=label, n_patches=24,
                          mae_mm=means[method]["accuracy_mm"],
                          completeness_mm=means[method]["completeness_mm"],
                          recall_pct=means[method]["recall"] * 100,
                          fscore=means[method]["fscore"],
                          displacement_rms_mm=means[method]["displacement_rms_mm"]))
    for baseline in ["identity", "v18", "apss2", "rimls2", MATCHED]:
        delta = [keyed[c, PRIMARY]["accuracy_mm"] - keyed[c, baseline]["accuracy_mm"] for c in cases]
        comparison[baseline] = dict(paired=24, wins=int(np.sum(np.array(delta) < 0)),
            mean_mae_difference_mm=float(np.mean(delta)),
            relative_gain=1 - means[PRIMARY]["accuracy_mm"] / means[baseline]["accuracy_mm"],
            recall_difference_pp=(means[PRIMARY]["recall"] - means[baseline]["recall"]) * 100)
        saved = stored["comparisons"][baseline]
        assert abs(comparison[baseline]["relative_gain"] - saved["relative_gain"]) < 5e-13
        assert abs(comparison[baseline]["wins"] / 24 - saved["win_fraction"]) < 5e-13
    for c in cases:
        assert abs(keyed[c, PRIMARY]["displacement_rms_mm"] - keyed[c, MATCHED]["displacement_rms_mm"]) < 1e-12
    paired = []
    for case in cases:
        base = keyed[case, "identity"]
        for method in [PRIMARY, MATCHED]:
            r = keyed[case, method]
            paired.append(dict(case=case, method=method, n_scored_input=r["n_input"],
                n_local_reference=r["n_reference"], mae_mm=r["accuracy_mm"], recall_pct=r["recall"]*100,
                delta_mae_mm=r["accuracy_mm"]-base["accuracy_mm"],
                delta_recall_pp=(r["recall"]-base["recall"])*100))
    DATA.mkdir(exist_ok=True)
    write_csv("table_summary.csv", table)
    write_csv("paired_deltas.csv", paired)
    write_csv("confirmation_rows.csv", rows)
    record = dict(source=str((RUN / "confirmation_RESULTS.json").relative_to(ROOT)),
        source_sha256=hashlib.sha256(raw).hexdigest(), input_rows=216, patches=24, methods=9,
        selected_source_points=24576, scored_input_points=sum(keyed[c, "identity"]["n_input"] for c in cases),
        scored_input_range=[min(keyed[c, "identity"]["n_input"] for c in cases), max(keyed[c, "identity"]["n_input"] for c in cases)],
        local_reference_slots=sum(keyed[c, "identity"]["n_reference"] for c in cases),
        local_reference_slots_note="Sum of local patch reference counts; uniqueness across local reference sets was not rechecked",
        selection="All 24 confirmation patches in fixed numeric case order; all nine frozen methods; equal patch weights",
        max_difference_from_saved_summary=max_diff, comparisons_primary_vs_baseline=comparison,
        saved_final_audit=dict(independently_scored_outputs=audit["independently_scored_outputs"],
                              max_metric_difference=audit["max_metric_difference"],
                              historical_verifier_status=audit["original_verifier_status"]),
        validation_scope="Stored row aggregation and figure provenance, not rerunning the experiment or independent geometry rescoring",
        matplotlib_version=matplotlib.__version__)
    (DATA / "validation.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return keyed, cases, means, comparison


def setup_font():
    path = font_manager.findfont("Times New Roman", fallback_to_default=False)
    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8.5,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GREY,
        "xtick.color": INK, "ytick.color": INK, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 7.5,
        "axes.linewidth": .55, "xtick.major.width": .5, "ytick.major.width": .5,
        "xtick.major.size": 2.5, "ytick.major.size": 0,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.facecolor": "white", "figure.facecolor": "white"})
    return path


def finish_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def build_figure(keyed, cases, means, comparisons):
    fig = plt.figure(figsize=(170/25.4, 178/25.4))
    grid = fig.add_gridspec(2, 2, left=.085, right=.977, bottom=.09, top=.85,
                           height_ratios=[1.57, 1], hspace=.56, wspace=.32)
    axa, axb, axc, axd = [fig.add_subplot(grid[i, j]) for i in range(2) for j in range(2)]
    fig.text(.085, .972, "Scan37 confirmation: local geometry and coverage", size=11, weight="bold", va="top")
    fig.text(.085, .942, "24 patches  ·  9 frozen methods  ·  equal patch weights  ·  recall threshold 1 mm", size=8.2, va="top")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=4, lw=0, label="Consensus (primary)"),
               Line2D([], [], color=ORANGE, marker="s", markerfacecolor="white", ms=4, lw=0, label="Matched damping"),
               Line2D([], [], color=GREY, marker="o", markerfacecolor="white", ms=4, lw=0, label="Other methods")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.074, .913), ncol=3,
               frameon=False, handletextpad=.35, columnspacing=1.5, fontsize=8)
    yy = np.arange(24)
    for ax, metric, scale, title, label in [
        (axa, "accuracy_mm", 1, "(a) Per-patch MAE change", "MAE minus identity (mm); lower is better"),
        (axb, "recall", 100, "(b) Per-patch recall change", "Recall minus identity (pp); higher is better")]:
        p = np.array([(keyed[c, PRIMARY][metric] - keyed[c, "identity"][metric])*scale for c in cases])
        m = np.array([(keyed[c, MATCHED][metric] - keyed[c, "identity"][metric])*scale for c in cases])
        for y, pv, mv in zip(yy, p, m):
            ax.plot([0, pv], [y, y], color=GUIDE, lw=.55, zorder=1)
            ax.plot([mv, pv], [y+.13, y-.13], color=GREY, lw=.65, zorder=2)
        ax.scatter(m, yy+.13, s=11, marker="s", facecolors="white", edgecolors=ORANGE, lw=.7, zorder=3)
        ax.scatter(p, yy-.13, s=10, marker="o", facecolors=BLUE, edgecolors=BLUE, lw=.4, zorder=4)
        ax.axvline(0, color=INK, lw=.75, zorder=0)
        ax.set_yticks(yy, [f"{i:02}" for i in yy])
        ax.set_ylim(23.7, -.7)
        ax.set_title(title, loc="left", pad=8, fontweight="bold")
        ax.set_xlabel(label, labelpad=5)
        ax.set_ylabel("Patch ID", labelpad=4)
        ax.grid(axis="x", color="#EEEEEE", lw=.4)
        finish_axes(ax)
    axa.set_xlim(-.0168, .0064)
    axa.set_xticks([-.015, -.010, -.005, 0, .005], ["−0.015", "−0.010", "−0.005", "0", "+0.005"])
    axb.set_xlim(-3, 1)
    axb.set_xticks([-3, -2, -1, 0, 1], ["−3", "−2", "−1", "0", "+1"])
    primary_comparison = comparisons["identity"]
    recall_wins = sum(keyed[c, PRIMARY]["recall"] > keyed[c, "identity"]["recall"] for c in cases)
    for ax, text in [
        (axa, f"Primary mean: {primary_comparison['mean_mae_difference_mm']:+.6f} mm  |  {primary_comparison['wins']}/24 improved"),
        (axb, f"Primary mean: {primary_comparison['recall_difference_pp']:+.3f} pp  |  {recall_wins}/24 improved")]:
        text = text.replace("-", "−")
        ax.text(0, -.195, text, transform=ax.transAxes, fontsize=7.8, color=INK)
    # All observations must remain in view, including the strongest matched-damping change.
    for ax, metric, scale in [(axa, "accuracy_mm", 1), (axb, "recall", 100)]:
        lo, hi = ax.get_xlim()
        assert all(lo < (keyed[c, m][metric]-keyed[c, "identity"][metric])*scale < hi
                   for c in cases for m in (PRIMARY, MATCHED))
    # Full view uses the original mean metrics for every frozen method.
    for method in METHODS:
        x, y = means[method]["accuracy_mm"], means[method]["recall"]*100
        color = BLUE if method == PRIMARY else ORANGE if method == MATCHED else GREY
        marker = "s" if method == MATCHED else "o"
        face = BLUE if method == PRIMARY else "white"
        for ax in (axc, axd):
            if ax is axd and method == "v18":
                continue
            ax.scatter([x], [y], s=19 if method in (PRIMARY, MATCHED) else 14,
                       marker=marker, facecolor=face, edgecolor=color, linewidth=.85, zorder=4)
    axc.set_title("(c) Method means: full view", loc="left", fontweight="bold", pad=8)
    axd.set_title("(d) Method means: detail of box", loc="left", fontweight="bold", pad=8)
    axc.set_xlim(.4295, .465)
    axc.set_ylim(87.2, 93)
    axc.set_xticks([.43, .44, .45, .46])
    axc.set_yticks([88, 90, 92])
    axc.add_patch(Rectangle((.4308, 91.22), .0074, 1.4, facecolor="none", edgecolor=GREY, lw=.75, ls="--"))
    axc.annotate("8 methods\n(detail at right)", xy=(.4382, 91.9), xytext=(.445, 92.2),
                 fontsize=7.6, ha="left", va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=.6))
    axc.annotate("Frozen V18", xy=(means["v18"]["accuracy_mm"], means["v18"]["recall"]*100),
                 xytext=(-7, 10), textcoords="offset points", ha="right", fontsize=8)
    axd.set_xlim(.4308, .4382)
    axd.set_ylim(91.22, 92.62)
    axd.set_xticks([.432, .434, .436, .438])
    axd.set_yticks([91.4, 91.8, 92.2, 92.6])
    # Direct labels with explicit offsets; data coordinates are never altered.
    offsets = {
        "identity": ("Identity", -.00015, .155, "right"),
        "apss2": ("APSS2", -.00025, .195, "right"),
        "rimls2": ("RIMLS2", .0003, -.225, "right"),
        "local_plane64": ("Local plane 64", .00017, .105, "left"),
        "quadratic64": ("Quadratic 64", -.00055, .29, "right"),
        "multiscale_full": ("Multiscale full", -.00018, -.26, "right"),
        PRIMARY: ("Consensus", .0006, -.25, "left"),
        MATCHED: ("Matched", -.00014, -.42, "center"),
    }
    for method, (label, dx, dy, align) in offsets.items():
        x, y = means[method]["accuracy_mm"], means[method]["recall"]*100
        color = BLUE if method == PRIMARY else ORANGE if method == MATCHED else INK
        axd.annotate(label, (x, y), (x+dx, y+dy), fontsize=7.4, color=color,
                     ha=align, va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=.5), zorder=5)
    for ax in (axc, axd):
        ax.set_xlabel("Mean MAE (mm); lower is better", labelpad=5)
        ax.set_ylabel("Mean recall (%)", labelpad=4)
        ax.grid(color="#EEEEEE", lw=.4)
        finish_axes(ax)
    fig.text(.085, .013, "Local patch metrics; this is not a full-scene DTU score.  Grey rows connect paired observations, not time steps.", fontsize=7.5, va="bottom")
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"fig02_confirmation.{suffix}", dpi=350)
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    font = setup_font()
    keyed, cases, means, comparisons = load_and_validate()
    build_figure(keyed, cases, means, comparisons)
    print(json.dumps({"font": font, "outputs": ["fig02_confirmation.pdf", "fig02_confirmation.svg", "fig02_confirmation.png"],
                      "rows": 216, "methods": 9, "patches": 24,
                      "primary_vs_identity": comparisons["identity"],
                      "primary_vs_matched": comparisons[MATCHED]}, indent=2))


if __name__ == "__main__":
    main()
