#!/usr/bin/env python3
"""Postprocess sealed results only; never import or rerun experiment methods."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

_cache = tempfile.TemporaryDirectory(prefix="decision-figures-mpl-")
os.environ["MPLCONFIGDIR"] = _cache.name
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
SOURCES = {}
GENERATED = []
COLORS = ["#737373", "#E69F00", "#0072B2", "#009E73", "#CC79A7", "#D55E00"]
HATCHES = ["", "//", "", "..", "xx", "\\\\"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "svg.fonttype": "none", "svg.hashsalt": "decision-interface-20261008",
                     "savefig.facecolor": "white"})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    path = Path(path)
    SOURCES[str(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
    return path


def read(path):
    return json.loads(record(path).read_text())


def sealed(stage, filename):
    directory = ROOT / stage / "evaluation"
    seal = read(directory / "SEAL.json")
    path = directory / filename
    assert sha(path) == seal["files"][str(path)], str(path)
    return read(path)


def save(fig, name):
    for suffix in ("png", "svg"):
        path = OUT / (name + "." + suffix)
        if path.exists():
            raise FileExistsError("Refusing to overwrite figure: " + str(path))
        metadata = {"Creator": "analysis/make_figures.py", "Date": None} if suffix == "svg" else None
        fig.savefig(path, dpi=300, bbox_inches="tight", metadata=metadata)
        GENERATED.append({"path": str(path), "sha256": sha(path), "bytes": path.stat().st_size})
    plt.close(fig)


def endpoints(summary, arms, labels, colors, title, subtitle, name):
    index = {(r["arm"], r["initial_kind"]): r for r in summary["overall"]}
    rows = [index[(arm, kind)] for arm in arms for kind in ("minus", "correct", "plus")]
    n = rows[0]["n"]
    assert all(row["n"] == n for row in rows)
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.2), sharey=True)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.76, bottom=0.17, wspace=0.23)
    fig.suptitle(title, x=0.13, y=0.96, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.13, 0.885, subtitle, fontsize=10, color="#444444")
    specs = [("minus", "mae", "Initial offset: -60 mm", "Depth MAE (mm)"),
             ("correct", "mae", "Initially correct", "Depth MAE (mm)"),
             ("plus", "mae", "Initial offset: +60 mm", "Depth MAE (mm)"),
             ("correct", "harmed", "Initially correct: damage", f"Strict damage count / {n}")]
    y = np.arange(len(arms))
    for ax, (kind, metric, heading, xlabel) in zip(axes, specs):
        values = [index[(arm, kind)][metric] for arm in arms]
        bars = ax.barh(y, values, height=0.66, color=colors, edgecolor="#333333", linewidth=0.5)
        for j, bar in enumerate(bars):
            bar.set_hatch(HATCHES[j % len(HATCHES)])
        maximum = max(max(values), 1.0)
        ax.set_xlim(0, maximum * 1.31)
        for yi, value in zip(y, values):
            text = str(int(value)) if metric == "harmed" else ("0" if value == 0 else f"{value:.3f}" if value < 0.1 else f"{value:.2f}")
            ax.text(value + maximum * 0.035, yi, text, va="center", fontsize=9.5)
        ax.set_title(heading, pad=12)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, integer=(metric == "harmed")))
        ax.set_yticks(y, labels)
    axes[0].invert_yaxis()
    fig.text(0.13, 0.05, "KEEP/rejections remain in the denominator. Strict damage: error increase > 1e-9 mm.\nDescriptive point estimates; no confidence intervals are shown. Each panel has its own x-axis scale.",
             fontsize=9, color="#444444")
    save(fig, name)
    return rows


def witness(rows):
    selected = {r["arm"]: r for r in rows if r["id"] == "w002" and r["initial_kind"] == "correct"}
    p, m = selected["ED_S0_P"], selected["ED_S1_M"]
    curve_path = Path(p["curve_file"])
    old_seal = read(curve_path.parent / "SEAL.json")
    assert sha(curve_path) == old_seal["files"][str(curve_path)]
    record(curve_path)
    with np.load(curve_path, allow_pickle=False) as artifact:
        grid, loss = artifact["grid"], artifact["loss"]
    assert p["intervals"] == [[486.0, 782.0]] and p["support_mean"] == 634.0
    assert p["selected_depth"] == 660.0 and m["selected_depth"] == 600.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=0.085, right=0.98, top=0.77, bottom=0.19, wspace=0.30)
    fig.suptitle("w002: a low residual does not survive uniform-interval averaging", x=0.085,
                 y=0.96, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.085, 0.885, "One already-exposed E1 world | initially correct at 600 mm | same sealed ED score curve", fontsize=10)
    ax = axes[0]
    ax.axvspan(486, 782, color="#E69F00", alpha=0.15, label="P support [486, 782]")
    ax.plot(grid, loss, color="#333333", lw=1.9, label="Sealed raw pixel MSE")
    for depth, color, style, label in ((600, "#0072B2", "-", "Candidate minimum / M: 600"),
                                      (634, "#737373", "--", "P interval mean: 634"),
                                      (660, "#D55E00", ":", "P selected action: 660")):
        ax.axvline(depth, color=color, ls=style, lw=1.6, label=label)
    candidates = np.asarray(p["candidates"])
    candidate_losses = np.asarray([loss[np.flatnonzero(grid == c)[0]] for c in candidates])
    ax.scatter(candidates, candidate_losses, facecolor="white", edgecolor="#333333", zorder=5, s=34)
    ax.set(xlim=(450, 900), ylim=(-8, 255), xlabel="Candidate depth (mm)", ylabel="Raw pixel MSE (gray level squared)")
    ax.legend(loc="upper center", fontsize=8.1, frameon=False)
    ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
    ax = axes[1]
    ax.hlines(1.9, 486, 782, lw=9, color="#E69F00", alpha=0.5)
    ax.text(634, 2.08, "P support interval", ha="center", fontsize=10)
    ax.axvline(600, color="#0072B2", lw=1, alpha=0.5)
    ax.scatter([600], [1], s=65, facecolor="white", edgecolor="#333333", zorder=4)
    ax.annotate("", xy=(660, 1), xytext=(600, 1), arrowprops={"arrowstyle": "->", "color": "#D55E00", "lw": 2})
    ax.scatter([660], [1], s=65, color="#D55E00", marker="X", zorder=5)
    ax.text(700, 1, "660 mm\n+60 mm error", va="center", color="#9B4000", fontsize=10)
    ax.scatter([600], [0], s=65, color="#0072B2", marker="o", zorder=5)
    ax.text(650, 0, "600 mm\nKEEP; zero error", va="center", color="#005580", fontsize=10)
    ax.set(xlim=(450, 900), ylim=(-0.5, 2.4), xlabel="Selected depth (mm)", yticks=[0, 1], yticklabels=["M", "P"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.085, 0.055, "The labelled minimum is among the five frozen action candidates. P averages interval length; M retains candidate score ordering.\nThis example diagnoses an interface failure; it is not new confirmation evidence.", fontsize=9, color="#444444")
    save(fig, "02_w002_score_to_action")
    return {"id": "w002", "old_P": p, "new_M": m, "candidate_depths_mm": candidates.tolist(),
            "candidate_losses": candidate_losses.tolist()}


def main():
    assert matplotlib.__version__.startswith("3.10.7"), matplotlib.__version__
    assert np.__version__ == "2.3.5", np.__version__
    OUT.mkdir(exist_ok=True)
    provenance_path = OUT / "PROVENANCE.json"
    if provenance_path.exists():
        raise FileExistsError(str(provenance_path))
    b0 = read(ROOT / "B0/RESULTS.json")
    assert b0["passed"] and b0["tests"] == 46
    replay = sealed("replay", "SUMMARY.json")
    replay_rows = sealed("replay", "ROWS.json")
    confirmation = sealed("confirmation", "SUMMARY.json")
    confirmation_gate = sealed("confirmation", "GATE.json")
    record(Path(__file__))
    plotted = {}
    plotted["01_replay_ablation"] = endpoints(
        replay, ["ED_S0_P", "ED_S1_P", "ED_S1_M", "ED_S1_U", "ED_S1_R"],
        ["S0 + P", "S1 + P", "S1 + M", "S1 + U", "S1 + R"], COLORS[:5],
        "Old replay: scale fallback and decision rule have separate effects",
        "Frozen ED curves; n = 36 worlds per initial state (30 distinct image tensors; 24 base groups).",
        "01_replay_ablation")
    plotted["02_w002_score_to_action"] = witness(replay_rows)
    plotted["03_confirmation_endpoints"] = endpoints(
        confirmation, ["ED_S1_M", "EF_S1_M", "ED_S1_R", "N_M", "N_P", "KEEP"],
        ["ED + M", "EF + M", "ED + R", "full9 + M", "full9 + P", "KEEP"],
        ["#0072B2", "#009E73", "#CC79A7", "#D55E00", "#E69F00", "#737373"],
        "Confirmation: similar offset error does not erase a preservation failure",
        "n = 48 independent scenes per initial state; 3 states per scene. ED + M: 1 strict damage / 48; registered gate failed.",
        "03_confirmation_endpoints")
    for path, expected in SOURCES.items():
        assert sha(path) == expected["sha256"], "Source changed during plotting: " + path
    provenance = {"scope": "Postprocessing of sealed artifacts only; no scoring, calibration, tuning or evaluation rerun.",
                  "python": "/usr/bin/python3", "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                  "sources": SOURCES, "plotted_values": plotted, "outputs": GENERATED,
                  "confirmation_gate": confirmation_gate, "input_hashes_unchanged": True,
                  "skill": "/home/grf/.codex/skills/kdense-matplotlib/SKILL.md"}
    with provenance_path.open("x") as handle:
        json.dump(provenance, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({"figures": GENERATED, "provenance": str(provenance_path), "input_hashes_unchanged": True}, indent=2))


if __name__ == "__main__":
    main()
