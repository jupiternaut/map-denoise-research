"""Same viewpoint, crop, point size and colour scale on every comparison figure."""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from paths import PATCHES, PILOT, PREVIEW, PROJECT, SYNTHETICS
from schema import read_evaluation, read_patch

VIEW = dict(elev=18, azim=-60)
POINT_SIZE = 8
CMAP = "tab10"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _style(ax, title):
    ax.view_init(**VIEW)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)


def patch_section(json_path: Path, dest: Path):
    plt = _mpl()
    points, meta = read_patch(json_path)
    fig = plt.figure(figsize=(6.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    scans = points["scan_id"]
    for sid in np.unique(scans):
        xyz = points["xyz_world"][scans == sid]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=POINT_SIZE, label=f"scan {int(sid)}")
    _style(ax, f"{json_path.stem} by station")
    ax.legend(fontsize=7)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def synthetic_sections(dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    plt = _mpl()
    picks = [
        SYNTHETICS / "identifiable" / "ghost_s912101_b4.json",
        SYNTHETICS / "identifiable" / "dual_g4_s912101_b4.json",
        SYNTHETICS / "ambiguous" / "ambig_g8_s912101.json",
    ]
    for path in picks:
        if not path.exists():
            continue
        points, meta = read_patch(path)
        ev = read_evaluation(path)
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
        z = points["xyz_world"][:, 2] * 1000.0
        axes[0].scatter(points["scan_id"], z, s=POINT_SIZE, c=points["scan_id"], cmap=CMAP)
        axes[0].set_xlabel("scan_id")
        axes[0].set_ylabel("z / mm")
        axes[0].set_title(path.stem)
        labels = ev.get("gt_layer")
        if labels is not None:
            axes[1].scatter(points["xyz_world"][:, 0] * 1000.0, z, s=POINT_SIZE, c=labels, cmap=CMAP)
        else:
            axes[1].scatter(points["xyz_world"][:, 0] * 1000.0, z, s=POINT_SIZE)
        axes[1].set_xlabel("x / mm")
        axes[1].set_ylabel("z / mm")
        axes[1].set_title("layer labels are evaluation-only")
        fig.tight_layout()
        fig.savefig(dest_dir / f"{path.stem}_section.png", dpi=140)
        plt.close(fig)


def method_compare(case_prefix: str, dest: Path):
    plt = _mpl()
    out = PILOT / "outputs"
    files = sorted(out.glob(f"{case_prefix}__*.npz"))
    if not files:
        return
    n = min(4, len(files))
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.4), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]
    for ax, path in zip(axes, files[:n]):
        with np.load(path) as data:
            xyz = data["xyz_mm"]
        method = path.stem.split("__")[-1]
        ax.scatter(xyz[:, 0], xyz[:, 2], s=POINT_SIZE, c="0.2")
        ax.set_title(method, fontsize=9)
        ax.set_xlabel("u / mm")
        ax.set_ylabel("normal / mm")
        ax.set_aspect("auto")
    fig.suptitle(f"{case_prefix} same crop/point size")
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main():
    raise SystemExit("Legacy plots disabled; use summarize_repair_v2.py with a new V2 run directory")
    fig_dir = PROJECT / "figures"
    fig_dir.mkdir(exist_ok=True)
    synthetic_sections(fig_dir)
    for path in sorted(PATCHES.rglob("*.json")):
        if path.name == "EXTRACT_REPORT.json" or "evaluation" in path.parts:
            continue
        patch_section(path, fig_dir / f"{path.stem}_stations.png")
    method_compare("ghost_s912101_b4", fig_dir / "ghost_s912101_b4_methods.png")
    method_compare("dual_g4_s912101_b4", fig_dir / "dual_g4_s912101_b4_methods.png")
    method_compare("ambig_g8_s912101", fig_dir / "ambig_g8_s912101_methods.png")
    for stem in ("da_wall", "da_junction", "da_thin", "cy_wall", "cy_junction", "cy_thin"):
        method_compare(f"{stem}_unpert", fig_dir / f"{stem}_unpert_methods.png")
    print("figures written", fig_dir)


if __name__ == "__main__":
    main()
