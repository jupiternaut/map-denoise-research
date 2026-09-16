"""Photo overlays and comparison figures from already-written geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .cameras import load_scene, project_matrix
from .cli import ARMS_TO_RUN, EXTRA_ARMS, run_dir
from .io_util import read_json
from .paths import WORKSPACE


def _load_points(path: Path) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(cloud.points, dtype=np.float64)
    return pts if pts.size else np.zeros((0, 3), dtype=np.float64)


def overlay_roi(roi_id: str, dest: Path) -> None:
    row = next(r for r in read_json(run_dir() / "rois.json") if r["roi_id"] == roi_id)
    scene = load_scene(row["scene_id"], load_mesh=False)
    view = scene.view_by_name(row["ref_view"])
    base = scene.load_image(view.name)
    colors = {
        "identity": (255, 255, 255),
        "fusion_wta": (80, 180, 255),
        "v25_atlas": (255, 90, 90),
        "v25_gated_move": (255, 220, 40),
        "v25_depth_cc": (80, 255, 140),
    }
    dest.mkdir(parents=True, exist_ok=True)
    for name, color in colors.items():
        ply = run_dir() / roi_id / f"{name}.ply"
        if not ply.is_file():
            continue
        image = Image.fromarray(base.copy())
        draw = ImageDraw.Draw(image, "RGBA")
        pts = _load_points(ply)
        if len(pts) == 0:
            image.save(dest / f"{roi_id}_{name}.png")
            continue
        if len(pts) > 8000:
            pts = pts[:: max(1, len(pts) // 8000)]
        uv, z = project_matrix(pts, view.P_phys)
        keep = np.isfinite(uv).all(1) & (z > 0)
        uv = uv[keep]
        for u, v in uv:
            x, y = int(u), int(v)
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color + (180,))
        image.save(dest / f"{roi_id}_{name}.png")


def main() -> None:
    out = WORKSPACE / "figures" / "overlays"
    rois = [r["roi_id"] for r in read_json(run_dir() / "rois.json")]
    for roi in rois:
        overlay_roi(roi, out)
        print("overlay", roi, flush=True)


if __name__ == "__main__":
    main()
