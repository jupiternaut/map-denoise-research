"""Authorized local paths. Reference geometry is listed but must not be read by constructors."""

from pathlib import Path

WORKSPACE = Path("/home/grf/Documents/Codex/2026-09-15/map-denoise-v25-surface-regeneration")
DATA = Path("/srv/slam-research/grf/map-denoise/datasets")
LEGACY_PYTHON = Path("/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python")
EXPECTED_HOST = "liekkas"

SCENE_INPUTS = {
    24: {
        "images": DATA / "loss-alignment-v23/scan24/image",
        "colmap": DATA / "loss-alignment-v23/scan24/sparse/0",
        "cameras_npz": DATA / "real-closure-v21/cameras_geosvr_linked.npz",
        "mesh": DATA / "published-outputs-v1/scan24_mesh.ply",
    },
    37: {
        "images": DATA / "loss-alignment-v23/scan37/image",
        "colmap": DATA / "loss-alignment-v23/scan37/sparse/0",
        "cameras_npz": DATA / "reconstruction-v22-scan37/cameras.npz",
        "mesh": DATA / "reconstruction-v22-scan37/scan37_mesh.ply",
    },
}

# Evaluation-only. Constructors and selectors must not open these.
EVAL_ONLY = {
    24: {
        "laser": DATA / "published-outputs-v2-reference/stl024_total.ply",
        "mask": DATA / "published-outputs-v2-reference/ObsMask24_10.mat",
        "laser_sha256": "963f2893d40d72957acdeb0affcd8aaa888408b180f5a62b62af747554ba4665",
    },
    37: {
        "laser": DATA / "reconstruction-v22-scan37/stl037_total.ply",
        "mask": DATA / "reconstruction-v22-scan37/ObsMask37_10.mat",
        "laser_sha256": "dcd290f8d6bee24b51fa6df7d0fa3cf017e2c46d9edea0897dc935af2cc56c55",
    },
}
