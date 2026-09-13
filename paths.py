"""Absolute host paths for the liekkas 2026-09-12 pilot. Do not write large data under /home."""
from __future__ import annotations

from pathlib import Path
import socket

HOST = "liekkas"
PROJECT = Path("/home/grf/Documents/Codex/2026-09-12/map-denoise-dataset-pilot-v1")
DATASETS = Path("/srv/slam-research/grf/map-denoise/datasets/multiscan-pilot-v1")
RUNS = Path("/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1")
OLD_CHECKPOINT = Path("/home/grf/Documents/Codex/2026-09-11/map-denoise-correlation-v1")
OPEN3D_PY = Path("/srv/slam-research/grf/map-denoise/envs/open3d-019/bin/python")
TORCH_PY = Path("/srv/slam-research/grf/map-denoise/envs/pathnet-v5/bin/python")

ETH3D_DOWNLOADS = DATASETS / "eth3d" / "downloads"
ETH3D_EXTRACTED = DATASETS / "eth3d" / "extracted"
UCL_DOWNLOADS = DATASETS / "ucl" / "downloads"
UCL_EXTRACTED = DATASETS / "ucl" / "extracted"
TUM_META = DATASETS / "tum_metadata"
PATCHES = RUNS / "patches"
PREVIEW = RUNS / "preview"
PILOT = RUNS / "pilot"
SYNTHETICS = RUNS / "synthetics"
LOGS = RUNS / "logs"

BUDGET_BYTES = 20 * 1024 ** 3
MIN_FREE_BYTES = 30 * 1024 ** 3


def require_liekkas() -> None:
    name = socket.gethostname()
    if name != HOST:
        raise RuntimeError(f"this task must run on {HOST}, got {name}")
