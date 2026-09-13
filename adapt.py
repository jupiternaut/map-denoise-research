"""Shared world↔local millimetre adapter. The local third axis is the common patch normal."""
from __future__ import annotations

import numpy as np

from transforms import estimate_patch_frame, local_mm_to_world, world_to_local_mm


def build_adapter(xyz_world_m) -> dict:
    frame = estimate_patch_frame(xyz_world_m)
    return {
        "origin_m": frame["origin_m"],
        "basis": frame["basis"],
        "normal_world": frame["normal_world"],
        "source": frame["source"],
        "units": "operator input is millimetres; third column is the shared normal axis",
    }


def to_operator_mm(xyz_world_m, adapter) -> np.ndarray:
    return world_to_local_mm(xyz_world_m, adapter["origin_m"], adapter["basis"])


def from_operator_mm(xyz_local_mm, adapter) -> np.ndarray:
    return local_mm_to_world(xyz_local_mm, adapter["origin_m"], adapter["basis"])


def adapter_payload(adapter) -> dict:
    return {
        "origin_m": np.asarray(adapter["origin_m"]).tolist(),
        "basis": np.asarray(adapter["basis"]).tolist(),
        "normal_world": np.asarray(adapter["normal_world"]).tolist(),
        "source": adapter["source"],
        "units": adapter["units"],
    }
