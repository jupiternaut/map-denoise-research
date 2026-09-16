"""Official DTU observation mask. Evaluation-only."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def load_obs_mask(path: Path | str) -> dict:
    data = loadmat(str(path))
    return {
        "ObsMask": np.asarray(data["ObsMask"]),
        "BB": np.asarray(data["BB"], dtype=np.float64),
        "Res": float(np.asarray(data["Res"]).reshape(-1)[0]),
    }


def observed(points: np.ndarray, obs: dict) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    origin = np.asarray(obs["BB"]).reshape(-1, 3)[:1]
    grid = np.around((points - origin) / obs["Res"]).astype(np.int32)
    shape = np.asarray(obs["ObsMask"].shape)
    valid = np.all((grid >= 0) & (grid < shape), axis=1)
    keep = np.zeros(len(points), dtype=bool)
    if valid.any():
        g = grid[valid]
        keep[valid] = obs["ObsMask"][g[:, 0], g[:, 1], g[:, 2]].astype(bool)
    return keep
