"""Frozen proposal policies and same-information pose-profiled baseline."""
from fractions import Fraction as Q
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source_snapshot"))
from continuous_world.interval_model import Model
from continuous_world.interval_solver import encode, decode, decide, rational

DEPTH_GRID = tuple(Q(540) + Q(i, 2) for i in range(241))
POSE_VALUES = (Q(-1), Q(-1, 2), Q(-1, 5), Q(-1, 10), Q(0),
               Q(1, 10), Q(1, 5), Q(1, 2), Q(1))


def model_for(kind, radius):
    if kind in ("lovelace_triangle", "rectangle"):
        return Model(kind, Q(radius))
    from layered_model import LayeredModel
    return LayeredModel(kind, Q(radius))


def certificate_projection(outer, incumbent):
    """Maximize worst squared-error improvement over the outer hull."""
    a = rational(incumbent)
    if outer["hull"] is None:
        return {"status": "INCOMPATIBLE", "output": encode(a), "gain_lower": None,
                "reason": "empty_certified_outer_set"}
    left, right = map(decode, outer["hull"])
    b = min(max(a, left), right)
    return decide(outer, a, proposal=b)


class ProfiledResidual:
    """Exact render forecasts; float64 finite depth/pose residual search.

    Left and right pose minimizations separate under l-infinity loss, so the
    five-by-five joint pose grid is searched without constructing 25 images.
    This is a finite grid baseline, never a continuous global optimum claim.
    """

    def __init__(self, kind, depth_grid=DEPTH_GRID):
        self.kind = kind
        self.depth_grid = tuple(depth_grid)
        self.model = model_for(kind, Q(1))
        started = time.perf_counter()
        self.forecasts = np.stack([self.point_forecasts(z) for z in self.depth_grid])
        self.compile_seconds = time.perf_counter() - started

    @lru_cache(maxsize=512)
    def point_forecasts(self, depth):
        values = []
        for c in POSE_VALUES:
            exact = self.model.point_prediction(Q(depth), (c, Q(0), c))
            values.append(np.asarray([float(x) for x in exact]).reshape(3, -1))
        return np.stack(values)

    @staticmethod
    def allowed_indices(radius):
        radius = Q(radius)
        allowed = {radius * Q(j, 2) for j in (-2, -1, 0, 1, 2)}
        return [i for i, c in enumerate(POSE_VALUES) if c in allowed]

    def losses(self, observation, radius, forecasts=None):
        predicted = self.forecasts if forecasts is None else forecasts
        if predicted.ndim == 3:
            predicted = predicted[None, ...]
        obs = np.asarray([float(x) for x in observation]).reshape(3, -1)
        per_camera = np.max(np.abs(predicted - obs), axis=-1)
        allowed = self.allowed_indices(radius)
        best = np.min(per_camera[:, allowed, :], axis=1)
        return np.max(best, axis=1)

    def grid_losses(self, observation, radius):
        return self.losses(observation, radius)

    def select(self, observation, radius, incumbent, grid_losses):
        a = Q(incumbent)
        incumbent_loss = float(self.losses(observation, radius, self.point_forecasts(a))[0])
        best = min(float(np.min(grid_losses)), incumbent_loss)
        if incumbent_loss <= best + 1e-12:
            b = a
        else:
            tied = [z for z, loss in zip(self.depth_grid, grid_losses) if float(loss) <= best + 1e-12]
            b = min(tied, key=lambda z: (abs(z-a), z))
        return {"status": "KEEP" if b == a else "MOVE", "output": encode(b),
                "gain_lower": None, "reason": "finite_pose_profiled_residual",
                "incumbent_residual": incumbent_loss, "minimum_residual": best}
