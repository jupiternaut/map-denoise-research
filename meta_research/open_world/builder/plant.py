"""Hidden plant. Builder-only. Never import from participant policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from host.sim_core import DT_MAX, rk4 as _rk


@dataclass
class HiddenSpec:
    system_id: str
    mismatch: str  # none | dynamics | observation | memory | representation
    m: float = 1.0
    k: float = 4.0
    c: float = 0.4
    c_nl: float = 0.0
    k_nl: float = 0.0
    memory_gain: float = 0.0
    memory_decay: float = 0.6
    drift_rate: float = 0.0
    sensor_bias0: float = 0.0
    process_noise: float = 0.0
    meas_noise: float = 0.0
    seed: int = 0
    split: str = "construction"
    notes: str = ""


@dataclass
class HiddenPlant:
    spec: HiddenSpec
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.spec.seed)

    def pack(self, x: float, v: float, mem: float = 0.0, bias: Optional[float] = None) -> np.ndarray:
        if bias is None:
            bias = self.spec.sensor_bias0
        return np.array([x, v, mem, bias], dtype=float)

    def deriv(self, t: float, z: np.ndarray, u: float) -> np.ndarray:
        x, v, mem, bias = z
        s = self.spec
        acc = (-s.k * x - s.c * v - s.k_nl * x**3 - s.c_nl * (v**3) - s.memory_gain * mem + u) / s.m
        dmem = -s.memory_decay * mem + u
        dbias = s.drift_rate
        return np.array([v, acc, dmem, dbias], dtype=float)

    def measure(self, z: np.ndarray) -> np.ndarray:
        x, v, _mem, bias = z
        y = np.array([x + bias, v], dtype=float)
        if self.spec.meas_noise > 0:
            y = y + self.rng.normal(0.0, self.spec.meas_noise, size=2)
        return y

    def simulate(
        self,
        z0: np.ndarray,
        t_grid: np.ndarray,
        u_of_t: Callable[[float], float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return times, measurements (T,2), hidden states (T,4)."""
        t_grid = np.asarray(t_grid, dtype=float)
        z = np.array(z0, dtype=float).copy()
        zs = [z.copy()]
        ts = [float(t_grid[0])]
        for i in range(1, len(t_grid)):
            t0 = float(t_grid[i - 1])
            t1 = float(t_grid[i])
            nsub = max(1, int(np.ceil((t1 - t0) / DT_MAX)))
            dt = (t1 - t0) / nsub
            t = t0
            for _ in range(nsub):
                u = float(u_of_t(t))
                z = _rk(lambda tt, zz: self.deriv(tt, zz, u), t, z, dt)
                t += dt
            if self.spec.process_noise > 0:
                z = z + self.rng.normal(0.0, self.spec.process_noise, size=z.shape)
            zs.append(z.copy())
            ts.append(t1)
        Z = np.stack(zs, axis=0)
        Y = np.stack([self.measure(zi) for zi in Z], axis=0)
        return np.array(ts), Y, Z


def make_u(kind: str, amp: float, freq: float, phase: float = 0.0) -> Callable[[float], float]:
    kind = str(kind)
    if kind == "sine":
        w = 2.0 * np.pi * float(freq)
        return lambda t, a=amp, w=w, p=phase: a * np.sin(w * t + p)
    if kind == "step":
        return lambda t, a=amp: a
    if kind == "zero":
        return lambda t: 0.0
    if kind == "pulse":
        dur = 1.0 / max(float(freq), 1e-6)
        return lambda t, a=amp, d=dur: a if 0.0 <= t < d else 0.0
    if kind == "chirp":
        return lambda t, a=amp, f=freq: a * np.sin(2.0 * np.pi * (f * t + 0.15 * t * t))
    raise ValueError(kind)


CONSTRUCTION_SPECS: List[HiddenSpec] = [
    HiddenSpec(
        system_id="C0_linear",
        mismatch="none",
        m=1.0,
        k=4.0,
        c=0.5,
        seed=1,
        notes="no mismatch: linear mass-spring-damper",
    ),
    HiddenSpec(
        system_id="C1_cubic_damp",
        mismatch="dynamics",
        m=1.0,
        k=3.5,
        c=0.15,
        c_nl=0.55,
        seed=2,
        notes="nonlinear cubic damping; linear ODE cannot hold both amplitudes",
    ),
    HiddenSpec(
        system_id="C2_memory",
        mismatch="memory",
        m=1.0,
        k=4.0,
        c=0.25,
        memory_gain=1.4,
        memory_decay=0.35,
        seed=3,
        notes="hidden memory of drive history",
    ),
    HiddenSpec(
        system_id="C3_drift",
        mismatch="observation",
        m=1.0,
        k=4.2,
        c=0.4,
        drift_rate=0.08,
        sensor_bias0=0.0,
        seed=4,
        notes="slow additive sensor bias on position",
    ),
    HiddenSpec(
        system_id="C4_memory_pair_B",
        mismatch="memory",
        m=1.0,
        k=4.0,
        c=0.25,
        memory_gain=1.4,
        memory_decay=0.35,
        seed=3,
        notes="same plant as C2; paired branch via different pre-drive",
    ),
    HiddenSpec(
        system_id="C5_unrepresentable",
        mismatch="representation",
        m=1.0,
        k=5.0,
        c=0.2,
        k_nl=1.8,
        c_nl=0.4,
        memory_gain=0.9,
        memory_decay=0.5,
        drift_rate=0.05,
        seed=6,
        notes="combined mismatch used as stress; exact recovery not required",
    ),
]


def spec_by_id(system_id: str) -> HiddenSpec:
    for s in CONSTRUCTION_SPECS:
        if s.system_id == system_id:
            return s
    raise KeyError(system_id)
