"""Shared RK stepper. No hidden parameters."""
from __future__ import annotations

DT_MAX = 0.02


def rk4(fun, t, y, dt):
    k1 = fun(t, y)
    k2 = fun(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = fun(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = fun(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
