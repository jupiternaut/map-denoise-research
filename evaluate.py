"""LEGACY V1 evaluator with known scoring defects; retained only for audit.

Do not use for new results. evaluate_v2.py is the corrected world-space scorer.
"""
from __future__ import annotations

import numpy as np


def _fit_block(info: dict):
    for blob in (info, info.get("backend_info") or {}):
        fit = blob.get("fit")
        if isinstance(fit, dict) and fit:
            first = next(iter(fit.values()))
            if isinstance(first, dict):
                return first
    return {}


def _method_k(info: dict):
    for blob in (info, info.get("backend_info") or {}, _fit_block(info)):
        for key in ("k", "K", "n_layers", "selected_k"):
            if key in blob and blob[key] is not None:
                try:
                    return int(blob[key])
                except (TypeError, ValueError):
                    pass
    return None


def _method_means(info: dict):
    for blob in (info, info.get("backend_info") or {}, _fit_block(info)):
        mu = blob.get("mu_mm")
        if mu is not None:
            values = np.sort(np.asarray(mu, dtype=np.float64).reshape(-1))
            if len(values):
                return values
    return None


def cluster_normal_mm(values, max_k=2):
    """Tiny 1D 1/2-mean fit used only when a method does not report K."""
    z = np.asarray(values, dtype=np.float64)
    best = (np.inf, 1, np.array([z.mean()]))
    for k in (1, 2)[:max_k]:
        if k == 1:
            means = np.array([z.mean()])
            sse = float(np.sum((z - means[0]) ** 2))
        else:
            lo, hi = np.quantile(z, [0.2, 0.8])
            if abs(hi - lo) < 1e-9:
                means = np.array([z.mean(), z.mean()])
            else:
                means = np.array([lo, hi], dtype=np.float64)
            for _ in range(12):
                d = np.abs(z[:, None] - means[None, :])
                lab = d.argmin(axis=1)
                new = np.array([
                    z[lab == 0].mean() if np.any(lab == 0) else means[0],
                    z[lab == 1].mean() if np.any(lab == 1) else means[1],
                ])
                if np.max(np.abs(new - means)) < 1e-9:
                    means = new
                    break
                means = new
            sse = float(np.sum((z - means[lab]) ** 2))
        if sse < best[0] - 1e-9 or (abs(sse - best[0]) <= 1e-9 and k < best[1]):
            best = (sse, k, means)
    return best[1], np.sort(best[2])


def synthetic_geometry(output_mm, info, eval_data, adapter_normal_is_z=True):
    z = np.asarray(output_mm, dtype=np.float64)[:, 2]
    k_method = _method_k(info)
    identifiable = bool(eval_data.get("identifiable", True))
    if not identifiable:
        return {
            "identifiable": False,
            "k_method": k_method,
            "note": "ambiguous case: a single recovered explanation is not scored as unique truth",
            "false_split": None,
            "false_merge": None,
            "gap_err_mm": None,
            "surface_err_mm": None,
            "coverage": 1.0,
            "layer_match_failed": False,
        }
    n_true = int(eval_data["n_true_layers"])
    true_gap = float(eval_data["true_gap_mm"])
    k_hat = k_method if k_method is not None else cluster_normal_mm(z)[0]
    false_split = int(n_true == 1 and k_hat >= 2)
    false_merge = int(n_true >= 2 and k_hat <= 1)
    if n_true == 1:
        surface = float(np.mean(np.abs(z - float(np.median(z)))))
        return {
            "identifiable": True,
            "k_method": k_method,
            "k_hat": k_hat,
            "n_true_layers": n_true,
            "false_split": false_split,
            "false_merge": 0,
            "gap_err_mm": 0.0 if k_hat == 1 else None,
            "surface_err_mm": surface,
            "coverage": 1.0,
            "layer_match_failed": bool(k_hat != 1),
        }
    means = _method_means(info)
    if means is None or len(means) < 2:
        means = np.sort(cluster_normal_mm(z, max_k=2)[1])
    if len(means) < 2 or k_hat < 2:
        return {
            "identifiable": True,
            "k_method": k_method,
            "k_hat": k_hat,
            "n_true_layers": n_true,
            "false_split": 0,
            "false_merge": 1,
            "gap_err_mm": None,
            "surface_err_mm": None,
            "coverage": 1.0,
            "layer_match_failed": True,
        }
    gap = float(means[1] - means[0])
    # Match unordered means to {0, gap}.
    candidates = [np.array([0.0, true_gap]), np.array([true_gap, 0.0])]
    err = min(float(np.mean(np.abs(means - cand))) for cand in candidates)
    return {
        "identifiable": True,
        "k_method": k_method,
        "k_hat": k_hat,
        "n_true_layers": n_true,
        "false_split": 0,
        "false_merge": false_merge,
        "gap_err_mm": abs(gap - true_gap),
        "surface_err_mm": err,
        "coverage": 1.0,
        "layer_match_failed": False,
    }


def point_errors(output_world, reference_world):
    delta = np.asarray(output_world) - np.asarray(reference_world)
    rms = float(np.sqrt(np.mean(np.sum(delta ** 2, axis=1))))
    mae = float(np.mean(np.linalg.norm(delta, axis=1)))
    return {"point_mae_m": mae, "point_rms_m": rms, "point_mae_mm": 1000 * mae, "point_rms_mm": 1000 * rms}
