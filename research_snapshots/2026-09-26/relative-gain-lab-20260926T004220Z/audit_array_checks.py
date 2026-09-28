"""Independent NumPy checks; never used to fit or choose deployed policies."""
from __future__ import annotations

import numpy as np


def metrics(e0, e1, accepted, displacement, support):
    """e0/e1 are squared NN errors in mm²; displacement is candidate length in mm."""
    e0, e1, displacement = (np.asarray(x, dtype=np.float64) for x in (e0, e1, displacement))
    accepted, support = (np.asarray(x, dtype=bool) for x in (accepted, support))
    assert e0.ndim == 1 and all(x.shape == e0.shape for x in (e1, displacement, accepted, support))
    assert support.any() and all(np.isfinite(x).all() for x in (e0, e1, displacement))
    assert (e0 >= 0).all() and (e1 >= 0).all() and (displacement >= 0).all()
    squared = np.where(accepted, e1, e0)[support]
    distance = np.sqrt(squared)
    base = e0[support]
    gain = base - squared
    delta = distance - np.sqrt(base)
    actual = np.where(accepted, displacement, 0.0)
    return {
        "source_MSE_mm2": float(squared.mean()),
        "source_MAE_mm": float(distance.mean()),
        "source_p95_mm": float(np.quantile(distance, 0.95)),
        "relative_MSE": float(squared.mean() / max(float(base.mean()), 1e-6)),
        "improved_fraction": float((delta < -0.1).mean()),
        "harmed_fraction": float((delta > 0.1).mean()),
        "unchanged_band_fraction": float((np.abs(delta) <= 0.1).mean()),
        "benefit_sum_mm2": float(np.maximum(gain, 0).sum()),
        "harm_sum_mm2": float(np.maximum(-gain, 0).sum()),
        "accepted_fraction": float(accepted.mean()),
        "accepted_support_fraction": float(accepted[support].mean()),
        "moved_fraction": float((actual > 1e-7).mean()),
        "move_RMS_mm": float(np.sqrt(np.square(actual).mean())),
    }


def verify_matching(selected, control, displacement, support, edges=None):
    """Return counts; assert exact count matching separately inside/outside support."""
    selected, control, support = (np.asarray(v, dtype=bool) for v in (selected, control, support))
    displacement = np.asarray(displacement, dtype=float)
    assert all(v.shape == selected.shape for v in (control, support, displacement))
    bins = np.zeros(len(selected), dtype=int) if edges is None else np.searchsorted(edges, displacement, side="right")
    records = []
    for is_supported in (False, True):
        for bin_id in np.unique(bins):
            group = (support == is_supported) & (bins == bin_id)
            left, right = int((selected & group).sum()), int((control & group).sum())
            assert left == right, (is_supported, int(bin_id), left, right)
            records.append((is_supported, int(bin_id), left))
    return records


def threshold_rows(scores, e0, e1, case_ids, conditions, displacement, support, thresholds):
    """Brute-force equal-case calibration objective, independent of runner reducers."""
    scores, e0, e1, displacement = (np.asarray(v, dtype=float) for v in (scores, e0, e1, displacement))
    case_ids, conditions, support = np.asarray(case_ids), np.asarray(conditions), np.asarray(support, dtype=bool)
    case_masks = [(case_ids == case_id) & support for case_id in np.unique(case_ids)]
    assert all(mask.any() for mask in case_masks)
    case_conditions = []
    for mask in case_masks:
        condition = np.unique(conditions[mask])
        assert len(condition) == 1
        case_conditions.append(str(condition[0]))
    native = np.array([c == "native" for c in case_conditions])
    injected = np.array([c in ("minus3", "plus3") for c in case_conditions])
    assert native.any() and injected.any()
    identity_mse = np.array([e0[mask].mean() for mask in case_masks])
    rows = []
    for threshold in thresholds:
        accepted = scores > threshold
        chosen = np.where(accepted, e1, e0)
        mse = np.array([chosen[mask].mean() for mask in case_masks])
        relative = mse / np.maximum(identity_mse, 1e-6)
        move = np.array([(accepted[mask] & (displacement[mask] > 1e-7)).mean() for mask in case_masks])
        accept_fraction = np.array([accepted[mask].mean() for mask in case_masks])
        rows.append({
            "threshold": float(threshold), "objective": float(relative.mean()),
            "mean_move_fraction": float(move.mean()),
            "accepted_fraction": float(accept_fraction.mean()),
            "native_MSE": float(mse[native].mean()),
            "native_identity_MSE": float(identity_mse[native].mean()),
            "injected_relative_MSE": float(relative[injected].mean()),
            "native_feasible": bool(mse[native].mean() <= identity_mse[native].mean()
                                    and relative[injected].mean() <= 0.95),
        })
    return rows


def pick_threshold(rows, native_priority=False):
    eligible = [row for row in rows if row["native_feasible"] or not native_priority]
    if not eligible:
        return {"threshold": float("inf"), "status": "infeasible_KEEP"}
    # Exact ties only. A runner using a numerical tolerance should state it.
    return min(eligible, key=lambda row: (row["objective"], row["mean_move_fraction"]))
