"""Evaluation-only association x slope-sharing counterfactual on frozen V5.

The ordinary API has no truth argument. The explicitly named oracle API accepts
only source-layer labels, never truth coordinates or surface parameters. No
files are read during fitting and no new measurements or neighborhoods enter.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
V5_PATH = PROJECT / "exploration_v5" / "slope_pooling.py"
_spec = importlib.util.spec_from_file_location("_association_frozen_v5", V5_PATH)
V5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V5)
ARMS = ("original_independent", "original_shared", "oracle_independent", "oracle_shared")
SHARING = ("independent", "shared")


def freeze(xyz_world_m, scan_id, sigma_mm):
    """Legal-input boundary. Arrays in returned state are read-only owned copies."""
    world = np.array(xyz_world_m, dtype=float, copy=True)
    scans = np.array(scan_id, copy=True)
    sigma = float(sigma_mm)
    if world.ndim != 2 or world.shape[1] != 3 or not len(world) or not np.isfinite(world).all():
        raise ValueError("finite nonempty N by 3 world metres required")
    if scans.shape != (len(world),) or scans.dtype.kind not in "iu":
        raise ValueError("one integer scan ID per point required")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite sigma_mm required")
    start = time.perf_counter()
    state = V5._freeze(world, scans, sigma)
    state.update(world=world, scan_id=scans, sigma_mm=sigma,
                 freeze_seconds=time.perf_counter()-start)
    if state["v4_info"]["status"] != "APPLY":
        raise ValueError("this bounded diagnostic requires supported frozen V4 initialization")
    state["original_groups"] = state["node_to_group"][state["assignment"][state["active"]]]
    state["common_sha256"] = common_fingerprints(state)
    for value in state.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return state


def common_fingerprints(state):
    keys = ("world", "scan_id", "order", "bias", "basis", "normal", "center", "design",
            "corrected", "weights", "active", "support", "assignment", "node_to_group")
    result = {key + "_sha256": V5._fingerprint(**{key: state[key]}) for key in keys}
    result["frozen_common_sha256"] = V5._fingerprint(
        **{key: state[key] for key in keys}, sigma_mm=np.asarray(state["sigma_mm"]),
        common_scale_mm=np.asarray(state["common_scale"]))
    return result


def fit_original(state, sharing="independent"):
    """No GT parameter or evaluator input: original input-derived associations."""
    return _fit(state, state["original_groups"], sharing, "original", None)


def fit_oracle_split(state, gt_layer, sharing="independent"):
    """EVALUATION ONLY: split each original group by layer label; never merge."""
    labels = np.asarray(gt_layer)
    if labels.shape != (len(state["world"]),) or labels.dtype.kind not in "iu":
        raise ValueError("oracle requires exactly one integer source-layer label per input row")
    ordered_labels = labels[state["order"]][state["active"]]
    pairs = np.c_[state["original_groups"], ordered_labels]
    keys, groups = np.unique(pairs, axis=0, return_inverse=True)
    # Encoding labels by value cannot affect fitted geometry: only the partition matters.
    return _fit(state, groups, sharing, "oracle", keys)


def _fit(state, groups, sharing, association, oracle_keys):
    if sharing not in SHARING:
        raise ValueError(f"sharing must be one of {SHARING}")
    start = time.perf_counter()
    active = state["active"]
    groups = np.asarray(groups, dtype=np.int64)
    xy = state["design"][active, 1:]
    values, weights = state["corrected"][active], state["weights"][active]
    group_count = int(groups.max())+1
    # V5's common WLS helper treats the active point group as an artificial node;
    # the identity map introduces no second grouping or changed information.
    prediction, fit = V5._fit_model(
        xy, values, weights, groups, np.arange(group_count),
        "shared_group_slope" if sharing == "shared" else "node_intercepts")
    predicted = state["corrected"].copy()
    predicted[active] = prediction
    ordered_output = state["ordered_world"].copy()
    support = state["support"]
    ordered_output[support] += ((predicted[support]-state["local"][support, 2])/1000.)[:, None]*state["normal"]
    output = state["world"].copy()
    output[state["order"]] = ordered_output
    support_original = np.empty(len(output), dtype=bool)
    support_original[state["order"]] = support
    point_groups = np.full(len(output), -1, dtype=np.int64)
    point_groups[state["order"][active]] = groups
    if not np.array_equal(output[~support_original], state["world"][~support_original]):
        raise AssertionError("fixed unsupported point changed")
    if not np.isfinite(output).all() or output.shape != state["world"].shape:
        raise AssertionError("invalid output")
    records = []
    for group in range(group_count):
        take = groups == group
        parents = np.unique(state["original_groups"][take])
        if len(parents) != 1:
            raise AssertionError("oracle expanded beyond an original group")
        nodes = np.unique(state["assignment"][active[take]])
        record = dict(group=group, original_group=int(parents[0]), point_count=int(take.sum()),
                      total_weight=float(weights[take].sum()), original_nodes=nodes.tolist(),
                      original_cells=np.unique(state["node_cells"][nodes]).tolist(),
                      measured_common_xy_min_mm=(xy[take].min(axis=0)*state["common_scale"]).tolist(),
                      measured_common_xy_max_mm=(xy[take].max(axis=0)*state["common_scale"]).tolist())
        if oracle_keys is not None:
            record["evaluation_only_split_label"] = int(oracle_keys[group, 1])
        records.append(record)
    info = dict(fit, method="association_counterfactual_v6", arm=association+"_"+sharing,
                association=association, sharing=sharing, evaluation_only_oracle=association=="oracle",
                truth_fields_used=["gt_layer"] if association=="oracle" else [],
                group_count=group_count, groups=records,
                group_assignment_sha256=V5._fingerprint(group_assignment=point_groups),
                frozen_common_sha256=state["common_sha256"]["frozen_common_sha256"],
                common_fingerprints=state["common_sha256"],
                bias_mm=state["bias"].tolist(), normal_world=state["normal"].tolist(),
                gauge=state["v4_info"]["gauge"], common_tangent_scale_mm=state["common_scale"],
                sigma_mm=state["sigma_mm"], active_point_count=len(active),
                total_fit_weight=float(weights.sum()), supported_fraction=float(support.mean()),
                observations="frozen bias-corrected original measurements, never projected cloud",
                frozen_common_scope="all observation rows, weights, upstream geometry, bias and support; not parameter count",
                output_sha256=V5._fingerprint(xyz_world_m=output),
                freeze_seconds=state["freeze_seconds"], fit_seconds=time.perf_counter()-start)
    info["intercept_scope"] = "one per original compatible group" if association=="original" else "one per original group x GT-layer intersection"
    info["slope_scope"] = "all groups" if sharing=="shared" else "one per current group"
    return output, info, point_groups
