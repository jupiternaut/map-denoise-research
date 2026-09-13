"""Finite surface-dictionary reassociation of frozen, original measurements.

Public estimation takes measured XYZ, scan provenance and supplied sigma only.
No evaluator, truth coordinate, layer label, or injected transform is imported.
This is a conditional development operator, not a full registration pipeline.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import time

import numpy as np
from scipy.special import logsumexp

PROJECT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("_v7_readonly_v5", PROJECT / "exploration_v5/slope_pooling.py")
V5 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(V5)
VARIANTS = ("original", "local_multistart", "reassociate")
SHARING = ("independent", "shared")
QUANTILE_STARTS = ((.2, .8), (.1, .9), (.35, .65))


def freeze(xyz_world_m, scan_id, sigma_mm):
    """Prepare owned read-only arrays using unchanged legal V4/V5 upstream."""
    started = time.perf_counter()
    world, scans = np.array(xyz_world_m, dtype=float, copy=True), np.array(scan_id, copy=True)
    sigma = float(sigma_mm)
    if world.ndim != 2 or world.shape[1] != 3 or not len(world) or not np.isfinite(world).all():
        raise ValueError("finite nonempty N x 3 world metres required")
    if scans.shape != (len(world),) or scans.dtype.kind not in "iu":
        raise ValueError("one integer scan ID per point required")
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite sigma_mm required")
    state = V5._freeze(world, scans, sigma)
    state.update(world=world, scan_id=scans, sigma_mm=sigma)
    if state["v4_info"]["status"] == "APPLY":
        active = state["active"]
        state["original_groups"] = state["node_to_group"][state["assignment"][active]]
        # Reconstruct the exact V3 occupied-grid coordinates, rather than using
        # sequential occupied-cell IDs as though they were spatial coordinates.
        xy = state["local"][:, :2]
        extent = np.maximum(np.ptp(xy, axis=0), 1e-6)
        bins = np.asarray(state["v4_info"]["grid_shape"])
        grid_xy = np.minimum(((xy - xy.min(axis=0)) / extent * bins).astype(int), bins - 1)
        cell_grid, point_cell = np.unique(grid_xy, axis=0, return_inverse=True)
        if not np.array_equal(point_cell[active], state["node_cells"][state["assignment"][active]]):
            raise AssertionError("cell coordinates do not reproduce frozen nodes")
        state.update(cell_grid=cell_grid, point_cell=point_cell)
        keys = ("world", "scan_id", "order", "bias", "basis", "normal", "center", "design",
                "corrected", "weights", "active", "support", "assignment", "node_to_group")
        state["frozen_common_sha256"] = V5._fingerprint(**{k: state[k] for k in keys},
            sigma_mm=np.asarray(sigma), common_scale_mm=np.asarray(state["common_scale"]))
    else:
        state["active"] = np.empty(0, dtype=np.int64)
        state["original_groups"] = np.empty(0, dtype=np.int64)
        state["frozen_common_sha256"] = V5._fingerprint(world=world, scans=scans, support=state["support"])
    state["freeze_seconds"] = time.perf_counter() - started
    for value in state.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return state


def _counter():
    return dict(residual_pair_evaluations=0, weighted_fit_row_visits=0,
                weighted_design_elements=0, least_squares_calls=0,
                residual_dot_multiply_adds=0)


def _work(counter):
    return int(counter["residual_pair_evaluations"] + counter["weighted_fit_row_visits"])


def _wls(x, y, weight, counter):
    root = np.sqrt(weight)
    a = x * root[:, None]
    beta, _, rank, singular = np.linalg.lstsq(a, y * root, rcond=1e-12)
    counter["weighted_fit_row_visits"] += len(y)
    counter["weighted_design_elements"] += a.size
    counter["least_squares_calls"] += 1
    return beta, dict(rank=int(rank), parameter_count=x.shape[1], rows=len(y),
                      effective_mass=float(weight.sum()),
                      effective_condition=float(singular[0] / singular[rank-1]) if rank else None)


def _cost(x, y, beta, candidate, sigma, counter):
    cost = np.full(candidate.shape, np.inf)
    for group in range(candidate.shape[1]):
        take = candidate[:, group]
        count = int(take.sum())
        if count:
            cost[take, group] = .5 * ((y[take] - x[take] @ beta[group]) / sigma) ** 2
            counter["residual_pair_evaluations"] += count
            counter["residual_dot_multiply_adds"] += count * x.shape[1]
    return cost


def _e_step(x, y, beta, candidate, sigma, counter, prior=None):
    """Unit row mass; uniform feasible prior, NOT balanced optimal transport."""
    if not candidate.any(axis=1).all():
        raise ValueError("each active observation needs at least one candidate")
    cost = _cost(x, y, beta, candidate, sigma, counter)
    if prior is None:
        log_prior = -np.log(candidate.sum(axis=1))[:, None]
    else:
        prior = np.broadcast_to(np.asarray(prior, dtype=float), candidate.shape)
        if (not np.isfinite(prior).all() or np.any(prior[candidate] <= 0)
                or np.any(prior[~candidate] != 0) or not np.allclose(prior.sum(axis=1), 1., atol=1e-12, rtol=0)):
            raise ValueError("prior must be positive on candidates, zero elsewhere, with unit row sum")
        log_prior = np.full(candidate.shape, -np.inf)
        log_prior[candidate] = np.log(prior[candidate])
    log_partition = logsumexp(-cost + log_prior, axis=1)
    responsibility = np.exp(-cost + log_prior - log_partition[:, None])
    return responsibility, cost, -log_partition


def _free_energy(r, cost, weights, candidate):
    pos = r > 0
    terms = np.zeros_like(r)
    log_count = np.log(candidate.sum(axis=1))[:, None]
    terms[pos] = r[pos] * (cost[pos] + np.log(r[pos]) + np.broadcast_to(log_count, r.shape)[pos])
    return float(np.dot(weights, terms.sum(axis=1)))


def _m_step(x, y, weights, r, candidate, beta, counter):
    updated, fits = beta.copy(), []
    for group in range(candidate.shape[1]):
        take = candidate[:, group] & (r[:, group] > 0)
        weight = weights[take] * r[take, group]
        if not len(weight) or float(weight.sum()) <= np.finfo(float).tiny:
            fits.append(dict(rank=0, parameter_count=3, rows=0, effective_mass=0., empty_retained=True))
            continue
        updated[group], record = _wls(x[take], y[take], weight, counter)
        fits.append(record)
    return updated, fits


def _hard_assignment(r, candidate, original=None):
    result = np.argmax(r, axis=1)
    # The tolerance only resolves numerical ties; it is not calibrated evidence.
    best = r[np.arange(len(r)), result]
    if original is not None:
        valid = (original >= 0) & (original < r.shape[1])
        rows = np.flatnonzero(valid)
        keep = candidate[rows, original[rows]] & np.isclose(r[rows, original[rows]], best[rows], atol=1e-12, rtol=0)
        result[rows[keep]] = original[rows[keep]]
    sorted_r = np.sort(r, axis=1)
    margin = sorted_r[:, -1] - (sorted_r[:, -2] if r.shape[1] > 1 else 0.)
    entropy = -np.sum(np.where(r > 0, r * np.log(np.maximum(r, np.finfo(float).tiny)), 0.), axis=1)
    return result, entropy, margin


def _dictionary(state):
    active = state["active"]
    cells = state["point_cell"][active]
    groups = state["original_groups"]
    size = int(groups.max()) + 1
    cell_grid = state["cell_grid"]
    near = np.max(abs(cell_grid[:, None] - cell_grid[None, :]), axis=2) <= 1
    candidate = np.zeros((len(active), size), dtype=bool)
    records = []
    for group in range(size):
        origin_cells = np.unique(cells[groups == group])
        candidate[:, group] = near[cells][:, origin_cells].any(axis=1)
        records.append(dict(group=group, origin_cells=origin_cells.tolist(),
                            reachable_cells=np.flatnonzero(near[:, origin_cells].any(axis=1)).tolist(),
                            original_points=int(np.count_nonzero(groups == group))))
    if not candidate[np.arange(len(active)), groups].all():
        raise AssertionError("original candidate became inaccessible")
    return candidate, records


def _reassociate(state, budget):
    active, counter = state["active"], _counter()
    x, y, weights = state["design"][active], state["corrected"][active], state["weights"][active]
    candidate, records = _dictionary(state)
    # Start from the same planes as V4, with no synthetic component birth.
    beta = np.asarray(state["v4_info"]["group_coefficients_common_origin"], dtype=float).copy()
    r, cost, nll = _e_step(x, y, beta, candidate, state["sigma_mm"], counter)
    trace = [dict(step="initial_E", iteration=0, free_energy=float(np.dot(weights, nll)))]
    fits = []
    for iteration in range(1, budget + 1):
        update, proposed_fits = _m_step(x, y, weights, r, candidate, beta, counter)
        # Evaluate the same fixed responsibility objective after M; this extra
        # diagnostic evaluation is charged to the actual work budget.
        m_cost = _cost(x, y, update, candidate, state["sigma_mm"], counter)
        m_value = _free_energy(r, m_cost, weights, candidate)
        before = trace[-1]["free_energy"]
        tol = 1e-8 * max(1., abs(before))
        if m_value > before + tol:
            # Rare numerical failure: retain the actual previous feasible model.
            trace.append(dict(step="M_numerical_rejected", iteration=iteration, free_energy=before,
                              proposed_free_energy=m_value, rejected_component_fits=proposed_fits))
            break
        beta, cost = update, m_cost
        fits = proposed_fits
        trace.append(dict(step="M", iteration=iteration, free_energy=m_value))
        log_prior = -np.log(candidate.sum(axis=1))[:, None]
        log_partition = logsumexp(-cost + log_prior, axis=1)
        r = np.exp(-cost + log_prior - log_partition[:, None])
        nll = -log_partition
        trace.append(dict(step="E", iteration=iteration, free_energy=float(np.dot(weights, nll))))
    groups, entropy, margin = _hard_assignment(r, candidate, state["original_groups"])
    return dict(groups=groups, responsibility=r, candidate_mask=candidate, entropy=entropy, margin=margin,
                coefficients=beta, objective_trace=trace, component_fits=fits, work=counter,
                dictionary=records, dictionary_size=len(beta), grouping_description="MAP original dictionary IDs",
                search_objective=float(np.dot(weights, nll)), search_objective_kind="fixed-candidate empirical free energy",
                compute_target=None, compute_target_met=None)


def _local_multistart(state, budget):
    """Input-only local K=1/2 starts; extra compute does not see neighbouring planes.

The scheduler targets reassociation's measured-pair + fitted-row search budget.
Final pooling/fitting costs are separately disclosed. This is a work proxy,
not equal FLOPs or an independent published strong baseline.
    """
    active, counter = state["active"], _counter()
    x, y, weights = state["design"][active], state["corrected"][active], state["weights"][active]
    cells = state["point_cell"][active]
    reachable, _ = _dictionary(state)
    # One initial E plus budget*(M row-visits + post-M residual evaluation).
    target = int(reachable.sum()) * (1 + 2 * budget)
    jobs, pending, cell_rows = [], [], {}
    for cell in np.unique(cells):
        rows = np.flatnonzero(cells == cell)
        cell_rows[int(cell)] = rows
        one, fit = _wls(x[rows], y[rows], weights[rows], counter)
        candidate = np.ones((len(rows), 1), dtype=bool)
        r, cost, nll = _e_step(x[rows], y[rows], one[None], candidate, state["sigma_mm"], counter)
        jobs.append(dict(cell=int(cell), rows=rows, k=1, start="single_WLS", beta=one[None],
                         candidate=candidate, r=r, cost=cost, nll=nll, iterations=0,
                         trace=[float(np.dot(weights[rows], nll))], fits=[fit]))
        original_nodes = np.unique(state["assignment"][active[rows]])
        if len(original_nodes) == 2:
            old = []
            for node in original_nodes:
                sub = rows[state["assignment"][active[rows]] == node]
                coefficient, _ = _wls(x[sub], y[sub], weights[sub], counter)
                old.append(coefficient)
            pending.append((int(cell), "original_two_nodes", np.asarray(old)))
        residual = y[rows] - x[rows] @ one
        counter["residual_pair_evaluations"] += len(rows)
        counter["residual_dot_multiply_adds"] += 3 * len(rows)
        for q in QUANTILE_STARTS:
            beta = np.repeat(one[None], 2, axis=0)
            beta[:, 0] += np.quantile(residual, q)
            pending.append((int(cell), "residual_quantiles_" + "_".join(map(str, q)), beta))
    # Interleave starts across cells before the next quantile family. The old
    # node start is included where it exists; no evaluation score orders starts.
    queues = {cell: [p for p in pending if p[0] == cell] for cell in cell_rows}
    pending = [queues[cell][depth] for depth in range(max(map(len, queues.values())))
               for cell in sorted(queues) if depth < len(queues[cell])]
    cursor, completed_sweeps = 0, 0
    while _work(counter) < target:
        changed = False
        # Add at most one start per cell per sweep, then let existing starts work.
        for _ in range(len(cell_rows)):
            if cursor >= len(pending):
                break
            cell, name, beta = pending[cursor]
            rows = cell_rows[cell]
            charge = 2 * len(rows)
            if _work(counter) + charge > target:
                break
            cursor += 1
            candidate = np.ones((len(rows), 2), dtype=bool)
            r, cost, nll = _e_step(x[rows], y[rows], beta, candidate, state["sigma_mm"], counter)
            jobs.append(dict(cell=cell, rows=rows, k=2, start=name, beta=beta, candidate=candidate,
                             r=r, cost=cost, nll=nll, iterations=0,
                             trace=[float(np.dot(weights[rows], nll))], fits=[]))
            changed = True
        for job in jobs:
            if job["k"] == 1:
                continue
            rows, candidate = job["rows"], job["candidate"]
            charge = 4 * len(rows)
            if _work(counter) + charge > target:
                continue
            beta, fits = _m_step(x[rows], y[rows], weights[rows], job["r"], candidate, job["beta"], counter)
            r, cost, nll = _e_step(x[rows], y[rows], beta, candidate, state["sigma_mm"], counter)
            value = float(np.dot(weights[rows], nll))
            if value > job["trace"][-1] + 1e-8 * max(1., abs(job["trace"][-1])):
                raise FloatingPointError("local EM objective increased")
            job.update(beta=beta, r=r, cost=cost, nll=nll, fits=fits, iterations=job["iterations"]+1)
            job["trace"].append(value)
            changed = True
        completed_sweeps += 1
        if not changed:
            break
    # Conditional BIC-like K/start selection. All original rows/weights are
    # used and K=1 competes explicitly; this is not a physical layer certificate.
    selected, job_records = [], []
    for job in jobs:
        rows = job["rows"]
        score = 2 * float(np.dot(weights[rows], job["nll"])) + 3 * job["k"] * np.log(max(len(rows), 2))
        hard, _, _ = _hard_assignment(job["r"], job["candidate"])
        valid = job["k"] == 1 or np.bincount(hard, minlength=2).min() >= 5
        job["score"], job["valid"] = score, bool(valid)
        job_records.append(dict(cell=job["cell"], start=job["start"], k=job["k"], score=score,
                                eligible=bool(valid), iterations=job["iterations"], objective_trace=job["trace"]))
    for cell in sorted(cell_rows):
        selected.append(min((j for j in jobs if j["cell"] == cell and j["valid"]), key=lambda j: j["score"]))
    nodes, point_node = [], np.full(len(active), -1, dtype=int)
    component_count = sum(j["k"] for j in selected)
    soft = np.zeros((len(active), component_count))
    mask = np.zeros_like(soft, dtype=bool)
    cursor = 0
    for job in selected:
        rows, k = job["rows"], job["k"]
        hard, _, _ = _hard_assignment(job["r"], job["candidate"])
        soft[np.ix_(rows, np.arange(cursor, cursor+k))] = job["r"]
        mask[np.ix_(rows, np.arange(cursor, cursor+k))] = True
        for layer in range(k):
            take = rows[hard == layer]
            point_node[take] = cursor + layer
            fit = V5._V4._sufficient_fit(take, x, y, weights, state["sigma_mm"])
            counter["weighted_fit_row_visits"] += len(take)
            counter["weighted_design_elements"] += 3 * len(take)
            counter["least_squares_calls"] += 1
            fit.update(cell=job["cell"], local_layer=layer)
            nodes.append(fit)
        cursor += k
    search_work = dict(counter)
    allowed, _ = V5._V4._compatibility(nodes, state["sigma_mm"])
    pooled, history = V5._V4._pool(nodes, allowed, state["sigma_mm"], len(active))
    mapping = np.empty(len(nodes), dtype=int)
    pooled_r = np.zeros((len(active), len(pooled)))
    pooled_mask = np.zeros_like(pooled_r, dtype=bool)
    records = []
    for group, pooled_group in enumerate(pooled):
        members = pooled_group["members"]
        mapping[members] = group
        pooled_r[:, group] = soft[:, members].sum(axis=1)
        pooled_mask[:, group] = mask[:, members].any(axis=1)
        records.append(dict(group=group, member_nodes=members,
                            origin_cells=sorted({int(nodes[n]["cell"]) for n in members})))
    groups = mapping[point_node]
    _, entropy, margin = _hard_assignment(pooled_r, pooled_mask)
    return dict(groups=groups, responsibility=pooled_r, candidate_mask=pooled_mask, entropy=entropy, margin=margin,
                coefficients=np.asarray([g["fit"]["beta"] for g in pooled]), objective_trace=[],
                component_fits=[], work=counter, dictionary=records, dictionary_size=len(pooled),
                search_objective=float(sum(j["score"] for j in selected)),
                search_objective_kind="sum of local input BIC-like scores, not comparable to reassociation F",
                grouping_description="local MAP nodes then unchanged V4 complete-link pooling",
                compute_target=target, compute_target_met=abs(_work(search_work)-target) <= max(1., .1*target),
                compute_proxy_ratio=float(_work(search_work)/max(target, 1)),
                compute_target_overshoot=max(0, _work(search_work)-target),
                budget_scope="search plus final local sufficient fits; final pooling and output fit separately timed",
                local_jobs=job_records, local_selected=[dict(cell=j["cell"], k=j["k"], start=j["start"],
                    score=j["score"], iterations=j["iterations"]) for j in selected],
                local_starts_not_scheduled=len(pending)-cursor, pooling_merge_count=len(history),
                no_neighbour_plane_feedback=True)


def _final_fit(state, groups, sharing):
    active = state["active"]
    ids, compact = np.unique(groups, return_inverse=True)
    prediction, fit = V5._fit_model(state["design"][active, 1:], state["corrected"][active],
        state["weights"][active], compact, np.arange(len(ids)),
        "shared_group_slope" if sharing == "shared" else "node_intercepts")
    predicted = state["corrected"].copy()
    predicted[active] = prediction
    ordered = state["ordered_world"].copy()
    use = state["support"]
    ordered[use] += ((predicted[use]-state["local"][use, 2])/1000.)[:, None]*state["normal"]
    output = state["world"].copy()
    output[state["order"]] = ordered
    fit["fitted_dictionary_ids"] = ids.tolist()
    fit["final_fit_row_visits"] = len(active)
    fit["final_fit_design_elements"] = len(active)*fit["fit_parameter_count"]
    return output, fit


def fit_frozen(state, variant="reassociate", budget=3, sharing="independent"):
    """Return (same-order XYZ, JSON metadata, ndarray artifacts).

Budget counts feedback sweeps, or a matched local-search work target; it is not
an equal wall-time promise. Returned group IDs name dictionary components.
    """
    started = time.perf_counter()
    if variant not in VARIANTS or sharing not in SHARING:
        raise ValueError("unknown variant or sharing")
    if not isinstance(budget, (int, np.integer)) or isinstance(budget, bool) or not 0 <= budget <= 100:
        raise ValueError("integer budget in [0,100] required")
    world, active = state["world"], state["active"]
    n = len(world)
    support = np.empty(n, dtype=bool)
    support[state["order"]] = state["support"]
    info = dict(method="finite_surface_reassociation_v7", variant=variant, sharing=sharing, budget=int(budget),
                truth_fields_used=[], evaluation_only_oracle=False, sigma_mm=state["sigma_mm"],
                frozen_common_sha256=state["frozen_common_sha256"], freeze_seconds=state["freeze_seconds"],
                supported_fraction=float(support.mean()), active_point_count=len(active),
                point_count=n, input_units="world metres", bias_mm=state["bias"].tolist(),
                normal_world=state["normal"].tolist(), gauge=state["v4_info"]["gauge"],
                frozen_fields=["normal", "bias", "weights", "fit rows", "output support", "cells"],
                observation_source="bias-corrected original measurement, never projected output",
                interpretation="conditional prototype; fixed support is attribution choice, not full architecture",
                prior="uniform among each point's feasible candidates; no equal target mass constraint",
                final_action="hard component MAP, then original-measurement refit; not posterior geometric mean")
    if not len(active):
        artifacts = dict(support_mask=support, group_ids=np.full(n, -1, dtype=int),
                         responsibility=np.zeros((n, 0)), candidate_mask=np.zeros((n, 0), dtype=bool),
                         entropy=np.zeros(n), margin=np.zeros(n), coefficients=np.empty((0, 3)),
                         objective_trace=np.empty(0), active_mask=np.zeros(n, dtype=bool))
        info.update(status="UNSUPPORTED", fit_rank=0, fit_parameter_count=0, group_count=0,
                    input_edit_rms_mm=0., search_work=_counter(), fit_seconds=time.perf_counter()-started,
                    total_seconds=state["freeze_seconds"]+time.perf_counter()-started)
        return world.copy(), info, artifacts
    if variant == "original":
        groups = state["original_groups"].copy()
        size = int(groups.max())+1
        r = np.eye(size)[groups]
        candidate, dictionary = _dictionary(state)
        search = dict(groups=groups, responsibility=r, candidate_mask=candidate, entropy=np.zeros(len(active)),
                      margin=np.ones(len(active)), coefficients=np.asarray(state["v4_info"]["group_coefficients_common_origin"]),
                      objective_trace=[], component_fits=[], work=_counter(), dictionary=dictionary,
                      dictionary_size=size, grouping_description="unchanged original compatible groups",
                      search_objective=None, search_objective_kind="none: fixed original assignments",
                      compute_target=None, compute_target_met=None)
    elif variant == "reassociate":
        search = _reassociate(state, budget)
    else:
        search = _local_multistart(state, budget)
    search_seconds = time.perf_counter()-started
    fit_started = time.perf_counter()
    output, fit = _final_fit(state, search["groups"], sharing)
    final_fit_seconds = time.perf_counter()-fit_started
    if not np.array_equal(output[~support], world[~support]) or not np.isfinite(output).all():
        raise AssertionError("invalid output or unsupported rows modified")
    original_rows = state["order"][active]
    group_ids = np.full(n, -1, dtype=int)
    group_ids[original_rows] = search["groups"]
    size = search["responsibility"].shape[1]
    r = np.zeros((n, size)); r[original_rows] = search["responsibility"]
    candidate = np.zeros((n, size), dtype=bool); candidate[original_rows] = search["candidate_mask"]
    entropy, margin = np.zeros(n), np.zeros(n)
    entropy[original_rows], margin[original_rows] = search["entropy"], search["margin"]
    active_mask = np.zeros(n, dtype=bool); active_mask[original_rows] = True
    arrays = dict(support_mask=support, group_ids=group_ids, responsibility=r, candidate_mask=candidate,
                  entropy=entropy, margin=margin, coefficients=search["coefficients"],
                  objective_trace=np.asarray([s["free_energy"] for s in search["objective_trace"]]),
                  active_mask=active_mask, original_group_ids=np.full(n, -1, dtype=int))
    arrays["original_group_ids"][original_rows] = state["original_groups"]
    group_records = []
    for group in np.unique(search["groups"]):
        take = search["groups"] == group
        group_records.append(dict(group=int(group), point_count=int(take.sum()),
                                  weight=float(state["weights"][active[take]].sum()),
                                  original_groups=np.unique(state["original_groups"][take]).tolist(),
                                  original_cells=np.unique(state["point_cell"][active[take]]).tolist()))
    info.update(fit, status="APPLY" if support.any() else "UNSUPPORTED", group_count=len(group_records),
                groups=group_records, dictionary_size=search["dictionary_size"], dictionary=search["dictionary"],
                dictionary_capacity_parameters=3*search["dictionary_size"],
                dictionary_candidate_pair_count=int(search["candidate_mask"].sum()),
                association_change_fraction=float(np.mean(search["groups"] != state["original_groups"])) if variant != "local_multistart" else None,
                association_change_scope="dictionary IDs comparable only original/reassociate; local pool IDs arbitrary",
                grouping_description=search["grouping_description"], search_work=search["work"],
                search_work_proxy=_work(search["work"]), compute_target=search["compute_target"],
                compute_target_met=search["compute_target_met"], search_objective=search["search_objective"],
                budget_matching_scope="target is planned search-work proxy, not actual paired wall-time or FLOP equivalence; compare both arms' measured counters",
                search_objective_kind=search["search_objective_kind"], objective_steps=search["objective_trace"],
                component_fits=search["component_fits"], entropy_mean=float(search["entropy"].mean()),
                margin_mean=float(search["margin"].mean()),
                confidence_scope="model-conditional ambiguity only, not calibrated correctness probability",
                responsibility_stage="end of soft search, before hard assignment and final refit; local responsibilities additionally aggregated by final pooling",
                search_seconds=search_seconds, final_fit_seconds=final_fit_seconds,
                fit_seconds=time.perf_counter()-started,
                input_edit_rms_mm=float(np.sqrt(np.mean(np.sum((output-world)**2, axis=1))))*1000.,
                actual_changed_fraction=float(np.mean(np.any(output != world, axis=1))),
                output_sha256=V5._fingerprint(xyz_world_m=output),
                group_assignment_sha256=V5._fingerprint(groups=group_ids))
    for key in ("local_jobs", "local_selected", "local_starts_not_scheduled", "pooling_merge_count", "compute_proxy_ratio", "compute_target_overshoot", "budget_scope"):
        if key in search:
            info[key] = search[key]
    info["total_seconds"] = state["freeze_seconds"] + info["fit_seconds"]
    return output, info, arrays


def estimate(xyz_world_m, scan_id, sigma_mm, variant="reassociate", budget=3, sharing="independent"):
    return fit_frozen(freeze(xyz_world_m, scan_id, sigma_mm), variant, budget, sharing)
