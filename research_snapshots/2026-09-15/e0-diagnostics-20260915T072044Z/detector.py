"""Bounded development diagnostic A; test truth enters evaluate_rejections only.

Run with the frozen environment in AGENTS.md. No E0 files are written. The
selection boundary is select_points(s0, observed_stratum, calibration, ...).
Calibration labels are used by collect_calibration to construct calibration
scores, as declared in PROTOCOL.md; test labels are not accepted by a selector.
"""
import csv
import hashlib
import inspect
import json
import math
import time
from pathlib import Path

import numpy as np

from common import ROOT, load_old, write_json

E0 = load_old()
BUDGETS = (2000, 8000, 32000)
CHUNK = 2000
SEEDS = (101, 102, 103)
F_BIGS = (0.01, 0.10)
Q = 0.10
POINT_ALPHA = 0.10


def arrays_hash(**arrays):
    """Stable hash of names, shapes, dtypes and array content; no pickle."""
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        arr = np.ascontiguousarray(value)
        if arr.dtype.hasobject:
            arr = arr.astype(str)
        digest.update(json.dumps([name, list(arr.shape), arr.dtype.str]).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def calibrate_scores(calibration_scores, calibration_stratum, min_stratum=50):
    """Calibrate with declared calibration labels; preserve old pooled fallback."""
    scores = np.asarray(calibration_scores)
    strata = np.asarray(calibration_stratum)
    if scores.ndim != 1 or scores.shape != strata.shape:
        raise ValueError('calibration scores and strata must be matching vectors')
    if len(scores) == 0 or not np.all(np.isfinite(scores)):
        raise ValueError('calibration scores must be nonempty and finite')
    if not np.all(np.isin(strata, (0, 1, 2))):
        raise ValueError('invalid calibration stratum')
    result = {}
    for k in range(3):
        mask = strata == k
        fallback = int(mask.sum()) < min_stratum
        result[k] = {'scores': np.sort(scores if fallback else scores[mask]),
                     'n_stratum': int(mask.sum()), 'fallback': fallback}
    return result


def test_pvalues(s0, observed_stratum, calibration):
    """Old tie-conservative conformal p-values, using observed test inputs only."""
    s0 = np.asarray(s0)
    strata = np.asarray(observed_stratum)
    if s0.ndim != 1 or s0.shape != strata.shape or not np.all(np.isfinite(s0)):
        raise ValueError('observed inputs must be matching finite vectors')
    if not np.all(np.isin(strata, (0, 1, 2))):
        raise ValueError('invalid observed stratum')
    p = np.empty(len(s0))
    floors = np.empty(len(s0))
    for k in range(3):
        mask = strata == k
        scores = calibration[k]['scores']
        nk = len(scores)
        count = nk - np.searchsorted(scores, s0[mask], side='left')
        p[mask] = (1.0 + count) / (nk + 1.0)
        floors[mask] = 1.0 / (nk + 1.0)
    return p, floors


def bh_rank_diagnostics(p, q=Q):
    """Full step-up condition; floor counts are descriptive, not a BH test."""
    p = np.asarray(p)
    if p.ndim != 1 or len(p) == 0 or np.any((p < 0) | (p > 1)) or not np.all(np.isfinite(p)):
        raise ValueError('p must be a nonempty vector in [0,1]')
    if not 0 < q <= 1:
        raise ValueError('q must be in (0,1]')
    order = np.argsort(p, kind='stable')
    sorted_p = p[order]
    rank = np.arange(1, len(p) + 1)
    threshold = q * rank / len(p)
    margin = threshold - sorted_p
    passing = np.flatnonzero(margin >= 0)
    rejected = E0.bh_reject(p, q)  # unchanged implementation in frozen old E0
    k_star = int(passing[-1] + 1) if len(passing) else 0
    assert int(rejected.sum()) == k_star
    summary = {'n_rejected': k_star, 'first_passing_rank': int(passing[0] + 1) if len(passing) else None,
               'last_passing_rank': k_star, 'n_passing_ranks': int(len(passing)),
               'max_margin': float(margin.max()), 'max_margin_rank': int(margin.argmax() + 1),
               'smallest_p': float(sorted_p[0]), 'threshold_at_last_passing_rank': float(threshold[k_star - 1]) if k_star else None}
    return rejected, summary, {'sort_order': order, 'sorted_p': sorted_p, 'rank': rank,
                               'bh_threshold': threshold, 'bh_margin': margin}


def select_points(s0, observed_stratum, calibration, q=Q, point_alpha=POINT_ALPHA):
    """Truth-free selector API. Pointwise rejection is a diagnostic, no FDR claim."""
    p, floors = test_pvalues(s0, observed_stratum, calibration)
    bh, summary, ranks = bh_rank_diagnostics(p, q)
    return {'p': p, 'p_floor': floors, 'bh': bh, 'pointwise': p <= point_alpha,
            'bh_summary': summary, 'ranks': ranks}


def evaluate_rejections(rejected, test_truth, subset, multimodal):
    """EVALUATION ONLY; masks below never feed a calibration or selector."""
    rejected = np.asarray(rejected, dtype=bool)
    truth = np.asarray(test_truth)
    subset = np.asarray(subset)
    within = np.abs(truth) <= E0.EPS_TOL
    masks = {'all': np.ones(len(truth), dtype=bool), 'big': subset == 'big',
             'within_tolerance': within, 'outside_tolerance': ~within,
             'misfit': subset == 'misfit', 'twosheet_back': subset == 'twosheet_back',
             'twosheet_front': subset == 'twosheet_front', 'base': subset == 'base',
             'multimodal': np.asarray(multimodal, dtype=bool),
             'misfit_within_tolerance': (subset == 'misfit') & within,
             'back_within_tolerance': (subset == 'twosheet_back') & within}
    return {name: {'n': int(mask.sum()), 'n_rejected': int((rejected & mask).sum()),
                   'rejection_rate': float(rejected[mask].mean()) if mask.any() else None}
            for name, mask in masks.items()}


def collect_calibration(seed, fb_code, props, cutoffs):
    """One matched draw in fixed 2k chunks; every budget reuses its prefix.

    The RNG identity for the first 2k chunk matches old E0's exch calibration.
    Matching uses the generator's declared/realized mixture, not selector labels.
    """
    rng = np.random.default_rng([seed, fb_code, 10, 1])
    pieces = {name: [] for name in ('score', 'minimum', 'contrast', 'stratum', 'truth', 'subset', 'mm')}
    checkpoints = {}
    curve_digest = hashlib.sha256()
    t0 = time.perf_counter()
    for end in range(CHUNK, max(BUDGETS) + 1, CHUNK):
        block = E0.gen_calset(rng, CHUNK, 'exch', props)
        scores, minimum, _, contrast = E0.scores(block['c'])
        vals = {'score': E0.interp_rows(scores, block['tstar']), 'minimum': minimum,
                'contrast': contrast, 'stratum': np.digitize(contrast, cutoffs),
                'truth': block['tstar'], 'subset': block['subset'].astype('U20'), 'mm': block['mm']}
        curve_digest.update(np.ascontiguousarray(block['c']).tobytes())
        for name, value in vals.items():
            pieces[name].append(value.copy())
        if end in BUDGETS:
            joined = {name: np.concatenate(parts) for name, parts in pieces.items()}
            checkpoints[end] = {'prefix_sha256': arrays_hash(**joined),
                                'curve_rows_sha256': curve_digest.hexdigest(),
                                'generation_seconds_cumulative': time.perf_counter() - t0}
    joined = {name: np.concatenate(parts) for name, parts in pieces.items()}
    for end in BUDGETS:
        assert arrays_hash(**{name: arr[:end] for name, arr in joined.items()}) == checkpoints[end]['prefix_sha256']
    return joined, checkpoints


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def deterministic_controls():
    controls = {}
    psets = {'small_p_positive': (np.r_[np.full(20, 1e-6), np.ones(80)], 20),
             'negative': (np.full(100, 0.5), 0),
             'later_rank_despite_one_floor_hit': (np.r_[0.005, np.full(9, 0.010), np.ones(90)], 10)}
    for name, (p, expected) in psets.items():
        rejected, summary, _ = bh_rank_diagnostics(p)
        assert int(rejected.sum()) == expected
        controls[name] = {'p': p, 'expected_rejections': expected, 'observed': summary}
    later = controls['later_rank_despite_one_floor_hit']
    later.update({'declared_p_floor': 0.005, 'n_at_floor': 1,
                  'necessary_rank_from_global_floor': 5,
                  'explanation': 'Only one point is at floor. Rank 10 qualifies: p_(10)=.01 <= .1*10/100; BH rejects all ten.'})
    return controls


def run_case(seed, f_big, output):
    case = f'seed{seed}_fbig{f_big:.2f}'
    case_dir = output / case
    case_dir.mkdir()
    fb_code = int(round(f_big * 100))
    t0 = time.perf_counter()
    population = E0.gen_population(np.random.default_rng([seed, fb_code, 0]), 200, 100, f_big)
    curves = population['c']
    minimum = curves.min(axis=1)
    s0 = curves[:, E0.IDX0] - minimum
    contrast = curves.max(axis=1) - minimum
    cutoffs = np.quantile(contrast, [1 / 3, 2 / 3])
    strata = np.digitize(contrast, cutoffs)
    observation_hash = arrays_hash(c=curves, x=population['x'], y=population['y'], z0=population['z0'])
    selector_hash = arrays_hash(s0=s0, observed_stratum=strata)
    population_seconds = time.perf_counter() - t0
    observations = {'s0': s0, 'contrast': contrast, 'stratum': strata, 'cutoffs': cutoffs,
                    'truth_evaluation_only': population['tstar'], 'subset_evaluation_only': population['subset'].astype('U20'),
                    'mm_evaluation_only': population['mm'], 'x': population['x'], 'y': population['y'], 'z0': population['z0']}
    np.savez_compressed(case_dir / 'observations.npz', **observations)
    props = {sub: float(np.mean(population['subset'] == sub)) for sub in E0.SUBSETS}
    calibration, checkpoints = collect_calibration(seed, fb_code, props, cutoffs)
    np.savez_compressed(case_dir / 'calibration_full.npz', **calibration)
    rows = []
    for n_cal in BUDGETS:
        tcal = time.perf_counter()
        cal = calibrate_scores(calibration['score'][:n_cal], calibration['stratum'][:n_cal])
        calibration_seconds = time.perf_counter() - tcal
        tselect = time.perf_counter()
        selected = select_points(s0, strata, cal)
        selection_seconds = time.perf_counter() - tselect
        assert arrays_hash(s0=s0, observed_stratum=strata) == selector_hash
        evaluation = {name: evaluate_rejections(selected[name], population['tstar'], population['subset'], population['mm'])
                      for name in ('bh', 'pointwise')}
        floors = selected['p_floor']
        p = selected['p']
        floor_stats = {}
        for k, name in enumerate(E0.STRATA):
            mask = strata == k
            nused = len(cal[k]['scores'])
            cal_mask = calibration['stratum'][:n_cal] == k
            floor_stats[name] = {'n_calibration_stratum': cal[k]['n_stratum'], 'n_calibration_used': nused,
                                 'fallback': cal[k]['fallback'], 'p_floor': 1 / (nused + 1),
                                 'n_test': int(mask.sum()), 'n_test_at_floor': int((p[mask] == floors[mask]).sum()),
                                 'minimum_observed_p': float(p[mask].min()),
                                 'calibration_score_max': float(cal[k]['scores'][-1]),
                                 'calibration_misfit': int((cal_mask & (calibration['subset'][:n_cal] == 'misfit')).sum()),
                                 'calibration_back': int((cal_mask & (calibration['subset'][:n_cal] == 'twosheet_back')).sum()),
                                 'calibration_multimodal': int((cal_mask & calibration['mm'][:n_cal]).sum())}
        prefix_hash = arrays_hash(**{name: arr[:n_cal] for name, arr in calibration.items()})
        assert prefix_hash == checkpoints[n_cal]['prefix_sha256']
        metadata = {'case': case, 'seed': seed, 'f_big': f_big, 'N': len(s0), 'n_calibration': n_cal,
                    'q': Q, 'pointwise_alpha': POINT_ALPHA, 'pointwise_fdr_claim': False,
                    'interpretation': 'Development detector diagnostic; increasing n changes resolution and sampled calibration tails. Tolerance-subset rejection is not a validated FDR estimand.',
                    'generator_rng': [seed, fb_code, 0], 'calibration_rng': [seed, fb_code, 10, 1],
                    'calibration_chunk_size': CHUNK, 'calibration_regime': 'exch', 'mixture_props': props,
                    'population_realized': population['realized'], 'contrast_cutoffs': cutoffs,
                    'observation_sha256': observation_hash, 'selector_input_sha256': selector_hash,
                    'calibration_prefix': checkpoints[n_cal], 'p_sha256': arrays_hash(p=p),
                    'latency_seconds': {'population_generation_reduction_hash': population_seconds,
                                        'calibration_generation_cumulative': checkpoints[n_cal]['generation_seconds_cumulative'],
                                        'calibration_sorting': calibration_seconds, 'both_selectors_and_rank_diagnostics': selection_seconds},
                    'bh': selected['bh_summary'], 'floor_stats': floor_stats,
                    'necessary_rank_from_global_floor': int(math.ceil(len(s0) * floors.min() / Q)),
                    'necessary_rank_warning': 'A necessary rank lower bound only. Counts at the floor are not necessary for rejection; inspect every sorted-rank margin.',
                    'evaluation': evaluation}
        write_json(case_dir / f'n{n_cal}.json', metadata)
        np.savez_compressed(case_dir / f'n{n_cal}_pvalues.npz', p=p, p_floor=floors,
                            rejected_bh=selected['bh'], rejected_pointwise=selected['pointwise'], **selected['ranks'])
        cell_rows = []
        for arm in ('bh', 'pointwise'):
            for subset, values in evaluation[arm].items():
                row = {'case': case, 'seed': seed, 'f_big': f_big, 'n_calibration': n_cal,
                       'selector': arm, 'subset': subset, **values, 'max_bh_margin': selected['bh_summary']['max_margin'],
                       'bh_rejections': selected['bh_summary']['n_rejected'], 'p_floor_min': float(floors.min()),
                       'p_floor_max': float(floors.max()), 'observation_sha256': observation_hash,
                       'selector_input_sha256': selector_hash, 'calibration_prefix_sha256': prefix_hash,
                       'p_sha256': metadata['p_sha256']}
                cell_rows.append(row)
        write_csv(case_dir / f'n{n_cal}.csv', cell_rows)
        rows.extend(cell_rows)
        print(f'{case} n_cal={n_cal}: BH={selected["bh_summary"]["n_rejected"]}, point={selected["pointwise"].sum()}, '
              f'BH margin max={selected["bh_summary"]["max_margin"]:.6g}, point big detection={evaluation["pointwise"]["big"]["rejection_rate"]:.4f}', flush=True)
    return rows


def main():
    output = ROOT / 'results' / 'detector'
    if output.exists():
        raise SystemExit(f'Refusing to overwrite existing outcomes: {output}')
    output.mkdir(parents=True)
    t0 = time.perf_counter()
    write_json(output / 'run_definition.json', {
        'seeds': SEEDS, 'f_big': F_BIGS, 'N': 20000, 'calibration_budgets': BUDGETS,
        'chunk_size': CHUNK, 'q': Q, 'pointwise_alpha': POINT_ALPHA,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'test_source_sha256': hashlib.sha256((ROOT / 'tests' / 'test_detector.py').read_bytes()).hexdigest(),
        'old_source_sha256': hashlib.sha256(Path(E0.__file__).read_bytes()).hexdigest(),
        'selector_api': str(inspect.signature(select_points)),
        'saved_observations': 'Complete selector inputs and evaluation labels; full curves are regenerated from frozen E0 RNG recipe and verified by observation_sha256.',
        'timing_scope': 'Wall clock in this process, CPU cap 2; concurrent other diagnostics may contend. Not an optimized deployment benchmark.',
        'no_claims': ['full CPR-2 success', 'FDR under tolerance null', 'real scenes', 'cross-domain generalization']})
    write_json(output / 'controls.json', deterministic_controls())
    rows = []
    for seed in SEEDS:
        for f_big in F_BIGS:
            rows.extend(run_case(seed, f_big, output))
    write_csv(output / 'all_metrics.csv', rows)
    overall = [row for row in rows if row['subset'] == 'all']
    bh_overall = [row for row in overall if row['selector'] == 'bh']
    write_json(output / 'summary.json', {
        'n_populations': len(SEEDS) * len(F_BIGS), 'n_calibration_cells': len(bh_overall),
        'n_cells_with_bh_rejections': sum(row['n_rejected'] > 0 for row in bh_overall),
        'any_larger_budget_has_nonzero_bh': any(row['n_rejected'] > 0 and row['n_calibration'] > min(BUDGETS) for row in bh_overall),
        'all_32000_cells_zero_bh': all(row['n_rejected'] == 0 for row in bh_overall if row['n_calibration'] == max(BUDGETS)),
        'overall': overall, 'total_wall_seconds': time.perf_counter() - t0,
        'controls_passed': True, 'matrix_complete': True})
    print(f'Completed 18 detector cells, elapsed {time.perf_counter() - t0:.2f}s', flush=True)


if __name__ == '__main__':
    main()
