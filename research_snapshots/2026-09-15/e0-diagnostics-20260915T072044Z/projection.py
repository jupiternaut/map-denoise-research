"""Fixed-support diagnostic of the original E0 projection (not deployment)."""
import argparse
import csv
import gzip
import hashlib
import json
import time

import numpy as np

from common import ROOT, OLD, load_old, write_json


def candidates(old, s, m, strata, calibration):
    """Observation-only function. No target coordinates or labels accepted."""
    dec = old.decide(s, m, strata, calibration)
    mask = (m <= dec['qabs_i']) & (dec['n_comp'] == 1) & ~dec['L'][:, old.IDX0]
    end = np.where(dec['a_near'] > old.IDX0,
                   old.TGRID[dec['a_near']], old.TGRID[dec['b_near']])
    argmin = old.TGRID[np.argmin(s, axis=1)]
    outputs = {}
    for name, support in [('original_bh', dec['out'] == 'MOVE'), ('eligible_no_bh', mask)]:
        outputs[name] = dict(mask=support,
                             identity=np.zeros(len(m)),
                             projection=np.where(support, end, 0.0),
                             argmin=np.where(support, argmin, 0.0))
    return dec, outputs


def prepared_population(old, seed, fbig):
    code = int(round(fbig * 100))
    pop = old.gen_population(np.random.default_rng([seed, code, 0]), 200, 100, fbig)
    s, m, argmin, contrast = old.scores(pop['c'])
    cuts = np.quantile(contrast, [1 / 3, 2 / 3])
    strata = np.digitize(contrast, cuts)
    props = {sub: float(np.mean(pop['subset'] == sub)) for sub in old.SUBSETS}
    return pop, s, m, strata, cuts, props


def calibration_for(old, seed, fbig, kind, cuts, props, alpha):
    code = int(round(fbig * 100))
    ci = list(old.CALIBS).index(kind)
    K = old.gen_calset(np.random.default_rng([seed, code, 10 + ci, 1]), 2000, kind, props)
    s, m, _, contrast = old.scores(K['c'])
    strata = np.digitize(contrast, cuts)
    calibration_truth_scores = old.interp_rows(s, K['tstar'])
    return old.calibrate(calibration_truth_scores, m, strata, alpha)


def audit_old(old):
    """Only compare regenerated observations/decisions with exposed seed-1 output."""
    checks = []
    for fbig in [.01, .10]:
        pop, s, m, strata, cuts, props = prepared_population(old, 1, fbig)
        for kind in old.CALIBS:
            for alpha in old.ALPHA_TGTS:
                cal = calibration_for(old, 1, fbig, kind, cuts, props, alpha)
                dec, outputs = candidates(old, s, m, strata, cal)
                key = old.scenario_key(fbig, kind, alpha)
                path = OLD / 'results' / 'decisions' / (key + '_seed1.csv.gz')
                with gzip.open(path, 'rt') as f:
                    records = list(csv.DictReader(f))
                max_theta = max(abs(float(row['theta_CPR2']) - dec['theta'][i])
                                for i, row in enumerate(records))
                max_p = max(abs(float(row['p']) - dec['p'][i]) for i, row in enumerate(records))
                flags_equal = all(row['out'] == dec['out'][i] and row['res'] == dec['res'][i]
                                  and int(row['rej']) == dec['rej'][i]
                                  for i, row in enumerate(records))
                # CSV intentionally rounds p to six decimal places and theta to three.
                ok = max_theta <= 5.01e-4 and max_p <= 5.01e-7 and flags_equal
                checks.append(dict(scenario=key, source=str(path), n=len(records),
                                   max_theta_difference=max_theta, max_p_difference=max_p,
                                   flags_equal=flags_equal, pass_check=ok))
                if not ok:
                    write_json(ROOT / 'results/projection/REPRODUCTION_FAILURE.json', checks)
                    raise AssertionError('old-source reproduction failed: ' + key)
    return checks


def summarize_case(old, pop, s, strata, dec, outputs):
    truth = pop['tstar']  # evaluation channel starts here
    big = pop['subset'] == 'big'
    covered = old.interp_rows(s, truth) <= dec['qtgt_i']
    selections = {'ALL': np.ones(len(truth), bool)}
    selections.update({str(k): pop['subset'] == k for k in old.SUBSETS})
    selections.update({'stratum_' + name: strata == k for k, name in enumerate(old.STRATA)})
    result = {}
    for support_name, arms in outputs.items():
        support = arms['mask']
        blocks = {}
        for name in ('identity', 'projection', 'argmin'):
            theta = arms[name]
            metrics = {sub: old.arm_block(theta, truth, sel, big) for sub, sel in selections.items()}
            metrics['covered_moves'] = old.dmg_count(theta, truth, support & covered)
            metrics['uncovered_moves'] = old.dmg_count(theta, truth, support & ~covered)
            blocks[name] = metrics
        result[support_name] = dict(eligible_count=int(support.sum()),
                                    covered_count=int((support & covered).sum()),
                                    same_actual_movement_mask=bool(np.array_equal(arms['projection'] != 0,
                                                                                  arms['argmin'] != 0)),
                                    metrics=blocks)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    old = load_old()
    dest = ROOT / 'results/projection'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'RESULTS.json').exists():
        raise SystemExit('RESULTS.json exists: choose a new experiment rather than overwrite it')
    start = time.perf_counter()
    checks = audit_old(old)
    write_json(dest / 'REPRODUCTION.json', checks)
    print('old E0 seed-1 reproduction:', sum(c['pass_check'] for c in checks), '/', len(checks), flush=True)
    if args.audit_only:
        return
    runs = []
    for seed in (101, 102, 103):
        for fbig in (.01, .10):
            pop, s, m, strata, cuts, props = prepared_population(old, seed, fbig)
            observation_hash = hashlib.sha256(s.tobytes() + m.tobytes() + strata.tobytes()).hexdigest()
            for kind in old.CALIBS:
                for alpha in old.ALPHA_TGTS:
                    cal = calibration_for(old, seed, fbig, kind, cuts, props, alpha)
                    dec, outputs = candidates(old, s, m, strata, cal)
                    key = old.scenario_key(fbig, kind, alpha) + '_seed' + str(seed)
                    metrics = summarize_case(old, pop, s, strata, dec, outputs)
                    run = dict(key=key, seed=seed, fbig=fbig, calibration=kind, alpha=alpha,
                               n=len(m), observation_hash=observation_hash, diagnostics=metrics)
                    runs.append(run)
                    # Targets saved solely to reproduce evaluation, separate from selector API.
                    np.savez_compressed(dest / (key + '.npz'),
                                        x=pop['x'], y=pop['y'], z0=pop['z0'],
                                        evaluation_truth=pop['tstar'], evaluation_subset=pop['subset'].astype(str),
                                        observation_strata=strata, p=dec['p'],
                                        covered_evaluation=old.interp_rows(s, pop['tstar']) <= dec['qtgt_i'],
                                        bh_mask=outputs['original_bh']['mask'],
                                        eligible_mask=outputs['eligible_no_bh']['mask'],
                                        projection=outputs['eligible_no_bh']['projection'],
                                        argmin=outputs['eligible_no_bh']['argmin'])
                    write_json(dest / (key + '.json'), run)
                    a = metrics['eligible_no_bh']
                    print(key, 'eligible=', a['eligible_count'],
                          'mae I/P/A=', [round(a['metrics'][x]['ALL']['e_after_mean'], 5)
                                         for x in ('identity', 'projection', 'argmin')], flush=True)
    write_json(dest / 'RESULTS.json', dict(scope='development module diagnostic; BH bypass not deployment',
                                         reproduction=checks, runs=runs,
                                         wall_seconds=time.perf_counter() - start))
    with (dest / 'METRICS.csv').open('w', newline='') as f:
        fields = ['key', 'seed', 'fbig', 'calibration', 'alpha', 'support', 'arm', 'subset',
                  'eligible_count', 'n_moved', 'n_dmg', 'dmg_rate_N', 'dmg_rate_moved',
                  'dmg_mag_mean_moved', 'dmg_mag_p95_moved', 'e_after_mean', 'e_after_rmse', 'big_repair_cov']
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            for support, blk in run['diagnostics'].items():
                for arm, metrics in blk['metrics'].items():
                    for subset, metric in metrics.items():
                        if subset.endswith('_moves'):
                            continue
                        row = {k: run[k] for k in ('key', 'seed', 'fbig', 'calibration', 'alpha')}
                        row.update(support=support, arm=arm, subset=subset, eligible_count=blk['eligible_count'])
                        row.update({k: metric[k] for k in fields if k in metric})
                        writer.writerow(row)


if __name__ == '__main__':
    main()
