"""Run the frozen E0-E protocol; never overwrite an existing result set."""
import csv
import json
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiment_support import (ROOT, DIAG, OLD, arrays_hash, calibrate_with_roughness,
    evaluate, linear_projection, plain, prepared_population, sha, snapshot,
    veto_account, write_json, write_ply)
from selective_operator import choose_outputs, curve_features


def run_case(dest, phase, seed, fbig, kind, alpha, pop, strata, cuts, props, feat):
    key = f'{phase}_seed{seed}_fbig{fbig:.2f}_{kind}_a{alpha:.1f}'
    cal, sigma, cal_hash = calibrate_with_roughness(seed, fbig, kind, cuts, props, alpha)
    outputs, obs = choose_outputs(OLD, linear_projection, pop['c'], strata, cal, sigma, feat)
    legacy_check = None
    if phase == 'development':
        old_key = OLD.scenario_key(fbig, kind, alpha) + '_seed' + str(seed)
        with np.load(DIAG / 'results/projection' / (old_key + '.npz')) as frozen:
            np.testing.assert_array_equal(obs['eligible'], frozen['eligible_mask'])
            np.testing.assert_array_equal(obs['bh'], frozen['bh_mask'])
        with np.load(DIAG / 'results/boundary' / (old_key + '.npz')) as frozen:
            np.testing.assert_array_equal(outputs['no_bh_projection'], frozen['linear_projection'])
        legacy_check = 'exact masks and continuous output match'
    xyz0 = np.column_stack([pop['x'], pop['y'], pop['z0']])
    # Everything above this boundary is independent of target evaluation truth.
    metrics = {name: evaluate(theta, xyz0, pop['tstar'], pop['subset'], pop['mm'])
               for name, theta in outputs.items()}
    ledger = {name: veto_account(outputs['test_projection'], outputs[name], pop['tstar'],
                                pop['subset'], pop['mm'], strata)
              for name in ('test_veto_projection', 'test_margin_projection')}
    s = pop['c'] - pop['c'].min(axis=1, keepdims=True)
    covered = OLD.interp_rows(s, pop['tstar']) <= obs['q_tgt']
    covered_harm = {}
    for name in ('original_bh_projection', 'no_bh_projection', 'test_projection',
                 'test_veto_projection', 'test_margin_projection'):
        harms = np.abs(outputs[name]-pop['tstar']) > np.abs(pop['tstar']) + 1e-9
        covered_harm[name] = int((harms & covered).sum())
        assert covered_harm[name] == 0, (key, name, covered_harm[name])
    # Metamorphic check: the evaluator cannot change either selector inputs or outputs.
    public_hash = arrays_hash(pop['c'], strata)
    decision_hash = arrays_hash(*outputs.values())
    observation = dict(public_hash=public_hash, calibration_hash=cal_hash,
        strata_cuts=cuts, sigma_hf_by_stratum=sigma,
        pointwise_reject_rate=float((obs['p'] <= .1).mean()),
        base_pointwise_reject_rate=float((obs['p'][pop['subset']=='base'] <= .1).mean()),
        test_candidates=int(obs['test'].sum()), bh_candidates=int(obs['bh'].sum()),
        no_bh_candidates=int(obs['eligible'].sum()), veto_all=int(obs['veto'].sum()),
        acquire_all=int(obs['research_acquire'].sum()),
        primary_boundary_count=int(obs['primary_boundary'].sum()),
        width_quantiles=np.quantile(obs['width_mm'], [.1, .5, .9]),
        threshold_quantiles=np.quantile(obs['threshold'], [.1, .5, .9]))
    result = dict(key=key, phase=phase, seed=seed, fbig=fbig, calibration=kind, alpha=alpha,
                  n=len(xyz0), legacy_replay=legacy_check, observation=observation,
                  decision_hash=decision_hash, covered_harm=covered_harm,
                  metrics=metrics, veto_ledger=ledger)
    folder = dest / 'cases' / key
    folder.mkdir(parents=True)
    write_json(folder / 'metrics.json', result)
    write_json(folder / 'calibration.json', dict(calibration=cal, sigma_hf=sigma))
    np.savez_compressed(folder / 'outputs.npz', xyz0=xyz0,
        evaluation_truth_delta=pop['tstar'], evaluation_subset=pop['subset'].astype(str),
        evaluation_mm=pop['mm'], observation_strata=strata, covered_evaluation=covered,
        **{'out_' + k: v for k, v in outputs.items()}, **{'obs_' + k: v for k, v in obs.items()})
    for name, theta in outputs.items():
        xyz = xyz0.copy()
        xyz[:, 2] += theta
        write_ply(folder / (name + '.ply'), xyz)
    truth_xyz = xyz0.copy()
    truth_xyz[:, 2] += pop['tstar']
    write_ply(folder / 'evaluation_reference.ply', truth_xyz)
    print(key, 'ΔMAE TEST/VETO/MARGIN=',
          [round(metrics[n]['groups']['ALL']['delta_mae'], 5) for n in
           ('test_projection','test_veto_projection','test_margin_projection')], flush=True)
    return result


def crossed_cases(dest):
    pop, _, _, _, cuts, props = prepared_population(OLD, 101, .10)
    cal, sigma, _ = calibrate_with_roughness(101, .10, 'exch', cuts, props, .2)
    del pop
    results = []
    for seed in (501, 502):
        for amplitude in (0., .15, .30, .60):
            for noise in (0., .03):
                n = 512
                rng = np.random.default_rng([seed, int(amplitude*100), int(noise*100)])
                A = rng.uniform(.5, .8, n)
                main = np.exp(-.5 * (OLD.TGRID[None, :]-6.)**2)
                second = amplitude * np.exp(-.5 * OLD.TGRID[None, :]**2)
                c = A[:, None] * (1.-np.maximum(main, second))
                if noise:
                    c += noise * OLD.smooth_noise(rng, n)
                strata = np.digitize(np.ptp(c, axis=1), cuts)
                outputs, obs = choose_outputs(OLD, linear_projection, c, strata, cal, sigma)
                xyz0 = np.column_stack([(np.arange(n) % 32)*.8,
                                        (np.arange(n)//32)*.8, np.zeros(n)])
                key = f'seed{seed}_weak{amplitude:.2f}_noise{noise:.2f}'
                worlds = {}
                for world, delta in [('real_back', 0.), ('ghost', 6.)]:
                    truth = np.full(n, delta)
                    subset = np.full(n, world)
                    worlds[world] = {name: evaluate(theta, xyz0, truth, subset, np.full(n, amplitude>0))
                                     for name, theta in outputs.items()}
                # The full exposed input and every output are shared, not only coordinates.
                paired, _ = choose_outputs(OLD, linear_projection, c.copy(), strata.copy(), cal, sigma)
                for name in outputs:
                    np.testing.assert_array_equal(outputs[name], paired[name])
                folder = dest / 'crossed' / key
                folder.mkdir(parents=True)
                np.savez_compressed(folder / 'outputs.npz', curves=c, grid=OLD.TGRID,
                    xyz0=xyz0, evaluation_truth_real_back=np.zeros(n),
                    evaluation_truth_ghost=np.full(n, 6.),
                    **{'out_'+k:v for k,v in outputs.items()},
                    **{'obs_'+k:v for k,v in obs.items()})
                for name, theta in outputs.items():
                    xyz = xyz0.copy(); xyz[:,2] += theta
                    write_ply(folder / (name+'.ply'), xyz)
                row = dict(key=key, seed=seed, amplitude=amplitude, noise=noise, n=n,
                    observation_hash=arrays_hash(c, xyz0, strata), paired_outputs_equal=True,
                    veto_rate=float(obs['veto'].mean()), test_rate=float(obs['test'].mean()), worlds=worlds)
                write_json(folder/'metrics.json', row)
                results.append(row)
                print('crossed', key, 'veto=', round(row['veto_rate'],4), flush=True)
    return results


def aggregate(runs):
    groups = {}
    for r in runs:
        key = (r['phase'], r['fbig'], r['calibration'], r['alpha'])
        groups.setdefault(key, []).append(r)
    summary = []
    for (phase, fbig, kind, alpha), rows in groups.items():
        arms = {}
        for name in rows[0]['metrics']:
            arms[name] = {k: float(np.mean([r['metrics'][name]['groups']['ALL'][k] for r in rows]))
                          for k in ('mae','delta_mae','move_rate','harm_rate','harm_sum','gain_sum')}
            arms[name]['seed_delta_mae'] = [r['metrics'][name]['groups']['ALL']['delta_mae'] for r in rows]
            arms[name]['nn_symmetric'] = float(np.mean([r['metrics'][name]['geometry']['nn_symmetric'] for r in rows]))
            arms[name]['recall_1mm'] = float(np.mean([r['metrics'][name]['geometry']['recall_1mm'] for r in rows]))
        back_before = sum(r['metrics']['no_bh_projection']['groups']['twosheet_back']['harm_sum'] for r in rows)
        back_after = sum(r['metrics']['test_veto_projection']['groups']['twosheet_back']['harm_sum'] for r in rows)
        big_before = sum(r['metrics']['test_projection']['groups']['big']['gain_sum'] for r in rows)
        big_after = sum(r['metrics']['test_veto_projection']['groups']['big']['gain_sum'] for r in rows)
        summary.append(dict(phase=phase, fbig=fbig, calibration=kind, alpha=alpha, seeds=[r['seed'] for r in rows],
            arms=arms, back_harm_ratio=back_after/back_before if back_before else None,
            big_gain_retained=big_after/big_before if big_before else None))
    gates = []
    for row in summary:
        if row['calibration']=='exch' and row['alpha']==.2:
            delta = row['arms']['test_veto_projection']['delta_mae']
            gates.append(dict(phase=row['phase'], fbig=row['fbig'],
                noninferior=delta<=0, back_damage_tenth=row['back_harm_ratio']<=.1,
                big_gain_retained_80=row['big_gain_retained']>=.8,
                peer_prediction=delta<=(-.15 if row['fbig']==.1 else .03)))
    return dict(groups=summary, continuation_checks=gates,
                proceed_to_real=all(g['noninferior'] and g['back_damage_tenth'] and g['big_gain_retained_80'] for g in gates))


def export_csv(dest, runs):
    rows = []
    for run in runs:
        for arm, block in run['metrics'].items():
            for group, metric in block['groups'].items():
                rows.append(dict(key=run['key'], phase=run['phase'], seed=run['seed'],
                    fbig=run['fbig'], calibration=run['calibration'], alpha=run['alpha'],
                    arm=arm, group=group, **metric))
    with (dest/'METRICS.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main():
    dest = ROOT / 'results'
    if dest.exists():
        raise SystemExit('results already exists; refusing overwrite')
    if platform.node() != 'liekkas':
        raise SystemExit('wrong host')
    before = snapshot()
    source_lock = {str(p.relative_to(ROOT)):sha(p) for p in ROOT.rglob('*')
                   if p.is_file() and p.suffix in ('.py','.md')}
    write_json(ROOT/'SOURCE_LOCK.json', dict(old_files=before, implementation=source_lock))
    dest.mkdir()
    test = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-v'],
                          cwd=ROOT, capture_output=True, text=True)
    (dest/'TEST_LOG.txt').write_text(test.stdout+test.stderr)
    if test.returncode:
        raise RuntimeError('tests failed before experiments; see TEST_LOG.txt')
    start = time.perf_counter()
    runs=[]
    for phase, seeds in [('development',(101,102,103)), ('locked_same_family',(201,202,203))]:
        for seed in seeds:
            for fbig in (.01,.10):
                pop,s,m,strata,cuts,props = prepared_population(OLD,seed,fbig)
                feat = curve_features(pop['c'],OLD.TGRID)
                for kind in ('exch','sfm'):
                    for alpha in (.2,.5):
                        runs.append(run_case(dest,phase,seed,fbig,kind,alpha,pop,strata,cuts,props,feat))
                del pop,s,m,feat
    crossed=crossed_cases(dest)
    summary=aggregate(runs)
    elapsed=time.perf_counter()-start
    after=snapshot()
    changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]
    if changed:
        write_json(dest/'OLD_FILE_CHANGE.json',changed)
        raise AssertionError('old files changed')
    write_json(dest/'RESULTS.json',dict(runs=runs,crossed=crossed,summary=summary,
        wall_seconds=elapsed,process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        old_files_checked=len(before),old_files_changed=changed,
        scope='development + locked same-family replication, not independent real validation'))
    write_json(dest/'SUMMARY.json',summary)
    export_csv(dest,runs)
    write_json(dest/'RUN_VERIFICATION.json',dict(tests_returncode=test.returncode,
        legacy_cases_exact=24, covered_harm_zero=True, paired_cases_equal=len(crossed),
        old_files_checked=len(before),old_files_changed=changed,wall_seconds=elapsed,
        python=sys.version,numpy=np.__version__))
    print('DONE',len(runs),'main configurations;',len(crossed),'paired evidence sets;',
          round(elapsed,2),'s; real continuation=',summary['proceed_to_real'],flush=True)


if __name__=='__main__':
    main()
