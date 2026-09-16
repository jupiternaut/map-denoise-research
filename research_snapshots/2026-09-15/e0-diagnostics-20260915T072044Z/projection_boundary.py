"""Numerical repair diagnostic: project onto the actually scored linear level set."""
import csv
import time

import numpy as np

from common import ROOT, load_old, write_json
from projection import candidates, prepared_population, calibration_for


def linear_projection(old, s, dec, support):
    n = len(s)
    row = np.arange(n)
    a, b, q = dec['a_near'], dec['b_near'], dec['qtgt_i']
    left = old.TGRID[a].copy()
    right = old.TGRID[b].copy()
    sel = support & (a > 0)
    r, j = row[sel], a[sel]
    left[sel] = old.TGRID[j-1] + old.T_STEP * (q[sel] - s[r, j-1]) / (s[r, j] - s[r, j-1])
    sel = support & (b < len(old.TGRID)-1)
    r, j = row[sel], b[sel]
    right[sel] = old.TGRID[j] + old.T_STEP * (q[sel] - s[r, j]) / (s[r, j+1] - s[r, j])
    result = np.zeros(n)
    result[support] = np.clip(0., left[support], right[support])
    return result


def main():
    old = load_old()
    dest = ROOT / 'results/boundary'
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'RESULTS.json').exists():
        raise SystemExit('follow-up results already exist; refusing overwrite')
    start = time.perf_counter()
    runs = []
    for seed in (101,102,103):
        for fbig in (.01,.10):
            pop,s,m,strata,cuts,props = prepared_population(old,seed,fbig)
            for kind in old.CALIBS:
                for alpha in old.ALPHA_TGTS:
                    cal=calibration_for(old,seed,fbig,kind,cuts,props,alpha)
                    dec,outputs=candidates(old,s,m,strata,cal)
                    key=old.scenario_key(fbig,kind,alpha)+'_seed'+str(seed)
                    saved=np.load(ROOT/'results/projection'/(key+'.npz'))
                    support=outputs['eligible_no_bh']['mask']
                    grid=outputs['eligible_no_bh']['projection']
                    np.testing.assert_array_equal(grid,saved['projection'])
                    np.testing.assert_array_equal(support,saved['eligible_mask'])
                    corrected=linear_projection(old,s,dec,support)
                    covered=old.interp_rows(s,pop['tstar'])<=dec['qtgt_i']
                    big=pop['subset']=='big'
                    allmask=np.ones(len(m),bool)
                    row=dict(key=key,seed=seed,fbig=fbig,calibration=kind,alpha=alpha,
                             eligible_count=int(support.sum()),
                             max_movement_difference=float(np.max(np.abs(grid-corrected))),
                             grid_covered=old.dmg_count(grid,pop['tstar'],support&covered),
                             linear_covered=old.dmg_count(corrected,pop['tstar'],support&covered),
                             grid=old.arm_block(grid,pop['tstar'],allmask,big),
                             linear=old.arm_block(corrected,pop['tstar'],allmask,big))
                    runs.append(row)
                    np.savez_compressed(dest/(key+'.npz'),grid_projection=grid,linear_projection=corrected,
                                        eligible_mask=support,evaluation_truth=pop['tstar'],covered_evaluation=covered)
                    print(key,'covered harms grid/linear',row['grid_covered']['n_dmg'],row['linear_covered']['n_dmg'],flush=True)
    write_json(dest/'RESULTS.json',dict(scope='post-exposure numerical diagnostic, same cases',runs=runs,
                                       wall_seconds=time.perf_counter()-start))
    with (dest/'METRICS.csv').open('w',newline='') as f:
        fields=['key','eligible_count','grid_covered_harm','linear_covered_harm','grid_mae','linear_mae','max_movement_difference']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in runs:
            w.writerow(dict(key=r['key'],eligible_count=r['eligible_count'],grid_covered_harm=r['grid_covered']['n_dmg'],
                            linear_covered_harm=r['linear_covered']['n_dmg'],grid_mae=r['grid']['e_after_mean'],
                            linear_mae=r['linear']['e_after_mean'],max_movement_difference=r['max_movement_difference']))


if __name__=='__main__':
    main()
