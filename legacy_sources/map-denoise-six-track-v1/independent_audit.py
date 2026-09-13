"""Recalculate reported T2/T5 measurements directly from saved geometry."""
import json
from pathlib import Path
import numpy as np

BASE = Path('/srv/slam-research/grf/map-denoise/runs/six-track-v1-20260911-1600')


def audit_t2(run):
    rows = json.loads((run/'raw_results.json').read_text())
    maximum = 0.
    for r in rows:
        name = f"{r['case']}_s{r['seed']:02d}"
        inp = np.load(run/'inputs'/f'{name}.npz')
        gt = np.load(run/'inputs'/f'{name}_EVAL_TRUTH.npz')
        out = np.load(run/'outputs'/f"{name}__{r['method']}.npz")
        target = inp['roi']==0
        mae = float(np.abs(out['xyz_m'][target,2]-gt['clean_xyz_m'][target,2]).mean()*1000)
        err = abs(mae-r['normal_mae_mm']); maximum=max(maximum,err)
        assert err < 1e-10
        bias = float(np.sqrt(np.mean((out['bias_mm']-gt['bias_mm'])**2)))
        assert abs(bias-r['bias_rmse_mm']) < 1e-10
        if r['status']=='WAIT': assert np.array_equal(inp['xyz_m'],out['xyz_m'])
    seeds=sorted({r['seed'] for r in rows})
    paired = []
    for s in seeds:
        a=np.load(run/'inputs'/f'single_ghost_no_anchor_s{s:02d}.npz')
        b=np.load(run/'inputs'/f'thin_segregated_no_anchor_s{s:02d}.npz')
        paired.append(all(np.array_equal(a[k],b[k]) for k in a.files))
    assert all(paired)
    return {'run':str(run),'outputs_recomputed':len(rows),'max_metric_difference':maximum,
            'same_input_world_pairs':len(paired),'wait_outputs_unchanged':True}


def audit_t5():
    run=BASE/'t5'; data=json.loads((run/'results.json').read_text())
    max_delta=0.; mismatches=0
    for row in data['comparisons']:
        name, arm = row['case'], row['arm']
        a=np.load(run/f'{name}-reference.npy');b=np.load(run/f'{name}-{arm}.npy')
        delta=float(np.linalg.norm(a-b,axis=1).max()) if len(a) else 0.
        assert abs(delta-row['max_output_delta_m']) < 1e-16
        max_delta=max(max_delta,delta)
        aa=np.load(run/f'{name}-reference-branches.npz');bb=np.load(run/f'{name}-{arm}-branches.npz')
        assert set(aa.files)==set(bb.files)
        for k in aa.files: mismatches+=int(np.sum(aa[k]!=bb[k]))
    assert mismatches==0
    return {'cases':data['case_count'],'comparisons':len(data['comparisons']),
            'max_output_delta_m':max_delta,'branch_mismatches':mismatches}


if __name__=='__main__':
    results={'t2_initial':audit_t2(BASE/'t1_t2/run-initial-01'),
             't2_newseed':audit_t2(BASE/'t1_t2/root-newseed-check-01'),
             't5':audit_t5()}
    path=Path(__file__).parent/'INDEPENDENT_AUDIT.json'
    with path.open('x') as f:json.dump(results,f,indent=2)
    print(json.dumps(results,indent=2))
