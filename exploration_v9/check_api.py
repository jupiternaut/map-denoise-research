"""Raw legal-input integration check against sealed cached-state outputs."""
import json,sys,socket
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from exploration_v7.algorithm.reassociation import freeze,fit_frozen
from exploration_v9.unmix import unmix_frozen
from exploration_v9.crossfit import candidate_search,project_candidate
RUNS=Path('/srv/slam-research/grf/map-denoise/runs/multiscan-pilot-v1')
UNMIX=RUNS/'selection-unmix-v9-qe05wnli';CROSS=RUNS/'crossfit-actions-v9-dcz583qe'
def read(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def main():
    assert socket.gethostname()=='liekkas';result=[]
    for case in ('ghost_s912101_b0','dual_g2_s912101_b0'):
        inputs=read(RUNS/'repair-v2-oyuie4pl/synthetics/identifiable'/(case+'.npz'))
        before=inputs['xyz_world'].copy();state=freeze(inputs['xyz_world'],inputs['scan_id'],1.)
        _,_,candidate=fit_frozen(state,variant='reassociate',budget=6,sharing='independent')
        for mode in ('offset','affine'):
            out,_,art=unmix_frozen(state,candidate,1.,mode)
            saved=read(UNMIX/'outputs'/f'{case}__b6_{mode}.npz')
            error=float(np.max(abs(out-saved['xyz_world']))*1000.)
            assert error<1e-8
            np.testing.assert_array_equal(art['support_mask'],saved['support_mask'])
            result.append(dict(case=case,method=mode,max_difference_mm=error))
        _,_,seed=fit_frozen(state,variant='original',budget=0,sharing='independent')
        cross,_=candidate_search(state,seed,1.,budget=6,folds=2)
        out,_,_=project_candidate(state,cross)
        saved=read(CROSS/'outputs'/f'{case}__crossfit_b6_candidate.npz')
        error=float(np.max(abs(out-saved['xyz_world']))*1000.);assert error<1e-8
        result.append(dict(case=case,method='crossfit_candidate',max_difference_mm=error))
        np.testing.assert_array_equal(before,inputs['xyz_world'])
    obj=dict(status='PASS',comparisons=result,truth_read=False,scope='two public inputs, full API versus cached state; no benchmark claim')
    with (UNMIX/'API_CHECK.json').open('x') as f:json.dump(obj,f,indent=2)
    print(json.dumps(obj,indent=2))
if __name__=='__main__':main()
