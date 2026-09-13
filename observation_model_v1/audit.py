import json
from pathlib import Path
import numpy as np
from run import rms,sha,PATCHES,RUNS
from source_filter import reference_id
from normal_followup import estimate_normal

FIRST=RUNS/'observation-model-v1-7riqb9c4'
SECOND=RUNS/'observation-normal-v1-svrph48g'
THIRD=RUNS/'observation-plane-v1-izu7fff7'

def main():
    result=json.loads((THIRD/'RESULTS.json').read_text());prior=json.loads((FIRST/'RESULTS.json').read_text())
    refs={p.stem:np.load(p)['xyz_world'] for p in PATCHES.glob('*/*.npz')}
    delta=0.;count=0
    for row in result['rows']:
        with np.load(THIRD/f"{row['case']}.npz") as a:out=a[row['method']]
        with np.load(FIRST/f"{row['case']}.npz") as a:q=a['input'];scan=a['scan_id']
        delta=max(delta,abs(rms(out,refs[row['patch']])-row['recovery_rms_mm']))
        anchor=scan==reference_id(scan);np.testing.assert_array_equal(out[anchor],q[anchor])
        assert np.isfinite(out).all()
        move=out[~anchor]-q[~anchor]
        np.testing.assert_allclose(move,np.tile(move[0],(len(move),1)),atol=1e-12)
        count+=1
    # Check the unchanged 2x2 control exactly reproduces the preceding scalar model.
    for d in prior['diagnostics']:
        with np.load(SECOND/f"{d['case']}.npz") as a,np.load(THIRD/f"{d['case']}.npz") as b:
            for name in ('normal_all','normal_confirmed'):
                np.testing.assert_array_equal(a[name],b[f'valid0_local0_{name}'])
    d=prior['diagnostics'][0]
    with np.load(FIRST/f"{d['case']}.npz") as a:q=a['input'];scan=a['scan_id']
    replay,_=estimate_normal(q,scan,True,True)
    with np.load(THIRD/f"{d['case']}.npz") as a:
        for name,out in replay.items():np.testing.assert_array_equal(out,a[f'valid1_local1_{name}'])
    for path,h in prior['input_hashes'].items():assert sha(Path(path))==h
    rng=np.random.default_rng(916751)
    for _ in range(20):
        normals=rng.normal(size=(100,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
        _,vectors=np.linalg.eigh(normals.T@normals);direction=vectors[:,-1]
        assert np.mean((normals@direction)**2)>=1/3-1e-12
    report=dict(outputs_recomputed=count,max_metric_difference_mm=delta,
        unchanged_reference_station=True,station_shape_preserved=True,control_replay_exact=True,
        candidate_replay_exact=True,initial_protected_hashes_unchanged=True,normal_direction_condition_checks=20)
    (THIRD/'AUDIT.json').write_text(json.dumps(report,indent=2));print(report)

if __name__=='__main__':main()
