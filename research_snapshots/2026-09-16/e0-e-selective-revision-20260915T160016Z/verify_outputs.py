"""Independently recompute saved numeric outputs; no model or generation rerun."""
import json
import struct

import numpy as np
from scipy.spatial import cKDTree

from experiment_support import ROOT, sha, snapshot, write_json


def read_ply(path):
    with path.open('rb') as f:
        n=None
        while True:
            line=f.readline()
            if not line:raise AssertionError('missing PLY header')
            if line.startswith(b'element vertex '):n=int(line.split()[-1])
            if line==b'end_header\n':break
        a=np.frombuffer(f.read(),dtype='<f4').reshape(n,3)
    return a


def main():
    data=json.loads((ROOT/'results/RESULTS.json').read_text())
    lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text())
    after=snapshot()
    assert lock['old_files']==after,'old-file snapshot mismatch'
    changed=[p for p,h in lock['implementation'].items() if sha(ROOT/p)!=h]
    assert not changed,changed
    max_delta=0.;max_ply_delta=0.;count=0;geometry_count=0
    false_veto_by_stratum={str(k):dict(proposed=0,blocked=0,available_gain=0.,lost_gain=0.) for k in range(3)}
    for run in data['runs']:
        folder=ROOT/'results/cases'/run['key']
        with np.load(folder/'outputs.npz') as a:
            truth=a['evaluation_truth_delta'];xyz0=a['xyz0']
            target=xyz0.copy();target[:,2]+=truth
            for arm,saved in run['metrics'].items():
                theta=a['out_'+arm]
                e=np.abs(theta-truth);b=np.abs(truth)
                values=dict(mae=float(e.mean()),rmse=float(np.sqrt((e**2).mean())),
                            delta_mae=float((e-b).mean()),harm_sum=float(np.maximum(e-b,0).sum()),
                            gain_sum=float(np.maximum(b-e,0).sum()))
                for k,v in values.items():max_delta=max(max_delta,abs(v-saved['groups']['ALL'][k]))
                assert int(np.sum(e>b+1e-9))==saved['groups']['ALL']['harm_count']
                assert int(np.count_nonzero(theta))==saved['groups']['ALL']['moved']
                output=xyz0.copy();output[:,2]+=theta
                d1=cKDTree(target).query(output)[0]
                d2=cKDTree(output).query(target)[0]
                gs=dict(nn_accuracy=float(d1.mean()),nn_completeness=float(d2.mean()),
                        nn_symmetric=float((d1.mean()+d2.mean())/2),
                        precision_1mm=float((d1<=1).mean()),recall_1mm=float((d2<=1).mean()))
                for k,v in gs.items():max_delta=max(max_delta,abs(v-saved['geometry'][k]))
                actual=read_ply(folder/(arm+'.ply'))
                max_ply_delta=max(max_ply_delta,float(np.max(np.abs(actual.astype(float)-output))))
                np.testing.assert_array_equal(actual,output.astype('<f4'))
                count+=1;geometry_count+=1
            # No veto may introduce a new per-point coordinate harm.
            p=a['out_test_projection'];v=a['out_test_veto_projection']
            assert np.all(np.maximum(np.abs(v-truth)-np.abs(truth),0)
                          <=np.maximum(np.abs(p-truth)-np.abs(truth),0)+1e-12)
            if (run['phase']=='locked_same_family' and run['fbig']==.1
                    and run['calibration']=='exch' and run['alpha']==.2):
                for k in range(3):
                    m=(a['evaluation_subset']=='big') & ~a['evaluation_mm'] & (a['observation_strata']==k)
                    b=m & (p!=0) & (v==0)
                    gain=np.maximum(np.abs(truth)-np.abs(p-truth),0)
                    row=false_veto_by_stratum[str(k)]
                    row['proposed']+=int(np.count_nonzero(m & (p!=0)))
                    row['blocked']+=int(b.sum())
                    row['available_gain']+=float(gain[m].sum())
                    row['lost_gain']+=float(gain[b].sum())
    for case in data['crossed']:
        folder=ROOT/'results/crossed'/case['key']
        with np.load(folder/'outputs.npz') as a:
            for world in ('real_back','ghost'):
                truth=a['evaluation_truth_'+world]
                for arm,blk in case['worlds'][world].items():
                    got=float(np.abs(a['out_'+arm]-truth).mean())
                    max_delta=max(max_delta,abs(got-blk['groups']['ALL']['mae']))
    assert max_delta<1e-10,max_delta
    result=dict(main_arm_blocks=count,geometry_blocks=geometry_count,
                crossed_world_arm_blocks=len(data['crossed'])*2*9,
                metrics_max_abs_difference=max_delta,
                float32_ply_max_coordinate_difference_mm=max_ply_delta,
                all_main_ply_match_float32_arrays=True,
                original_files_unchanged=len(after),locked_sources_unchanged=True,
                posthoc_big_single_veto_strata=false_veto_by_stratum)
    write_json(ROOT/'results/INDEPENDENT_RECHECK.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
