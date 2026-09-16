"""Independent arithmetic from saved arrays; never rerun methods or modify old runs."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from run_experiment import ROOT, history_hashes, sha, write_json


def main():
    result=json.loads((ROOT/'results/RESULTS.json').read_text())
    blocks=0;max_error=0.;geometry_max=0.;ply_count=0
    rows=[]
    for run in result['runs']:
        folder=ROOT/'results/cases'/run['key']
        with np.load(folder/'outputs.npz',allow_pickle=False) as z:
            truth=z['evaluation_truth_delta'];subset=z['evaluation_subset'];xyz=z['xyz0']
            initial=np.abs(truth)
            for arm,reported in run['metrics'].items():
                theta=z['out_'+arm];error=np.abs(theta-truth);delta=error-initial
                expected=dict(mae=float(error.mean()),delta_mae=float(delta.mean()),
                    harm_sum=float(np.maximum(delta,0).sum()),gain_sum=float(np.maximum(-delta,0).sum()),
                    harm_rate=float((delta>1e-9).mean()),move_rate=float((theta!=0).mean()))
                for k,v in expected.items():max_error=max(max_error,abs(v-reported['groups']['ALL'][k]))
                total=0.
                for s in np.unique(subset):
                    contribution=float(delta[subset==s].sum()/len(truth))
                    total+=contribution
                    max_error=max(max_error,abs(contribution-reported['groups'][s]['weighted_delta']))
                max_error=max(max_error,abs(total-float(delta.mean())))
                cloud=xyz.copy();cloud[:,2]+=theta
                target=xyz.copy();target[:,2]+=truth
                a=cKDTree(target).query(cloud)[0];b=cKDTree(cloud).query(target)[0]
                for k,v in dict(nn_accuracy=a.mean(),nn_completeness=b.mean(),nn_symmetric=(a.mean()+b.mean())/2,
                                precision_1mm=(a<=1).mean(),recall_1mm=(b<=1).mean()).items():
                    geometry_max=max(geometry_max,abs(float(v)-reported['geometry'][k]))
                with (folder/(arm+'.ply')).open('rb') as f:
                    while f.readline().strip()!=b'end_header':pass
                    data=np.frombuffer(f.read(),dtype='<f4').reshape(-1,3)
                np.testing.assert_array_equal(data,cloud.astype('<f4'));ply_count+=1;blocks+=1
                if run['phase']=='confirmation':
                    for s in np.unique(subset):
                        rows.append(dict(key=run['key'],fbig=run['fbig'],calibration=run['calibration'],
                            arm=arm,subset=s,n=int((subset==s).sum()),weighted_delta=float(delta[subset==s].sum()/len(truth))))
            for name in ('anchor1_projection','anchor2_projection'):
                theta=z['out_'+name];old=z['out_test_veto_projection']
                recovered=(old==0)&(theta!=0)
                imp=np.abs(old-truth)-np.abs(theta-truth)
                block=run['recovered_ledger'][name]['ALL']
                max_error=max(max_error,abs(float(np.maximum(imp[recovered],0).sum())-block['gain_mm']),
                    abs(float(np.maximum(-imp[recovered],0).sum())-block['harm_mm']))
    stress_count=0
    for run in result['stress']:
        with np.load(ROOT/'results/stress'/run['key']/'outputs.npz',allow_pickle=False) as z:
            for world,key in [('authentic_alternative','truth_authentic'),('ghost','truth_ghost')]:
                for arm,block in run['metrics'][world].items():
                    max_error=max(max_error,abs(float(np.abs(z['out_'+arm]-z[key]).mean())-block['mae']))
                    stress_count+=1
    lock=json.loads((ROOT/'SOURCE_LOCK.json').read_text());after=history_hashes()
    changed=[p for p in set(lock['history'])|set(after) if lock['history'].get(p)!=after.get(p)]
    source_changed=[p for p,h in lock['implementation'].items() if sha(ROOT/p)!=h]
    with (ROOT/'results/LOSS_LEDGER_CONFIRMATION.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    verification=dict(metric_blocks=blocks,stress_blocks=stress_count,verified_method_plys=ply_count,
        max_coordinate_metric_difference=max_error,max_geometry_metric_difference=geometry_max,
        historical_files=len(after),historical_changed=changed,source_changed=source_changed)
    write_json(ROOT/'results/INDEPENDENT_RECHECK.json',verification)
    assert max_error<1e-10 and geometry_max<1e-10 and not changed and not source_changed
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()

