"""Additional audit after strict historical APSS replay check failed.

Original verify.py and experimental outputs stay frozen. This audit verifies
actual-output metrics and records, rather than suppresses, historical drift.
"""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import run
def read(p):return json.loads(Path(p).read_text())
def main():
    dest=Path(sys.argv[1]);assert all(run.sha(p)==h for p,h in read(dest/'SOURCE_LOCK.json').items())
    count=0;maximum=0.;used=set();replay={m:[] for m in ('v18','apss2','rimls2')};shape_count=0
    jobs=read(dest/'JOBS.json');scene=run.DATA/'reconstruction-v22-scan37';matrix=np.asarray(jobs['matrix']);source=run.xyz(scene/'scan37_mesh.ply')@matrix[:3,:3].T+matrix[:3,3]
    for phase in ('development','confirmation'):
        ref=run.xyz(run.DATA/'published-outputs-v2-reference/stl024_total.ply' if phase=='development' else scene/'stl037_total.ply');tree=cKDTree(ref)
        rows={(r['case'],r['method']):r for r in read(dest/f'{phase}_RESULTS.json')}
        for bundle in read(dest/f'{phase}_SEALED.json'):
            job=bundle['job'];q=np.load(job['input']);assert run.sha(job['input'])==job['sha256'];e=np.load(dest/'evaluation'/f'{job["case"]}.npz');keep=e['input_mask'];gt=ref[e['reference_ids']];outputs={}
            if phase=='confirmation':
                ids=np.load(dest/'inputs'/f'{job["case"]}_ids.npy');assert not used.intersection(ids);used.update(ids);np.testing.assert_array_equal(q,source[ids])
            for rec in bundle['records']:
                assert rec['status']=='OK';assert run.sha(rec['path'])==rec['sha256'];out=np.load(rec['path']);assert out.shape==q.shape and np.isfinite(out).all();outputs[rec['method']]=out;shape_count+=1
                row=rows[(job['case'],rec['method'])];assert row['status']=='OK'
                a=tree.query(out[keep])[0];c=cKDTree(out).query(gt)[0];p=np.mean(a<=1);r=np.mean(c<=1)
                values=dict(accuracy_mm=float(a.mean()),p95_mm=float(np.quantile(a,.95)),completeness_mm=float(c.mean()),precision=float(p),recall=float(r),fscore=float(2*p*r/(p+r)) if p+r else 0.,displacement_rms_mm=float(np.sqrt(np.mean(np.sum((out-q)**2,axis=1)))))
                for key,value in values.items():
                    diff=abs(value-row[key]);maximum=max(maximum,diff);assert diff<1e-10
                count+=1
            np.testing.assert_array_equal(outputs['identity'],q)
            diagnostic=read(dest/'diagnostics'/f'{job["case"]}.json')['operator'];con=diagnostic['consensus'];alpha=np.asarray(con['alpha']);assert np.all((alpha>=0)&(alpha<=1))
            full=outputs['multiscale_full']-q;small=outputs['multiscale_consensus']-q
            np.testing.assert_allclose(small,alpha[:,None]*full,atol=1e-10,rtol=0)
            np.testing.assert_allclose(np.linalg.norm(small),np.linalg.norm(outputs['multiscale_matched']-q),atol=1e-9,rtol=0)
            normal=np.asarray(con['normal']);assert np.max(abs(small-np.sum(small*normal,axis=1)[:,None]*normal))<1e-10
            neighbours=np.asarray(diagnostic['neighbour_ids']);assert not np.any(neighbours==np.arange(len(q))[:,None])
            if phase=='development':
                case=job['case'].removeprefix('s24_')
                for name,old_name in [('v18','v18_0.5'),('apss2','apss_2.0'),('rimls2','rimls_2.0')]:
                    old=np.load(run.PREVIOUS/'outputs'/f'{case}__{old_name}.npy');replay[name].append(dict(case=case,max_coordinate_difference_mm=float(np.max(abs(outputs[name]-old))),rms_difference_mm=float(np.sqrt(np.mean(np.sum((outputs[name]-old)**2,axis=1))))))
    result=dict(checked_output_shapes=shape_count,independently_scored_outputs=count,max_metric_difference=maximum,matched_displacement_check=True,confirmation_distinct_source_points=len(used),source_hashes_unchanged=True,
                original_verifier_status='FAILED on historical APSS replay atol1e-5mm; not rewritten or declared passing',
                historical_replay=replay,max_historical_coordinate_difference_mm={k:max(r['max_coordinate_difference_mm'] for r in rr) for k,rr in replay.items()})
    run.save(dest/'FINAL_AUDIT.json',result);print(json.dumps({k:v for k,v in result.items() if k!='historical_replay'},indent=2))
if __name__=='__main__':main()
