"""Independent arithmetic checks of the saved V22 geometry and controls."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import run
def read(p):return json.loads(Path(p).read_text())
def main():
    dest=Path(sys.argv[1]);lock=read(dest/'SOURCE_LOCK.json');assert all(run.sha(p)==h for p,h in lock.items())
    counted=0;scored=0;largest=0;used=set();reference_counts={}
    jobs=read(dest/'JOBS.json');scene=run.DATA/'reconstruction-v22-scan37';m=np.asarray(jobs['matrix']);source=run.xyz(scene/'scan37_mesh.ply')@m[:3,:3].T+m[:3,3]
    for phase in ('development','confirmation'):
        ref=run.xyz(run.DATA/'published-outputs-v2-reference/stl024_total.ply' if phase=='development' else scene/'stl037_total.ply');tree=cKDTree(ref);reference_counts[phase]=len(ref)
        rows={(r['case'],r['method']):r for r in read(dest/f'{phase}_RESULTS.json')}
        for bundle in read(dest/f'{phase}_SEALED.json'):
            job=bundle['job'];q=np.load(job['input']);assert run.sha(job['input'])==job['sha256'];evals=np.load(dest/'evaluation'/f'{job["case"]}.npz');keep=evals['input_mask'];ids=evals['reference_ids'];outs={}
            if phase=='confirmation':
                source_ids=np.load(dest/'inputs'/f'{job["case"]}_ids.npy');assert not used.intersection(source_ids);used.update(source_ids);np.testing.assert_array_equal(q,source[source_ids])
            for rec in bundle['records']:
                if rec['status']!='OK':continue
                assert run.sha(rec['path'])==rec['sha256'];out=np.load(rec['path']);assert out.shape==q.shape and np.isfinite(out).all();outs[rec['method']]=out;counted+=1
                r=rows[(job['case'],rec['method'])]
                if r['status']!='OK':continue
                a=tree.query(out[keep])[0];b=cKDTree(out).query(ref[ids])[0];p=np.mean(a<=1);re=np.mean(b<=1)
                metrics=dict(accuracy_mm=float(a.mean()),p95_mm=float(np.quantile(a,.95)),completeness_mm=float(b.mean()),precision=float(p),recall=float(re),fscore=float(2*p*re/(p+re)) if p+re else 0.,displacement_rms_mm=float(np.sqrt(np.mean(np.sum((out-q)**2,axis=1)))))
                for k,v in metrics.items():
                    difference=abs(v-r[k]);largest=max(largest,difference);assert difference<1e-10
                scored+=1
            np.testing.assert_array_equal(q,outs['identity'])
            d=read(dest/'diagnostics'/f'{job["case"]}.json');c=d['operator']['consensus'];alpha=np.asarray(c['alpha']);assert np.all((alpha>=0)&(alpha<=1));normal=np.asarray(c['normal']);delta=outs['multiscale_full']-q
            np.testing.assert_allclose(outs['multiscale_consensus']-q,alpha[:,None]*delta,atol=1e-10)
            np.testing.assert_allclose(np.linalg.norm(outs['multiscale_consensus']-q),np.linalg.norm(outs['multiscale_matched']-q),atol=1e-9)
            for name in ('multiscale_full','multiscale_consensus'):
                move=outs[name]-q;tangent=move-np.sum(move*normal,axis=1)[:,None]*normal;assert np.max(abs(tangent))<1e-10
            neighbours=np.asarray(d['operator']['neighbour_ids']);assert not np.any(neighbours==np.arange(len(q))[:,None])
            if phase=='development':
                old_case=job['case'].removeprefix('s24_')
                for name,old_name in [('v18','v18_0.5'),('apss2','apss_2.0'),('rimls2','rimls_2.0')]:
                    previous=np.load(run.PREVIOUS/'outputs'/f'{old_case}__{old_name}.npy');np.testing.assert_allclose(outs[name],previous,atol=1e-5,rtol=0)
    result=dict(checked_outputs=counted,scored_outputs=scored,max_metric_difference=largest,confirmation_distinct_source_points=len(used),reference_points=reference_counts,history_hashes_unchanged=True,matched_movement_checked=True,old_baseline_replay_checked=True)
    run.save(dest/'VERIFICATION.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
