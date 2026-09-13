"""Read-only numerical recheck of sealed point outputs and fixed evaluation sets."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import run

def read(p):return json.loads(Path(p).read_text())

def main():
    dest=Path(sys.argv[1]);lock=read(dest/'SOURCE_INPUT_LOCK.json')
    assert all(run.sha(p)==h for p,h in lock.items())
    ref=run.xyz(run.REF/'stl024_total.ply');tree=cKDTree(ref)
    original=run.xyz(run.SOURCE);matrix=read(dest/'IDENTITY.json')['scale_matrix'];original=run.transform(original,np.asarray(matrix))
    jobs=read(dest/'JOBS.json');used=set();checked=0;maximum=0;count=0
    for phase in ('development','confirmation'):
        rows=read(dest/f'{phase}_RESULTS.json');index={(r['case'],r['method']):r for r in rows}
        for b in read(dest/f'{phase}_SEALED.json'):
            job=b['job'];q=np.load(job['input']);ids=np.load(dest/'inputs'/f'{job["case"]}_ids.npy')
            assert run.sha(job['input'])==job['sha256'];np.testing.assert_array_equal(q,original[ids])
            if phase=='confirmation':assert not used.intersection(ids)
            used.update(ids)
            ev=np.load(dest/'evaluation'/f'{job["case"]}.npz');keep=ev['input_mask'];reference=ref[ev['reference_ids']]
            center=q.mean(0);radius=.8*np.linalg.norm(q-center,axis=1).max()
            assert np.all(np.linalg.norm(reference-center,axis=1)<=radius+1e-9)
            if phase=='confirmation':count+=int(keep.sum())
            for r in b['records']:
                if r['status']!='OK':continue
                assert run.sha(r['path'])==r['sha256'];out=np.load(r['path']);assert out.shape==q.shape and np.isfinite(out).all()
                row=index[(job['case'],r['method'])]
                if row['status']!='OK':continue
                a=tree.query(out[keep])[0];c=cKDTree(out).query(reference)[0];p=np.mean(a<=1);rec=np.mean(c<=1)
                values=dict(accuracy_mm=float(a.mean()),accuracy_p95_mm=float(np.quantile(a,.95)),completeness_mm=float(c.mean()),precision_1mm=float(p),recall_1mm=float(rec),fscore_1mm=float(2*p*rec/(p+rec)) if p+rec else 0.)
                for key,value in values.items():
                    diff=abs(row[key]-value);maximum=max(maximum,diff);assert diff<1e-10
                if r['method']=='identity':np.testing.assert_array_equal(q,out)
                checked+=1
    cams=np.load(run.CAMERA);old=np.load(run.REF/'cameras_surfels.npz')
    assert run.sha(run.CAMERA)==run.sha(run.REF/'cameras_surfels.npz')
    result=dict(outputs_checked=checked,max_metric_difference=maximum,history_hashes_unchanged=True,
                confirmation_source_ids_disjoint=True,confirmation_accuracy_rows=count,
                original_vertices=len(original),coordinate_metadata_exact_match=True,
                scope='Same scene local point reconstruction accuracy, not layer-identity or mesh verification.')
    run.save(dest/'VERIFICATION.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
