"""Read-only experiment recomputation; writes one new verification record."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from PIL import Image
import run
def main():
    dest=Path(sys.argv[1]);load=lambda p:json.loads(Path(p).read_text());sealed=load(dest/'SEALED_BEFORE_GT.json');rows=load(dest/'RESULTS.json')
    locks=load(dest/'SOURCE_LOCK.json');assert all(run.sha(p)==h for p,h in locks.items())
    lookup={(r['case'],r['method']):r for r in rows};count=0;maxdiff=0.;photodiff=0.;photo_count=0
    for scene in (24,37):
        ref=run.xyz(run.DATA/'published-outputs-v2-reference/stl024_total.ply' if scene==24 else run.DATA/'reconstruction-v22-scan37/stl037_total.ply');tree=cKDTree(ref)
        provenance=load(dest/f'scan{scene}_PROVENANCE.json');images=[np.asarray(Image.open(p['path']).convert('RGB')) for p in provenance['images']];matrices=np.asarray(provenance['matrices'])
        for p in provenance['images']+provenance['calibration']:assert run.sha(p['path'])==p['sha256']
        for b in [b for b in sealed if b['scene']==scene]:
            case=b['job']['case'];q=np.load(b['job']['input']);assert run.sha(b['job']['input'])==b['job']['sha256'];evaluation=np.load(run.OLD/'evaluation'/f'{case}.npz');gt=ref[evaluation['reference_ids']];keep=evaluation['input_mask'];output={}
            for rec in b['records']:
                assert run.sha(rec['path'])==rec['sha256'];a=np.load(rec['path']);assert a.shape==q.shape and np.isfinite(a).all();output[rec['method']]=a
                da=tree.query(a[keep])[0];db=cKDTree(a).query(gt)[0];p=float(np.mean(da<=1));r=float(np.mean(db<=1))
                values=dict(accuracy_mm=float(da.mean()),completeness_mm=float(db.mean()),recall=r,precision=p,fscore=2*p*r/(p+r) if p+r else 0.,displacement_rms_mm=float(np.sqrt(np.mean(np.sum((a-q)**2,axis=1)))))
                row=lookup[(case,rec['method'])]
                for k,v in values.items():maxdiff=max(maxdiff,abs(v-row[k]));assert abs(v-row[k])<1e-10
                count+=1
            np.testing.assert_array_equal(output['identity'],q)
            for name,choice in b['choices'].items():
                np.testing.assert_array_equal(output[name],output[choice]);loss=b['fit_loss' if name=='fit_selector' else 'photo_loss'];valid={m:v for m,v in loss.items() if v is not None}
                if valid:assert abs(loss[choice]-min(valid.values()))<1e-15
            support=np.load(dest/'support'/f'{case}.npz')['support'];n=support.sum(0);eligible=n>=2
            for name in set(['identity',b['choices']['photo_selector']]):
                coords=output[name];colors=[];valid=[]
                for image,P in zip(images,matrices):
                    proj=coords@P[:3,:3].T+P[:3,3];uv=proj[:,:2]/proj[:,2,None];h,w=image.shape[:2]
                    ok=(proj[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<w-1)&(uv[:,1]>=0)&(uv[:,1]<h-1)
                    color=np.column_stack([map_coordinates(image[:,:,c].astype(float),[uv[:,1],uv[:,0]],order=1,mode='constant',cval=0,prefilter=False) for c in range(3)])/255.;colors.append(color);valid.append(ok)
                colors=np.asarray(colors);mean=np.sum(colors*support[:,:,None],axis=0)/np.maximum(n,1)[:,None]
                v=np.sum((colors-mean[None])**2*support[:,:,None],axis=(0,2))/(3*np.maximum(n,1));v[np.any(support&~np.asarray(valid),axis=0)]=1.
                if eligible.any():
                    diff=abs(float(v[eligible].mean())-b['photo_loss'][name]);photodiff=max(photodiff,diff);assert diff<1e-10;photo_count+=1
    run.save(dest/'VERIFICATION.json',dict(geometry_outputs=count,max_metric_difference=maxdiff,photo_losses_recomputed=photo_count,max_photo_difference=photodiff,selection_output_identity=True,source_hashes_unchanged=True,old_input_and_image_hashes_verified=True,scope='Actual output metrics, proxy minima and independent bilinear-sampling check; not a proof of photometric validity.'))
    print(count,maxdiff,photo_count,photodiff)
if __name__=='__main__':main()
