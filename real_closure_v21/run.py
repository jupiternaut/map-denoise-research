"""Frozen real reconstruction transfer. Fits never receive the reference cloud."""
from pathlib import Path
import sys, json, hashlib, tempfile, time, traceback, shutil, socket
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'exploration_v18'),str(ROOT/'exploration_v12'),str(ROOT/'published_outputs_v2')]
import v18_operator as v18
from external import project
from eval_reference import xyz, observed
DATA=Path('/srv/slam-research/grf/map-denoise/datasets')
REF=DATA/'published-outputs-v2-reference'
SOURCE=DATA/'published-outputs-v1/scan24_mesh.ply'
CAMERA=DATA/'real-closure-v21/cameras_geosvr_linked.npz'
RUNS=Path('/srv/slam-research/grf/map-denoise/runs')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    with Path(p).open('x') as f:json.dump(o,f,indent=2,allow_nan=False,default=lambda a:a.tolist() if isinstance(a,(np.ndarray,np.generic)) else str(a))

def transform(q,m):return q@m[:3,:3].T+m[:3,3]

def local(q):
    center=q.mean(0);a=q-center;ev,b=np.linalg.eigh(a.T@a/len(q));b=b[:,[2,1,0]]
    for j in range(3):
        if b[np.argmax(abs(b[:,j])),j]<0:b[:,j]*=-1
    nn=cKDTree(q).query(q,k=2)[0][:,1];h=float(np.median(nn[nn>0]));assert h>0
    return center,b,a@b,h,ev

def work(job,methods,dest):
    dest=Path(dest);q=np.load(job['input']);center,b,l,h,ev=local(q);records=[]
    for name in methods:
        start=time.perf_counter()
        try:
            model=None
            if name=='identity':out=q.copy()
            elif name=='plane':
                out=q-np.outer(l[:,2],b[:,2])
            elif name.startswith('v18_'):
                filtered,model,_=v18.filter_local(l,h*float(name.split('_')[1]));out=filtered@b.T+center
            else:
                kind,scale=name.split('_');out=project(q/1000.,kind,float(scale))*1000.
            assert out.shape==q.shape and np.isfinite(out).all()
            path=dest/'outputs'/f'{job["case"]}__{name}.npy';np.save(path,out)
            r=dict(method=name,status='OK',path=str(path),sha256=sha(path),seconds=time.perf_counter()-start,
                   moved=int(np.sum(np.linalg.norm(out-q,axis=1)>1e-7)),displacement_rms_mm=float(np.sqrt(np.mean(np.sum((out-q)**2,axis=1)))))
            if model is not None:
                mp=dest/'models'/f'{job["case"]}__{name}.json';save(mp,model);r.update(k=int(model['k']),model=str(mp))
        except Exception:r=dict(method=name,status='FAILED',error=traceback.format_exc(),seconds=time.perf_counter()-start)
        records.append(r)
    return dict(job=job,spacing_mm=h,eigenvalues=ev,records=records)

def score(bundles,dest,phase):
    # Reference is first read after all predictions in this phase are sealed.
    ref=xyz(REF/'stl024_total.ply');tree=cKDTree(ref);obs=loadmat(REF/'ObsMask24_10.mat');rows=[]
    for bundle in bundles:
        job=bundle['job'];q=np.load(job['input']);center=q.mean(0);radius=.8*np.linalg.norm(q-center,axis=1).max()
        keep=(np.linalg.norm(q-center,axis=1)<=radius)&observed(q,obs)
        ids=np.asarray(tree.query_ball_point(center,radius),int);ids=ids[observed(ref[ids],obs)]
        np.savez_compressed(dest/'evaluation'/f'{job["case"]}.npz',input_mask=keep,reference_ids=ids)
        for r in bundle['records']:
            row=dict(case=job['case'],phase=phase,method=r['method'],status=r['status'],n_accuracy=int(keep.sum()),n_reference=len(ids))
            if r['status']!='OK':row['error']=r['error'];rows.append(row);continue
            assert sha(r['path'])==r['sha256'];out=np.load(r['path'])
            if not keep.any() or not len(ids):row['status']='EMPTY_SUPPORT';rows.append(row);continue
            a=tree.query(out[keep])[0];c=cKDTree(out).query(ref[ids])[0];precision=float(np.mean(a<=1));recall=float(np.mean(c<=1))
            row.update(accuracy_mm=float(a.mean()),accuracy_p95_mm=float(np.quantile(a,.95)),completeness_mm=float(c.mean()),
                       precision_1mm=precision,recall_1mm=recall,fscore_1mm=2*precision*recall/(precision+recall) if precision+recall else 0,
                       moved=r['moved'],displacement_rms_mm=r['displacement_rms_mm'])
            rows.append(row)
    save(dest/f'{phase}_RESULTS.json',rows);return rows

def main():
    assert socket.gethostname()=='liekkas';assert CAMERA.exists(),'Exact linked metadata required before scoring'
    cameras=np.load(CAMERA);m=cameras['scale_mat_0'].astype(float)
    for key in cameras.files:
        if key.startswith('scale_mat_') and 'inv' not in key:np.testing.assert_array_equal(cameras[key],m)
    np.testing.assert_allclose(m[:3,:3],np.eye(3)*m[0,0]);assert m[0,0]>0
    dest=Path(tempfile.mkdtemp(prefix='real-closure-v21-',dir=RUNS));print('RUN',dest,flush=True)
    for name in ('inputs','outputs','models','evaluation','source'): (dest/name).mkdir()
    protected=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and 'evidence' not in p.parts]
    protected += [SOURCE,CAMERA,REF/'stl024_total.ply',REF/'ObsMask24_10.mat']
    hashes={str(p):sha(p) for p in protected};save(dest/'SOURCE_INPUT_LOCK.json',hashes)
    for p in protected:
        if p.suffix=='.py' and ROOT in p.parents:
            target=dest/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    save(dest/'IDENTITY.json',dict(host=socket.gethostname(),project=str(ROOT),python=sys.executable,scale_matrix=m))
    q=transform(xyz(SOURCE),m);tree=cKDTree(q);used=np.zeros(len(q),bool);dev=[];confirmation=[]
    for i in range(3):
        ids=np.load(RUNS/'published-outputs-v1/scan24'/f'patch{i}_source_ids.npy');used[ids]=True
        dev.append((f'dev{i}',ids))
    attempts=0
    for anchor in np.random.default_rng(921001).permutation(len(q)):
        attempts+=1
        if used[anchor]:continue
        ids=tree.query(q[anchor],k=1024)[1]
        if used[ids].any():continue
        used[ids]=True;confirmation.append((f'confirm{len(confirmation):02d}',ids))
        if len(confirmation)==24:break
    assert len(confirmation)==24
    jobs={}
    for phase,pairs in [('development',dev),('confirmation',confirmation)]:
        jobs[phase]=[]
        for case,ids in pairs:
            path=dest/'inputs'/f'{case}.npy';np.save(path,q[ids]);np.save(dest/'inputs'/f'{case}_ids.npy',ids)
            jobs[phase].append(dict(case=case,input=str(path),sha256=sha(path)))
    save(dest/'JOBS.json',dict(jobs=jobs,original_vertices=len(q),confirmation_vertices=24*1024,selection_attempts=attempts))
    selection={};summaries={};start=time.perf_counter()
    for phase in ('development','confirmation'):
        methods=['identity','plane']+([f'{f}_{s}' for f,ss in [('v18',(.5,1.,2.)),('apss',(2.,4.,8.)),('rimls',(2.,4.,8.))] for s in ss] if phase=='development' else list(selection.values()))
        bundles=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for future in as_completed([pool.submit(work,j,methods,str(dest)) for j in jobs[phase]]):
                b=future.result();bundles.append(b);print(phase,b['job']['case'],[r['status'] for r in b['records']],flush=True)
        bundles.sort(key=lambda b:b['job']['case']);save(dest/f'{phase}_SEALED.json',bundles)
        rows=score(bundles,dest,phase)
        summaries[phase]={method:dict(n=len(rr),failed=sum(r['status']!='OK' for r in rr),**{k:float(np.mean([r[k] for r in rr if r['status']=='OK'])) for k in ('accuracy_mm','completeness_mm','recall_1mm','fscore_1mm')}) for method in methods for rr in [[r for r in rows if r['method']==method]]}
        if phase=='development':
            for family in ('v18','apss','rimls'):
                eligible=[m for m in methods if m.startswith(family+'_') and summaries[phase][m]['failed']==0]
                assert eligible,f'No successful {family} baseline'
                selection[family]=min(eligible,key=lambda m:(summaries[phase][m]['accuracy_mm'],m))
            save(dest/'DEVELOPMENT_SELECTION.json',selection);print('LOCKED',selection,flush=True)
        else:
            v=selection['v18']; comparisons={}
            for other in ('identity',selection['apss'],selection['rimls']):
                vr={r['case']:r for r in rows if r['method']==v};orr={r['case']:r for r in rows if r['method']==other}
                clean=all(r['status']=='OK' for r in list(vr.values())+list(orr.values()))
                comparisons[other]=dict(no_failures=clean)
                if clean:
                    gain=1-summaries[phase][v]['accuracy_mm']/summaries[phase][other]['accuracy_mm'];wins=np.mean([vr[c]['accuracy_mm']<orr[c]['accuracy_mm'] for c in vr]);rec=summaries[phase][v]['recall_1mm']-summaries[phase][other]['recall_1mm']
                    comparisons[other].update(relative_gain=float(gain),win_fraction=float(wins),recall_difference=float(rec),passed=bool(gain>=.05 and wins>=2/3 and rec>=-.01))
            summaries['comparisons']=comparisons
    assert all(sha(p)==h for p,h in hashes.items())
    save(dest/'SUMMARY.json',dict(results=summaries,selection=selection,wall_seconds=time.perf_counter()-start,history_unchanged=True))
    print(json.dumps(summaries,indent=2),flush=True)

if __name__=='__main__':main()
