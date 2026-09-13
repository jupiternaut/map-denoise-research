"""V22 frozen development diagnostic and one-new-scene confirmation."""
from pathlib import Path
import sys,json,hashlib,tempfile,time,traceback,socket,importlib.util,shutil
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from scipy.spatial import cKDTree
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('v22_operator',HERE/'operator.py');op=importlib.util.module_from_spec(spec);spec.loader.exec_module(op)
sys.path[:0]=[str(ROOT/'exploration_v18'),str(ROOT/'exploration_v12'),str(ROOT/'published_outputs_v2')]
import v18_operator as old
from external import project
from eval_reference import xyz,observed
DATA=Path('/srv/slam-research/grf/map-denoise/datasets');RUNS=Path('/srv/slam-research/grf/map-denoise/runs')
PREVIOUS=RUNS/'real-closure-v21-dhb1ebbx'
METHODS=('identity','v18','apss2','rimls2','local_plane64','quadratic64','multiscale_full','multiscale_consensus','multiscale_matched')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,o):
    with Path(p).open('x') as f:json.dump(o,f,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,(np.ndarray,np.generic)) else str(x))

def work(job,dest):
    dest=Path(dest);q=np.load(job['input']);start=time.perf_counter();records=[];outputs={};errors={}
    try:
        outputs,diagnostic=op.construct(q)
        full=outputs['multiscale_full']-q;shrunk=outputs['multiscale_consensus']-q
        ratio=float(np.linalg.norm(shrunk)/max(np.linalg.norm(full),1e-30))
        outputs['multiscale_matched']=q+ratio*full
        np.testing.assert_allclose(np.linalg.norm(outputs['multiscale_matched']-q),np.linalg.norm(shrunk),atol=1e-9)
        save(dest/'diagnostics'/f'{job["case"]}.json',dict(operator=diagnostic,matched_alpha=ratio))
    except Exception:
        for name in METHODS[4:]:errors[name]=traceback.format_exc()
    outputs['identity']=q.copy()
    for name in ('v18','apss2','rimls2'):
        try:
            if name=='v18':
                c=q.mean(0);a=q-c;_,b=np.linalg.eigh(a.T@a/len(q));b=b[:,[2,1,0]]
                for j in range(3):
                    if b[np.argmax(abs(b[:,j])),j]<0:b[:,j]*=-1
                nn=cKDTree(q).query(q,k=2)[0][:,1];h=np.median(nn[nn>0]);v,m,d=old.filter_local(a@b,.5*h);outputs[name]=v@b.T+c
            else:outputs[name]=project(q/1000.,'apss' if name=='apss2' else 'rimls',2.)*1000.
        except Exception:errors[name]=traceback.format_exc()
    for name in METHODS:
        if name in errors:records.append(dict(method=name,status='FAILED',error=errors[name]));continue
        out=outputs[name];assert out.shape==q.shape and np.isfinite(out).all()
        p=dest/'outputs'/f'{job["case"]}__{name}.npy';np.save(p,out);delta=np.linalg.norm(out-q,axis=1)
        records.append(dict(method=name,status='OK',path=str(p),sha256=sha(p),displacement_rms_mm=float(np.sqrt(np.mean(delta**2))),displacement_p95_mm=float(np.quantile(delta,.95)),moved=int(np.sum(delta>1e-7))))
    return dict(job=job,records=records,seconds=time.perf_counter()-start)

def score(dest,phase,bundles,refpath,maskpath):
    ref=xyz(refpath);tree=cKDTree(ref);obs=loadmat(maskpath);rows=[]
    for b in bundles:
        job=b['job'];q=np.load(job['input']);c=q.mean(0);r=.8*np.linalg.norm(q-c,axis=1).max()
        keep=(np.linalg.norm(q-c,axis=1)<=r)&observed(q,obs)
        ids=np.asarray(tree.query_ball_point(c,r),int);ids=ids[observed(ref[ids],obs)]
        np.savez_compressed(dest/'evaluation'/f'{job["case"]}.npz',input_mask=keep,reference_ids=ids)
        for rec in b['records']:
            row=dict(case=job['case'],phase=phase,method=rec['method'],status=rec['status'],n_input=int(keep.sum()),n_reference=len(ids))
            if rec['status']!='OK':row['error']=rec['error'];rows.append(row);continue
            assert sha(rec['path'])==rec['sha256'];out=np.load(rec['path'])
            if not keep.any() or not len(ids):row['status']='EMPTY_SUPPORT';rows.append(row);continue
            a=tree.query(out[keep])[0];d=cKDTree(out).query(ref[ids])[0];p=float(np.mean(a<=1));recall=float(np.mean(d<=1))
            row.update(accuracy_mm=float(a.mean()),p95_mm=float(np.quantile(a,.95)),completeness_mm=float(d.mean()),precision=p,recall=recall,fscore=2*p*recall/(p+recall) if p+recall else 0.,displacement_rms_mm=rec['displacement_rms_mm'])
            rows.append(row)
    save(dest/f'{phase}_RESULTS.json',rows)
    summary={}
    for name in METHODS:
        rr=[r for r in rows if r['method']==name];valid=[r for r in rr if r['status']=='OK']
        summary[name]=dict(n=len(rr),valid=len(valid),**{key:float(np.mean([r[key] for r in valid])) if valid else None for key in ('accuracy_mm','completeness_mm','recall','fscore','displacement_rms_mm')})
    comparisons={};primary={r['case']:r for r in rows if r['method']=='multiscale_consensus' and r['status']=='OK'}
    for name in ('identity','v18','apss2','rimls2','multiscale_matched'):
        rr={r['case']:r for r in rows if r['method']==name and r['status']=='OK'};shared=sorted(primary.keys()&rr.keys())
        a=np.array([primary[c]['accuracy_mm'] for c in shared]);b=np.array([rr[c]['accuracy_mm'] for c in shared]);rec=float(np.mean([primary[c]['recall']-rr[c]['recall'] for c in shared])) if shared else None
        gain=float(1-a.mean()/b.mean()) if shared else None;wins=float(np.mean(a<b)) if shared else None
        comparisons[name]=dict(paired=len(shared),relative_gain=gain,win_fraction=wins,recall_difference=rec,passed=bool(len(shared)==len(bundles) and gain>=.05 and wins>=2/3 and rec>=-.01))
    return dict(summary=summary,comparisons=comparisons)

def prepare(dest):
    jobs={'development':[],'confirmation':[]}
    # Reuse exactly the 18 already-exposed scoreable V21 inputs.
    rows=json.loads((PREVIOUS/'confirmation_RESULTS.json').read_text());valid={r['case'] for r in rows if r['method']=='identity' and r['status']=='OK'}
    for b in json.loads((PREVIOUS/'confirmation_SEALED.json').read_text()):
        if b['job']['case'] not in valid:continue
        case='s24_'+b['job']['case'];p=dest/'inputs'/f'{case}.npy';shutil.copyfile(b['job']['input'],p)
        jobs['development'].append(dict(case=case,input=str(p),sha256=sha(p)))
    scene=DATA/'reconstruction-v22-scan37';cam=np.load(scene/'cameras.npz');m=cam['scale_mat_0'].astype(float)
    for key in cam.files:
        if key.startswith('scale_mat_') and 'inv' not in key:np.testing.assert_array_equal(cam[key],m)
    np.testing.assert_allclose(m[:3,:3],np.eye(3)*m[0,0]);assert m[0,0]>0
    q=xyz(scene/'scan37_mesh.ply')@m[:3,:3].T+m[:3,3];tree=cKDTree(q);used=np.zeros(len(q),bool);obs=loadmat(scene/'ObsMask37_10.mat');attempts=0
    for anchor in np.random.default_rng(922003).permutation(len(q)):
        attempts+=1
        if used[anchor]:continue
        ids=tree.query(q[anchor],k=1024)[1]
        if used[ids].any():continue
        z=q[ids];c=z.mean(0);r=.8*np.linalg.norm(z-c,axis=1).max();keep=(np.linalg.norm(z-c,axis=1)<=r)&observed(z,obs)
        if keep.sum()<512:continue
        case=f's37_confirm{len(jobs["confirmation"]):02d}';used[ids]=True;p=dest/'inputs'/f'{case}.npy';np.save(p,z);np.save(dest/'inputs'/f'{case}_ids.npy',ids)
        jobs['confirmation'].append(dict(case=case,input=str(p),sha256=sha(p)))
        if len(jobs['confirmation'])==24:break
    assert len(jobs['confirmation'])==24
    save(dest/'JOBS.json',dict(jobs=jobs,scan37_vertices=len(q),confirmation_vertices=int(used.sum()),attempts=attempts,matrix=m))
    return jobs

def main():
    assert socket.gethostname()=='liekkas';scene=DATA/'reconstruction-v22-scan37'
    for name in ('cameras.npz','scan37_mesh.ply','stl037_total.ply','ObsMask37_10.mat'):assert (scene/name).exists(),name
    dest=Path(tempfile.mkdtemp(prefix='reconstruction-v22-',dir=RUNS));print('RUN',dest,flush=True)
    for name in ('inputs','outputs','diagnostics','evaluation','source'):(dest/name).mkdir()
    files=[p for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and 'evidence' not in p.parts]+list(scene.glob('*'))
    lock={str(p):sha(p) for p in files};save(dest/'SOURCE_LOCK.json',lock)
    for p in files:
        if ROOT in p.parents and p.suffix=='.py':
            target=dest/'source'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target)
    jobs=prepare(dest);summaries={};start=time.perf_counter()
    for phase in ('development','confirmation'):
        bundles=[]
        with ProcessPoolExecutor(max_workers=4) as pool:
            for fut in as_completed([pool.submit(work,j,str(dest)) for j in jobs[phase]]):
                b=fut.result();bundles.append(b);print(phase,b['job']['case'],sum(r['status']!='OK' for r in b['records']),flush=True)
        bundles.sort(key=lambda b:b['job']['case']);save(dest/f'{phase}_SEALED.json',bundles)
        refpath,maskpath=(DATA/'published-outputs-v2-reference/stl024_total.ply',DATA/'published-outputs-v2-reference/ObsMask24_10.mat') if phase=='development' else (scene/'stl037_total.ply',scene/'ObsMask37_10.mat')
        summaries[phase]=score(dest,phase,bundles,refpath,maskpath);print(phase,json.dumps(summaries[phase]),flush=True)
    assert all(sha(p)==h for p,h in lock.items())
    save(dest/'SUMMARY.json',dict(results=summaries,wall_seconds=time.perf_counter()-start,history_unchanged=True));print('DONE',dest,flush=True)
if __name__=='__main__':main()
