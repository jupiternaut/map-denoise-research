"""Frozen official COLMAP transfer. Inference never opens reference geometry."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import argparse, hashlib, json, socket, subprocess, time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'colmap-fixed-pixels-20260930T172250Z'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def dump(p,v):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x') as f: json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def npz(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def verify(files):
    for p,h in files.items(): assert sha(ROOT/p)==h,p
def gray(p):
    with Image.open(p) as im: rgb=np.asarray(im.convert('RGB').resize((777,581),Image.Resampling.BILINEAR))
    return np.clip(rgb@np.array([.299,.587,.114]),0,255).astype(np.uint8)
def backproject(P,uv,z):
    P=np.asarray(P); C=-np.linalg.solve(P[:,:3],P[:,3])
    return C+np.asarray(z)[:,None]*(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(P[:,:3]).T)
def read_depth(p):
    with Path(p).open('rb') as f:
        fields=[]; b=bytearray()
        while len(fields)<3:
            c=f.read(1)
            if not c: raise ValueError('truncated depth')
            if c==b'&': fields.append(int(b)); b=bytearray()
            else:b.extend(c)
        w,h,n=fields; v=np.fromfile(f,dtype='<f4')
    assert v.size==w*h*n
    a=v.reshape((w,h,n),order='F').transpose(1,0,2)
    return a[:,:,0] if n==1 else a
def freeze():
    # All executable dependencies are bound before either CPU or GPU prediction.
    required=['prepare_transfer.py','cpu_baseline.py','mvs_transfer.py','evaluate_transfer.py',
              'fetch_reference.py','AGENTS.md','refine-logs/EXPERIMENT_PLAN_20260930_180000.md',
              'PLAN.json','CAMERA_CHECKS.json','SCENE_SELECTION.json','SCENE_SELECTION.md',
              'acquire_inputs.py','INPUT_MANIFEST.json','test_transfer_preparation.py']
    files={p:sha(ROOT/p) for p in required}
    for p in (ROOT/'inputs').rglob('*'):
        if p.is_file(): files[str(p.relative_to(ROOT))]=sha(p)
    dependency_paths=[ROOT.parent/'camera-pairing-replay-20260930T162130Z/camera_mapping.py',
       ROOT.parent/'upstream-photo-holdout-20260930T113213Z/rebuild.py',
       ROOT.parent/'upstream-photo-holdout-20260930T113213Z/dense_rebuild.py',
       Path('/home/grf/Documents/Codex/2026-09-22/closeout-confirmation-20260922T133739Z/data_probe.py')]
    dependencies={str(p):sha(p) for p in dependency_paths}
    dump(ROOT/'RUN_LOCK.json',dict(files=files,dependencies=dependencies,
       predecessor=str(OLD),predecessor_seal_sha256=sha(OLD/'DELIVERY_MANIFEST.json'),
       scenes=[118,122],gt_accessed=False,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('FROZEN',len(files),'files',len(dependencies),'dependencies',flush=True)
def verify_lock():
    lock=load(ROOT/'RUN_LOCK.json'); verify(lock['files'])
    for p,h in lock['dependencies'].items(): assert sha(p)==h,p
def prepare():
    import pycolmap as pc
    from scipy.spatial.transform import Rotation
    verify_lock(); assert pc.__version__=='4.2.1' and pc.has_cuda
    plan=load(ROOT/'PLAN.json'); jobs=[]; checks=[]
    for sid,s in plan['scenes'].items():
        names=[s['reference']]+s['Q']; ref=s['reference']; cams=s['cameras']
        interval=load(ROOT/f'RANGES_{sid}.json')['depth_range_mm']
        corners=np.array([[0,0],[776,0],[0,580],[776,580]],float)
        vertices=np.concatenate([backproject(cams[ref]['P'],corners,np.full(4,d)) for d in interval])
        ranges={ref:interval}
        for n in s['Q']:
            P=np.array(cams[n]['P']); z=vertices@P[2,:3]+P[2,3]
            ranges[n]=[float(np.floor(z.min())),float(np.ceil(z.max()))]
            assert 0<ranges[n][0]<ranges[n][1]
        for variant in ['photo','geo']:
            ws=ROOT/'workspaces'/f'scan{sid}'/variant
            for sub in ['images','sparse','stereo/depth_maps','stereo/normal_maps','stereo/consistency_graphs','configs']:
                (ws/sub).mkdir(parents=True,exist_ok=False)
            ca=[]; ims=[]
            for i,n in enumerate(names,1):
                c=cams[n]; K=np.array(c['K_half']); R=np.array(c['R']); C=np.array(c['center']); t=-R@C
                P=K@np.column_stack((R,t)); assert np.max(np.abs(P-np.array(c['P'])))<1e-7
                a=gray(c['image_path']); Image.fromarray(a).save(ws/'images'/n)
                uv=np.array([[0,0],[388,290],[776,580],[123,321]],float); z=np.array([600,700,800,750])
                local=np.c_[(uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(4)]*z[:,None]
                error=float(np.max(np.abs(local@R+C-backproject(P,uv,z)))); assert error<1e-8
                q=Rotation.from_matrix(R).as_quat(); q=np.r_[q[3],q[:3]]
                ca.append(f'{i} PINHOLE 777 581 '+' '.join(format(v,'.17g') for v in [K[0,0],K[1,1],K[0,2],K[1,2]]))
                ims.append(f'{i} '+' '.join(format(v,'.17g') for v in [*q,*t])+f' {i} {n}\n')
                (ws/'configs'/f'{n}.cfg').write_text(n+'\n'+', '.join(x for x in names if x!=n)+'\n')
                checks.append(dict(scene=int(sid),variant=variant,image=n,ray_max_abs_mm=error,gray_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
            (ws/'sparse/cameras.txt').write_text('\n'.join(ca)+'\n'); (ws/'sparse/images.txt').write_text('\n'.join(ims)+'\n')
            (ws/'sparse/points3D.txt').write_text('')
            rec=pc.Reconstruction(ws/'sparse'); assert rec.num_reg_images()==5 and rec.num_points3D()==0
            tasks=[(ref,False,True)] if variant=='photo' else [(n,False,False) for n in names]+[(ref,True,True)]
            for n,geom,filtering in tasks:
                o=pc.PatchMatchOptions(); o.gpu_index='0'; o.cache_size=2.; o.num_threads=2
                o.depth_min,o.depth_max=ranges[n]; o.geom_consistency=geom; o.filter=filtering
                jobs.append(dict(scene=int(sid),variant=variant,reference=n,workspace=str(ws),config=str(ws/'configs'/f'{n}.cfg'),options=o.todict()))
    files={str(p.relative_to(ROOT)):sha(p) for sub in ['workspaces','cpu'] for p in (ROOT/sub).rglob('*') if p.is_file()}
    for sid in plan['scenes']: files[f'RANGES_{sid}.json']=sha(ROOT/f'RANGES_{sid}.json')
    dump(ROOT/'MVS_LOCK.json',dict(files=files,jobs=jobs,checks=checks,version=pc.__version__,gt_accessed=False))
    print('MVS_PREPARED',len(jobs),flush=True)
def run():
    import pycolmap as pc
    verify_lock(); lock=load(ROOT/'MVS_LOCK.json'); verify(lock['files'])
    for i,j in enumerate(lock['jobs']):
        target=ROOT/'job_records'/f'{i:02d}.json'
        if target.exists(): verify(load(target)['files']); continue
        status=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        other=[x for x in status.splitlines() if x.strip() and 'chrome' not in x and int(x.split(',',1)[0])!=os.getpid()]
        if other:raise RuntimeError('Unrelated GPU compute present: '+str(other))
        tick=time.monotonic(); print('JOB',i,j['scene'],j['variant'],j['reference'],j['options']['geom_consistency'],flush=True)
        pc.patch_match_stereo(workspace_path=j['workspace'],options=pc.PatchMatchOptions(j['options']),config_path=j['config'])
        kind='geometric' if j['options']['geom_consistency'] else 'photometric'
        paths=[Path(j['workspace'])/'stereo'/f/(j['reference']+'.'+kind+'.bin') for f in ['depth_maps','normal_maps']]
        arrays=[read_depth(p) for p in paths]; assert arrays[0].shape==(581,777) and arrays[1].shape==(581,777,3)
        dump(target,dict(job=i,seconds=time.monotonic()-tick,files={str(p.relative_to(ROOT)):sha(p) for p in paths},valid_depths=int((arrays[0]>0).sum())))
        print('DONE',i,round(time.monotonic()-tick,2),flush=True)
    plan=load(ROOT/'PLAN.json')
    for sid,s in plan['scenes'].items():
        for roi in s['rois']:
            uv=np.array(roi['pixel_xy']); values=dict(pixel_xy=uv)
            for arm,kind in [('photo','photometric'),('geo','geometric')]:
                d=read_depth(ROOT/'workspaces'/f'scan{sid}'/arm/'stereo/depth_maps'/(s['reference']+'.'+kind+'.bin'))
                z=d[uv[:,1].astype(int),uv[:,0].astype(int)].astype(float); valid=np.isfinite(z)&(z>0)
                xyz=backproject(s['cameras'][s['reference']]['P'],uv,z); xyz[~valid]=np.nan
                values.update({arm+'_depth_mm':z,arm+'_valid':valid,arm+'_xyz_mm':xyz})
            p=ROOT/'predictions'/(roi['id']+'.npz'); p.parent.mkdir(exist_ok=True)
            with p.open('xb') as f:np.savez_compressed(f,**values)
    files={str(p.relative_to(ROOT)):sha(p) for sub in ['predictions','cpu','job_records','workspaces'] for p in (ROOT/sub).rglob('*') if p.is_file()}
    for n in ['RUN_LOCK.json','MVS_LOCK.json']: files[n]=sha(ROOT/n)
    dump(ROOT/'PREDICTIONS_SEALED.json',dict(files=files,gt_accessed=False,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print('PREDICTIONS_SEALED',len(files),flush=True)
if __name__=='__main__':
    assert socket.gethostname()=='liekkas'
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['freeze','prepare','run']); a=p.parse_args()
    {'freeze':freeze,'prepare':prepare,'run':run}[a.action]()
