"""Official COLMAP 4.2.1 MVS, frozen cameras/images/query pixels. No GT access."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse, hashlib, json, socket, time, subprocess
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import pycolmap as pc

ROOT=Path(__file__).resolve().parent
OLD=ROOT.parent/'camera-pairing-replay-20260930T162130Z'
BASE=ROOT.parent/'upstream-photo-holdout-20260930T113213Z'

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def load(p):return json.loads(Path(p).read_text())
def dump(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    with Path(p).open('x') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False)
def npz(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k].copy() for k in f.files}
def bind(files,p):files[str(p)]=sha(p)
def gray(path,size):
    with Image.open(path) as im:rgb=np.asarray(im.convert('RGB').resize(size,Image.Resampling.BILINEAR))
    return np.clip(rgb@np.array([.299,.587,.114]),0,255).astype(np.uint8)
def backproject(P,uv,z):
    P=np.asarray(P); C=-np.linalg.solve(P[:,:3],P[:,3])
    return C+np.asarray(z)[...,None]*(np.c_[uv,np.ones(len(uv))]@np.linalg.inv(P[:,:3]).T)
def read_depth(path):
    with Path(path).open('rb') as f:
        fields=[];b=bytearray()
        while len(fields)<3:
            c=f.read(1)
            if not c:raise ValueError('Truncated depth header')
            if c==b'&':fields.append(int(b));b=bytearray()
            else:b.extend(c)
        w,h,n=fields;v=np.fromfile(f,dtype='<f4')
    assert v.size==w*h*n,(path,v.size,fields)
    return v.reshape((w,h,n),order='F').transpose(1,0,2).squeeze(2) if n==1 else v.reshape((w,h,n),order='F').transpose(1,0,2)

def prepare():
    assert socket.gethostname()=='liekkas' and pc.__version__=='4.2.1' and pc.has_cuda
    plan=load(OLD/'PLAN_CORRECTED.json');mapping=load(OLD/'CAMERA_MAPPING.json')
    files={};checks=[];jobs=[]
    for p in [OLD/'PLAN_CORRECTED.json',OLD/'CAMERA_MAPPING.json',OLD/'QUERY_MANIFEST.json',
              Path(__file__),ROOT/'ADDENDUM_PRE_RUN.md',ROOT/'AGENTS.md',ROOT/'refine-logs/EXPERIMENT_PLAN.md']:
        bind(files,p)
    for sid,s in plan['scenes'].items():
        names=[s['reference']]+s['Q'];ref=s['reference'];cams=s['cameras']
        interval=[559.,786.] if sid=='24' else [581.,865.]
        corners=np.array([[0,0],[776,0],[0,580],[776,580]],float)
        vertices=np.concatenate([backproject(cams[ref]['P'],corners,np.full(4,d)) for d in interval])
        ranges={ref:interval}
        for n in s['Q']:
            P=np.array(cams[n]['P']);zs=vertices@P[2,:3]+P[2,3]
            ranges[n]=[float(np.floor(zs.min())),float(np.ceil(zs.max()))]
            assert 0<ranges[n][0]<ranges[n][1]
        variants=['photo','geo']+(['photo_repeat'] if sid=='24' else [])
        for variant in variants:
            ws=ROOT/'workspaces'/f'scan{sid}'/variant
            for sub in ['images','sparse','stereo/depth_maps','stereo/normal_maps','stereo/consistency_graphs','configs']:
                (ws/sub).mkdir(parents=True,exist_ok=False)
            camlines=[];imlines=[]
            for i,n in enumerate(names,1):
                rec=mapping['scenes'][sid]['views'][n]
                K=np.array(rec['K_raw_colmap']);K[:2]*=.5;K[:2,2]-=.5
                R=np.array(rec['R_world_to_camera']);C=np.array(rec['center_mm']);t=-R@C
                P=K@np.column_stack([R,t]);expected=np.array(cams[n]['P'])
                assert np.max(np.abs(P-expected))<1e-7
                size=(cams[n]['width'],cams[n]['height']);a=gray(cams[n]['image_path'],size)
                bind(files,cams[n]['image_path']);Image.fromarray(a).save(ws/'images'/n)
                assert np.array_equal(a,np.array(Image.open(ws/'images'/n)))
                # Dense kernel's integer col,row ray, independently from stored P.
                uv=np.array([[0,0],[388,290],[776,580],[123,321]],float)
                z=np.array([600.,700.,800.,750.])
                local=np.c_[(uv[:,0]-K[0,2])/K[0,0],(uv[:,1]-K[1,2])/K[1,1],np.ones(4)]*z[:,None]
                world=local@R+C
                error=float(np.max(np.abs(world-backproject(expected,uv,z))))
                assert error<1e-8
                q=Rotation.from_matrix(R).as_quat();q=np.r_[q[3],q[:3]]
                camlines.append(f'{i} PINHOLE {size[0]} {size[1]} '+' '.join(format(x,'.17g') for x in [K[0,0],K[1,1],K[0,2],K[1,2]]))
                imlines.append(f'{i} '+' '.join(format(x,'.17g') for x in [*q,*t])+f' {i} {n}\n')
                checks.append(dict(scene=sid,variant=variant,image=n,ray_max_abs_mm=error,
                    K_mvs_array=K.tolist(),gray_sha256=hashlib.sha256(a.tobytes()).hexdigest()))
                (ws/'configs'/f'{n}.cfg').write_text(n+'\n'+', '.join(v for v in names if v!=n)+'\n')
            (ws/'sparse/cameras.txt').write_text('\n'.join(camlines)+'\n')
            (ws/'sparse/images.txt').write_text('\n'.join(imlines)+'\n')
            (ws/'sparse/points3D.txt').write_text('')
            recon=pc.Reconstruction(ws/'sparse');assert recon.num_reg_images()==5 and recon.num_points3D()==0
            # Round-tripped rotations/centres independent of text serialization.
            for im in recon.images.values():
                assert np.max(np.abs(im.projection_center()-np.array(cams[im.name]['center'])))<1e-8
            if variant=='geo':
                tasks=[(n,False,False) for n in names]+[(ref,True,True)]
            else:tasks=[(ref,False,True)]
            for n,geom,filtering in tasks:
                o=pc.PatchMatchOptions();o.gpu_index='0';o.cache_size=2.;o.num_threads=2
                o.depth_min,o.depth_max=ranges[n];o.geom_consistency=geom;o.filter=filtering
                jobs.append(dict(scene=int(sid),variant=variant,workspace=str(ws),reference=n,
                    options=o.todict(),config=str(ws/'configs'/f'{n}.cfg')))
        for roi in s['rois']:
            path=OLD/'initialization'/f'scan{sid}'/roi['id']/'input.npz';a=npz(path);bind(files,path)
            u=a['fixed_query_pixel_xy'];assert u.shape==(128,2) and np.array_equal(u,u.astype(int))
    for p in (ROOT/'workspaces').rglob('*'):
        if p.is_file():bind(files,p)
    dump(ROOT/'RUN_LOCK.json',dict(version=pc.__version__,cuda=pc.has_cuda,files=files,jobs=jobs,
        checks=checks,gt_accessed=False,seed='GPU fixed thread-index initializer; no exposed seed',depth_units='mm'))
    print('PREPARED',len(jobs),'jobs',len(checks),'ray/image checks',flush=True)

def execute():
    lock=load(ROOT/'RUN_LOCK.json')
    for p,h in lock['files'].items():assert sha(p)==h,p
    for idx,j in enumerate(lock['jobs']):
        done=ROOT/'job_records'/f'{idx:02d}.json'
        if done.exists():continue
        # User desktop is not a research compute job. Never terminate other jobs.
        status=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
        other=[x for x in status.splitlines() if x.strip() and 'chrome' not in x and int(x.split(',',1)[0])!=os.getpid()]
        if other:raise RuntimeError('Unrelated GPU job present: '+str(other))
        tick=time.monotonic();print('JOB',idx,j,flush=True)
        pc.patch_match_stereo(workspace_path=j['workspace'],options=pc.PatchMatchOptions(j['options']),config_path=j['config'])
        kind='geometric' if j['options']['geom_consistency'] else 'photometric'
        paths=[Path(j['workspace'])/'stereo'/folder/(j['reference']+'.'+kind+'.bin') for folder in ['depth_maps','normal_maps']]
        arrays=[read_depth(p) for p in paths]
        assert arrays[0].shape==(581,777) and arrays[1].shape==(581,777,3)
        dump(done,dict(job=idx,seconds=time.monotonic()-tick,files={str(p):sha(p) for p in paths},
            valid_depths=int((arrays[0]>0).sum()),options=j['options']))
        print('DONE',idx,round(time.monotonic()-tick,2),flush=True)
    extract()

def extract():
    plan=load(OLD/'PLAN_CORRECTED.json')
    for sid,s in plan['scenes'].items():
        P=np.array(s['cameras'][s['reference']]['P'])
        for roi in s['rois']:
            a=npz(OLD/'initialization'/f'scan{sid}'/roi['id']/'input.npz');uv=a['fixed_query_pixel_xy']
            values=dict(pixel_xy=uv,query_old_ids=np.arange(128))
            for arm,variant,kind in [('photo','photo','photometric'),('geo','geo','geometric'),('geo_unfiltered','geo','photometric')]+([('photo_repeat','photo_repeat','photometric')] if sid=='24' else []):
                d=read_depth(ROOT/'workspaces'/f'scan{sid}'/variant/'stereo/depth_maps'/(s['reference']+'.'+kind+'.bin'))
                z=d[uv[:,1].astype(int),uv[:,0].astype(int)].astype(float)
                valid=np.isfinite(z)&(z>0);xyz=backproject(P,uv,z);xyz[~valid]=np.nan
                values[arm+'_depth_mm']=z;values[arm+'_valid']=valid;values[arm+'_xyz_mm']=xyz
            folder=ROOT/'predictions';folder.mkdir(exist_ok=True)
            with (folder/(roi['id']+'.npz')).open('xb') as f:np.savez_compressed(f,**values)
    files={}
    for sub in ['predictions','job_records','workspaces']:
        for p in (ROOT/sub).rglob('*'):
            if p.is_file():files[str(p.relative_to(ROOT))]=sha(p)
    for n in ['RUN_LOCK.json','run_colmap.py','ADDENDUM_PRE_RUN.md','evaluate_mvs.py']:
        files[n]=sha(ROOT/n)
    dump(ROOT/'PREDICTIONS_SEALED.json',dict(files=files,gt_accessed=False))
    print('SEALED',len(files),'files',flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);v=a.parse_args()
    assert socket.gethostname()=='liekkas'
    (prepare if v.action=='prepare' else execute)()
