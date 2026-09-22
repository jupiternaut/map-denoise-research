"""Input-only DTU adapter. No evaluator paths or reference geometry access."""
from pathlib import Path
import hashlib,json,struct
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
def read(f,fmt):
    return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))
def project(p,P):
    c=np.column_stack([p,np.ones(len(p))])@P.T
    return np.divide(c[:,:2],c[:,2,None],out=np.full((len(p),2),np.nan),where=abs(c[:,2,None])>1e-12),c[:,2]
def colmap(folder):
    cameras={};views={};points={}
    with (folder/'cameras.bin').open('rb') as f:
        for _ in range(read(f,'Q')[0]):
            i,model,w,h=read(f,'iiQQ')
            if model not in (0,1):raise ValueError('Unsupported camera model '+str(model))
            p=read(f,'ddd' if model==0 else 'dddd')
            fx,fy,cx,cy=(p[0],p[0],p[1],p[2]) if model==0 else p
            cameras[i]=(np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]),w,h)
    with (folder/'images.bin').open('rb') as f:
        for _ in range(read(f,'Q')[0]):
            a=read(f,'idddddddi');name=bytearray()
            while True:
                b=f.read(1)
                if b==b'\0':break
                if not b:raise EOFError('image name')
                name.extend(b)
            n=read(f,'Q')[0];tracks=np.frombuffer(f.read(n*24),dtype=[('x','<f8'),('y','<f8'),('id','<i8')]).copy()
            R=Rotation.from_quat([a[2],a[3],a[4],a[1]]).as_matrix();t=np.asarray(a[5:8]);K,w,h=cameras[a[8]]
            views[name.decode()]=dict(P=K@np.column_stack([R,t]),R=R,t=t,K=K,width=w,height=h,tracks=tracks)
    with (folder/'points3D.bin').open('rb') as f:
        for _ in range(read(f,'Q')[0]):
            a=read(f,'QdddBBBd');n=read(f,'Q')[0];f.seek(n*8,1);points[a[0]]=a[1:4]
    return views,points
def load_scene(root=None, *, spec=None):
    if spec is None:
        root=Path(root);spec=dict(images=root/'images',colmap=root/'sparse/0',cameras=root/'cameras.npz',mesh=root/'mesh.ply')
    spec={k:Path(v) for k,v in spec.items()}
    with np.load(spec['cameras']) as camera_file:
        scale=camera_file['scale_mat_0'].astype(float);inv=np.linalg.inv(scale)
        if not np.allclose(scale[:3,:3],np.eye(3)*scale[0,0],atol=1e-8) or scale[0,0]<=0:raise ValueError('nonisotropic scale')
        files=sorted(spec['images'].glob('*.png'))
        if [int(p.stem) for p in files]!=list(range(49)):raise ValueError('expected 49 indexed images')
        raw,sparse=colmap(spec['colmap'])
        if set(raw)!={p.name for p in files}:raise ValueError('image names disagree')
        views={};errors=[];centers=[];roundtrip=[]
        for path in files:
            v=raw[path.name];P=v['P']@inv;c=-np.linalg.solve(P[:,:3],P[:,3])
            world=camera_file[f'world_mat_{int(path.stem)}']
            c_world=-np.linalg.solve(world[:3,:3],world[:3,3])
            centers.append(float(np.linalg.norm((c_world-scale[:3,3])/scale[0,0]+v['R'].T@v['t'])))
            with Image.open(path) as im:
                if im.size!=(v['width'],v['height']):raise ValueError('image size mismatch')
            tr=v['tracks'];keep=np.array([int(x) in sparse for x in tr['id']]);tr=tr[keep]
            if len(tr):
                xyz=np.array([sparse[int(x)] for x in tr['id']]);uv,z=project(xyz,v['P'])
                errors.extend(np.linalg.norm(uv-np.column_stack([tr['x'],tr['y']]),axis=1))
                physical=xyz@scale[:3,:3].T+scale[:3,3]
                uv2,_=project(physical,P);roundtrip.append(float(np.max(abs(uv-uv2))))
            views[path.name]=dict(name=path.name,P=P,center=c,width=v['width'],height=v['height'],path=path)
        if not errors or max(centers)>=1e-4 or np.quantile(errors,.95)>5:raise ValueError('calibration audit failed')
    mesh=o3d.io.read_triangle_mesh(str(spec['mesh']))
    p=np.asarray(mesh.vertices).copy();p=p@scale[:3,:3].T+scale[:3,3]
    if len(p)<1000 or not np.isfinite(p).all():raise ValueError('invalid mesh')
    audit=dict(normalized_camera_center_error_max=max(centers),sparse_reprojection_median_px=float(np.median(errors)),
               sparse_reprojection_p95_px=float(np.quantile(errors,.95)),physical_projection_roundtrip_max_px=max(roundtrip),
               scale_factor=float(scale[0,0]),vertices=len(p),used_reference=False,
               files={str(path):sha(path) for path in [spec['cameras'],spec['mesh'],*sorted(spec['colmap'].glob('*.bin'))]})
    return dict(points=p,views=views,scale=scale,audit=audit)
def in_box(p,lo,hi):return np.all((p>=lo)&(p<=hi),axis=1)
def select_views(scene,roi):
    scores=[];p=scene['points'];inside=p[in_box(p,np.array(roi['lo']),np.array(roi['hi']))]
    for name,v in scene['views'].items():
        uv,z=project(np.array([roi['centroid']]),v['P']);u,y=uv[0]
        if not np.isfinite(uv).all() or z[0]<=0 or u<8 or y<8 or u>=v['width']-8 or y>=v['height']-8:continue
        uv,z=project(inside,v['P']);ok=np.isfinite(uv).all(1)&(z>0)&(uv[:,0]>=0)&(uv[:,0]<v['width'])&(uv[:,1]>=0)&(uv[:,1]<v['height'])
        if ok.sum()>=20:scores.append((int(ok.sum()),name))
    scores.sort(reverse=True);preferred=[roi['reference']] if any(n==roi['reference'] for _,n in scores) else []
    names=preferred+[n for _,n in scores if n not in preferred]
    if len(names)<5 or not preferred:raise ValueError('reference/four sources unsupported')
    return names[:5]
def camera(v):
    with Image.open(v['path']) as image:
        rgb=np.asarray(image.convert('RGB'));w,h=v['width']//2,v['height']//2
        a=np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR),float)/255.
    sx,sy=w/v['width'],h/v['height'];resize=np.array([[sx,0,(sx-1)/2],[0,sy,(sy-1)/2],[0,0,1.]])
    return dict(image=a@np.array([.299,.587,.114]),P=resize@v['P'],center=v['center'])
def rois(scene,sid,out):
    p=scene['points'];v=scene['views']['0022.png'];uv,z=project(p,v['P'])
    valid=np.isfinite(uv).all(1)&(z>0)&(uv[:,0]>=0)&(uv[:,0]<v['width'])&(uv[:,1]>=0)&(uv[:,1]<v['height'])
    low,high=np.quantile(uv[valid],[.05,.95],axis=0);span=high-low
    result=[]
    for i,center in enumerate(((.3,.3),(.7,.3),(.3,.7),(.7,.7))):
        a=low+(np.array(center)-.16)*span;b=low+(np.array(center)+.16)*span
        ids=np.flatnonzero(valid&(uv[:,0]>=a[0])&(uv[:,0]<b[0])&(uv[:,1]>=a[1])&(uv[:,1]<b[1]))
        item=dict(id=f'scan{sid}_roi{i}',index=i,box=[*a.tolist(),*b.tolist()],n=len(ids),reference='0022.png')
        if len(ids)<200:item['status']='INSUFFICIENT_INPUT'
        else:
            q=p[ids];item.update(status='READY',lo=(q.min(0)-10).tolist(),hi=(q.max(0)+10).tolist(),centroid=q.mean(0).tolist())
            item['views']=select_views(scene,item)
        np.save(out/f'roi{i}_native_ids.npy',ids);result.append(item)
    with Image.open(v['path']) as original:
        image=original.convert('RGB');draw=ImageDraw.Draw(image)
        for r in result:
            draw.rectangle(r['box'],outline='cyan',width=3);draw.text(tuple(r['box'][:2]),r['id'],fill='red')
        image.save(out/'ROIS.png')
    save(out/'ROIS.json',result)
    return result
def write_ply(path,p):
    if Path(path).exists():raise FileExistsError(path)
    cloud=o3d.geometry.PointCloud();cloud.points=o3d.utility.Vector3dVector(p)
    if not o3d.io.write_point_cloud(str(path),cloud,write_ascii=False):raise RuntimeError('PLY write failed')
