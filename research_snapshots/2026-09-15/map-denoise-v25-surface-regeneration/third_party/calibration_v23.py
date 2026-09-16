"""Minimal COLMAP binary reader with name, image-size and reprojection checks.

Format: https://colmap.github.io/format.html ; no optimization against evaluator GT.
"""
import struct
import numpy as np
from scipy.spatial.transform import Rotation
def read(f,fmt):return struct.unpack('<'+fmt,f.read(struct.calcsize('<'+fmt)))
def load(folder):
    cameras={};images={};points={}
    with (folder/'cameras.bin').open('rb') as f:
        for _ in range(read(f,'Q')[0]):
            i,model,w,h=read(f,'iiQQ');assert model in (0,1),model
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
            P=K@np.column_stack([R,t]);images[name.decode()]=dict(P=P,R=R,t=t,K=K,width=w,height=h,tracks=tracks)
    with (folder/'points3D.bin').open('rb') as f:
        for _ in range(read(f,'Q')[0]):
            a=read(f,'QdddBBBd');n=read(f,'Q')[0];f.seek(n*8,1);points[a[0]]=a[1:4]
    return images,points
