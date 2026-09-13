"""Analytic ruled-parabola geometry stress, not simulated physical lidar rays."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from generate_synthetics import make_ghost,make_dual
from schema import fill_rays

def make(seed,gap,bias,amplitude):
    points,meta,ev=make_ghost(seed,bias) if gap==0 else make_dual(seed,gap,bias)
    a=amplitude/3600.
    for array in (points['xyz_world'],ev['gt_clean_xyz_world']):array[:,2]+=a*(array[:,0]*1000)**2/1000
    points['xyz_local']=points['xyz_world']-points['scanner_origin_world'];fill_rays(points)
    meta.update(curvature_amplitude_mm=amplitude,parabola_a_per_mm=a,family='warped_ruled_parabola',
        geometry_note='Parameter-space warp of finite patches; recomputed rays are derived metadata, not physical first-return simulation.')
    ev['parabola_a_per_mm']=a
    return points,meta,ev

def scores(out,ev):
    q=out*1000;a=float(ev['parabola_a_per_mm']);best=np.full(len(q),np.inf)
    for c,xmin,xmax,ymin,ymax in ev['surface_rectangles_mm']:
        if a==0:u=np.clip(q[:,0],xmin,xmax)
        else:
            # derivative of squared distance: 2a²u³+(1+2a(c-z))u-x=0
            b=1+2*a*(c-q[:,2]);u=q[:,0].copy();good=b>0
            for _ in range(20):u[good]-=(2*a*a*u[good]**3+b[good]*u[good]-q[good,0])/(6*a*a*u[good]**2+b[good])
            u=np.clip(u,xmin,xmax)
            for i in np.flatnonzero(~good):
                roots=np.roots([2*a*a,0.,b[i],-q[i,0]])
                opts=[xmin,xmax]+[float(r.real) for r in roots if abs(r.imag)<1e-8 and xmin<=r.real<=xmax]
                u[i]=min(opts,key=lambda x:(x-q[i,0])**2+(a*x*x+c-q[i,2])**2)
        v=np.clip(q[:,1],ymin,ymax);d=(u-q[:,0])**2+(v-q[:,1])**2+(a*u*u+c-q[:,2])**2
        best=np.minimum(best,np.sqrt(d))
    return dict(surface_accuracy_mean_mm=float(best.mean()),surface_accuracy_p95_mm=float(np.quantile(best,.95)),
        matched_point_rms_mm=float(np.sqrt(np.mean(np.sum((out-ev['gt_clean_xyz_world'])**2,axis=1)))*1000))
