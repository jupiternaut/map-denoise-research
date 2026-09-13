"""Cheap same-information staged normal estimation, then frozen scalar filtering.

This is a baseline, not another joint normal/layer optimizer. Regression can
mistake layer/XY correlation for inclination; keep that outcome in comparisons.
"""
import importlib.util
from pathlib import Path
import sys
import numpy as np

def _load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
_scalar=_load(Path(__file__).with_name('scalar_reference.py'),'_staged_frozen_scalar')
_old=_load(Path('/home/grf/Documents/Codex/2026-09-11/map-denoise-six-track-v1/tracks/t1_t2/experiment.py'),'_staged_frozen_old')
METHODS=('within_frame_normal_then_profile','within_frame_normal_then_old')

def estimate(inp,method):
    if method not in METHODS:raise KeyError(method)
    xyz,f,target,anchor,nf=_scalar._data(inp)
    xy=xyz[target,:2]; y=xyz[target,2];tf=f[target]
    count=np.bincount(tf,minlength=nf)
    xmean=np.column_stack([np.bincount(tf,weights=xy[:,j],minlength=nf)/count for j in range(2)])
    ymean=np.bincount(tf,weights=y,minlength=nf)/count
    slope=np.linalg.lstsq(xy-xmean[tf],y-ymean[tf],rcond=None)[0]
    normal=np.r_[-slope,1.];normal/=np.linalg.norm(normal)
    axis=np.eye(3)[int(np.argmin(abs(normal)))]
    u=np.cross(axis,normal);u/=np.linalg.norm(u);v=np.cross(normal,u)
    R=np.vstack((u,v,normal))
    center=xyz.mean(0);local=(xyz-center)@R.T
    # R is orthogonal: no metric scale change. Sigma is passed unchanged as the
    # provided approximate normal noise. Unknown anisotropic covariance cannot
    # be exactly transformed from a scalar sigma alone; no GT is used to do so.
    transformed=type(inp)(xyz_mm=local,frame=inp.frame.copy(),roi=inp.roi.copy(),
                          sigma_mm=inp.sigma_mm,bias_bound_mm=inp.bias_bound_mm)
    if method=='within_frame_normal_then_profile':
        denoised,b,detail=_scalar.estimate(transformed,'scalar_profile_hard')
    else:denoised,b,detail=_old.estimate(transformed,'joint_forced')
    output=denoised@R+center
    info={**detail,'method':method,'normal':normal.tolist(),'slope_xy':slope.tolist(),
      'estimated_normal_then_frozen_scalar':True,
      'stage_method':'scalar_profile_hard' if method.endswith('profile') else 'joint_forced',
      'rotation_orthogonality_error':float(np.max(abs(R@R.T-np.eye(3)))),
      'sigma_policy':'unchanged under isometric rotation; approximate normal scalar, no full anisotropic covariance available',
      'normal_estimator':'target within-frame demeaned ordinary least squares; no layer labels',
      'scope':'cheap staged reference, not joint slope/layer estimation; can confound layer position with x/y'}
    return output,b,info
