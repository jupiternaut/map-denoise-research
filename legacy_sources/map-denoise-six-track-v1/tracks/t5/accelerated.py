"""Allocation-reduced compiled EM and optional analytic rejection pre-screen.

Reference Python is imported read-only into isolated module namespaces.
Equivalence means tested float64 tolerance and branch agreement, not bit equality.
"""
from __future__ import annotations
import ctypes
import importlib.util
from pathlib import Path
import numpy as np

OLD = Path('/home/grf/Documents/Codex/2026-09-10/map-denoise-v0/parallel_geometry_v5/association/operator.py')
ROOT = Path(__file__).resolve().parent

def load_reference(name='t5_reference'):
    spec=importlib.util.spec_from_file_location(name, OLD)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

REFERENCE=load_reference()

class FusedEM:
    def __init__(self, library, screen=False, boundary_tolerance=1e-8):
        self.library=ctypes.CDLL(str(library)); self.screen=screen
        self.boundary_tolerance=boundary_tolerance
        self.kernel=self.library.fit_rows
        ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
        self.kernel.argtypes=[ptr,ptr,ptr,ptr,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr,ptr,ptr,ptr]
        self.kernel.restype=None
        self.stats=[]

    def __call__(self, residual, variance_floor, config=None):
        cfg=REFERENCE.DEFAULTS | (config or {})
        x=np.ascontiguousarray(residual,dtype=float)
        if x.ndim==1: x=x[None,:]
        if x.ndim!=2 or x.shape[1]<2 or not np.isfinite(x).all(): raise ValueError('finite N x K required')
        n,k=x.shape
        floor=np.ascontiguousarray(np.broadcast_to(np.asarray(variance_floor,dtype=float),(n,)))
        if np.any(floor<=0) or not np.isfinite(floor).all(): raise ValueError('positive finite floor required')
        mean=x.mean(1); onevar=np.maximum(np.mean((x-mean[:,None])**2,axis=1),floor)
        one_ll=np.sum(-.5*np.log(2*np.pi*onevar)[:,None]-.5*(x-mean[:,None])**2/onevar[:,None],axis=1)
        # Every mixture density <= 1/sqrt(2*pi*floor), independently of EM.
        gain_upper=2*(-.5*k*np.log(2*np.pi*floor)-one_ll)-2*np.log(k)
        # Updated means lie in the observed range, and variance >= floor.
        separation_upper=np.ptp(x,axis=1)/np.sqrt(floor)
        excluded=self.screen & ((gain_upper<cfg['bic_gain']-1e-8) |
                   (separation_upper<cfg['separation_sigma']-1e-8))
        excluded=np.broadcast_to(excluded,(n,)).copy()
        active=~excluded; nx=int(active.sum())
        # Excluded-row diagnostics are explicitly placeholders; choose_two=False is certified.
        result={'means':np.repeat(mean[:,None],2,axis=1),'variance':onevar.copy(),
                'weights':np.full((n,2),.5),'responsibility':np.full((n,k,2),.5),
                'one_loglik':one_ll,'two_loglik':one_ll.copy()}
        if nx:
            xx=np.ascontiguousarray(x[active]); ff=np.ascontiguousarray(floor[active])
            initial=np.ascontiguousarray(np.stack([np.quantile(xx,q,axis=1).T for q in ((.25,.75),(.1,.9))]))
            means=np.empty((nx,2)); variance=np.empty(nx); pi=np.empty((nx,2)); resp=np.empty((nx,k,2)); ll=np.empty(nx)
            self.kernel(xx,ff,initial,np.ascontiguousarray(onevar[active]),nx,k,cfg['em_iterations'],means,variance,pi,resp,ll)
            for key,value in [('means',means),('variance',variance),('weights',pi),('responsibility',resp),('two_loglik',ll)]: result[key][active]=value
        result['bic_gain']=2*(result['two_loglik']-one_ll)-2*np.log(k)
        result['separation_sigma']=np.abs(result['means'][:,1]-result['means'][:,0])/np.sqrt(result['variance'])
        support=result['responsibility'].sum(1).min(1)
        threshold=max(cfg['min_count'],cfg['min_fraction']*k)
        boundary=active & ((np.abs(result['bic_gain']-cfg['bic_gain'])<self.boundary_tolerance) |
                    (np.abs(result['separation_sigma']-cfg['separation_sigma'])<self.boundary_tolerance) |
                    (np.abs(support-threshold)<self.boundary_tolerance))
        if boundary.any():
            exact=REFERENCE.fit_modes(x[boundary],floor[boundary],cfg)
            for key in result: result[key][boundary]=exact[key]
        support=result['responsibility'].sum(1).min(1)
        result['choose_two']=(active & (result['bic_gain']>=cfg['bic_gain']) &
              (result['separation_sigma']>=cfg['separation_sigma']) & (support>=threshold))
        self.stats.append({'rows':n,'screened':int(excluded.sum()),'boundary_reference_rows':int(boundary.sum())})
        return result

def traced_denoise(points, config, fit=None):
    module=load_reference('t5_trace')
    modes_trace=[]; planes_trace=[]
    original_planes=module._planes
    def planes(*args):
        result=original_planes(*args); planes_trace.append(result); return result
    fn=fit or module.fit_modes
    def modes(*args):
        result=fn(*args); modes_trace.append(result); return result
    module._planes=planes; module.fit_modes=modes
    output,diag=module.denoise(points,config)
    cfg=diag['config']; floor=max(diag['observed_spacing']*cfg['variance_floor_spacing'],np.finfo(float).tiny**.25)
    masks=[]
    for j,m in enumerate(modes_trace):
        (_,n,v),(_,n0,v0),(_,n1,v1)=planes_trace[3*j:3*j+3]
        good=v[:,1]>np.maximum(v[:,2]*1e-10,floor**2*1e-5)
        stable=(v0[:,1]>np.maximum(v0[:,2]*1e-10,floor**2*1e-5)) & (v1[:,1]>np.maximum(v1[:,2]*1e-10,floor**2*1e-5))
        agreement=np.abs(np.einsum('ni,ni->n',n0,n1))
        masks.append({'choose_two':m['choose_two'], 'split':m['choose_two'] & good & stable & (agreement>=cfg['normal_agreement'])})
    return output,diag,masks

def make_denoiser(fit=None):
    module=load_reference('t5_untraced')
    if fit is not None: module.fit_modes=fit
    return module.denoise
