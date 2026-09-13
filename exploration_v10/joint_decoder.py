"""V10 follow-up: refit frame offsets rather than inherit corrected heights.

Same local normal and output support; jointly estimate shared tilt, per-frame
centers and one/two surfaces. Spectral and free-gap methods use the same backend.
"""
import time
import numpy as np
from scipy.special import logsumexp
from spectral_filter import estimate_gap

def solve(design,y,scans,sigma,gap=None,force_k=None,iterations=24,initial_y=None):
    _,s=np.unique(scans,return_inverse=True);n=len(y);f=s.max()+1
    counts=np.bincount(s);a=np.column_stack((np.eye(f)[s],design[:,1:]))
    beta=np.linalg.lstsq(a,y,rcond=1e-12)[0]
    def package(beta,delta,groups,score):
        center=float(counts@beta[:f]/n)
        return dict(k=1 if delta==0 else 2,gap=delta,
            prediction=center+design[:,1:]@beta[f:]+(groups-.5)*delta,
            groups=groups,center=center,bias=beta[:f]-center,slope=beta[f:],score=score)
    one=package(beta,0.,np.zeros(n,int),float(np.sum(((y-a@beta)/sigma)**2)+(f+2)*np.log(n)))
    if force_k==1:return one
    candidates=[]
    for seed_y in (y-a@beta,initial_y if initial_y is not None else y-a@beta):
        for q in ((.2,.8),(.1,.9)):
            delta=gap if gap is not None else float(np.diff(np.quantile(seed_y,q))[0])
            # Initialize frame centers after removing an initial layer assignment.
            g=(seed_y>np.median(seed_y)).astype(float)
            beta=np.linalg.lstsq(a,y-(g-.5)*delta,rcond=1e-12)[0]
            pi=np.full((f,2),.5)
            for _ in range(iterations):
                pred=(a@beta)[:,None]+np.array([-.5,.5])*delta
                lp=-.5*((y[:,None]-pred)/sigma)**2+np.log(pi[s])
                r=np.exp(lp-logsumexp(lp,axis=1)[:,None])
                for j in range(f):
                    pi[j]=np.maximum(r[s==j].mean(0),.001);pi[j]/=pi[j].sum()
                if gap is not None:
                    beta=np.linalg.lstsq(a,y-(r[:,1]-.5)*delta,rcond=1e-12)[0]
                else:
                    matrix=np.column_stack((np.repeat(a,2,axis=0),np.tile([-.5,.5],n)))
                    w=np.sqrt(r.ravel())
                    fitted=np.linalg.lstsq(matrix*w[:,None],np.repeat(y,2)*w,rcond=1e-12)[0]
                    beta=fitted[:-1];delta=float(fitted[-1])
                    if delta<0:delta=-delta;pi=pi[:,::-1]
            pred=(a@beta)[:,None]+np.array([-.5,.5])*delta
            lp=-.5*((y[:,None]-pred)/sigma)**2+np.log(pi[s])
            score=float(-2*logsumexp(lp,axis=1).sum()+(2*f+3)*np.log(n))
            candidates.append(package(beta,delta,np.argmax(lp,axis=1),score))
    best=min(candidates,key=lambda m:m['score'])
    return best if force_k==2 or best['score']<one['score'] else one

def filter_frozen(state,sigma_mm,method='spectrum_fixed',iterations=24):
    start=time.perf_counter();y=state['local'][:,2];scan=state['scan_id'][state['order']]
    spectrum=None;extra={}
    if method=='spectrum_fixed':spectrum,extra=estimate_gap(y,scan,sigma_mm)
    model=solve(state['design'],y,scan,sigma_mm,
        gap=spectrum['gap_mm'] if spectrum and spectrum['k']==2 else None,
        force_k=spectrum['k'] if spectrum else None,iterations=iterations,initial_y=state['corrected'])
    output=state['world'].copy();take=np.flatnonzero(state['support']);rows=state['order'][take]
    output[rows]+=(model['prediction'][take]-y[take])[:,None]*state['normal']/1000.
    support=np.zeros(len(y),bool);support[state['order']]=state['support']
    groups=np.empty(len(y),int);groups[state['order']]=model['groups']
    pred=np.empty(len(y));pred[state['order']]=model['prediction']
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(output[~support],state['world'][~support])
    info=dict(method=method,iterations=iterations,k=model['k'],fitted_gap_mm=model['gap'],
        spectrum=spectrum,score=model['score'],truth_fields_used=[],fitted_rows=len(y),
        supported_fraction=float(support.mean()),seconds=time.perf_counter()-start,
        model_scope='joint raw-height frame centers and shared tilt; weighted zero-mean bias gauge')
    return output,info,dict(xyz_world=output,support_mask=support,group_ids=groups,
        prediction_world_order_mm=pred,means_mm=model['center']+np.array([-.5,.5])*model['gap'],
        slope=model['slope'],bias_mm=model['bias'],**extra)
